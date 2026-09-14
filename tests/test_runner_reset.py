"""Regressions for reusing PCStable runners with independent per-run state.

The first run uses the tester's existing initialization. Subsequent calls,
including retries after a failed call, must clear per-run state.
"""

import itertools
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from cddd.algorithms import PCStable
from cddd.independence import (
    CITester,
    ChiSquareTester,
    G2Tester,
    KernelCITest,
    PartialCorrelation,
)


class DataFlagTester(CITester):
    """Read explicit CI answers from the current input, without a backend cache."""

    def __init__(self):
        super().__init__()
        self.computations = 0

    def _compute_p_value(self, data, X, Y, cond_set):
        self.computations += 1
        p_value = 1.0 if data.attrs["independent"] else 0.0
        return p_value, -p_value


class InterruptedTester(DataFlagTester):
    def __init__(self):
        super().__init__()
        self.has_failed = False

    def _compute_p_value(self, data, X, Y, cond_set):
        if self.computations == 1 and not self.has_failed:
            self.has_failed = True
            raise RuntimeError("interrupted CI computation")
        return super()._compute_p_value(data, X, Y, cond_set)


class LegacyDuckTester:
    """A legacy tester that offers ci_test(), but no run-reset protocol."""

    def ci_test(self, data, X, Y, cond_set):
        p_value = 1.0 if data.attrs["independent"] else 0.0
        return p_value, -p_value


def flagged_data(independent, variables=2):
    data = pd.DataFrame(np.zeros((4, variables)))
    data.attrs["independent"] = independent
    return data


def fisher_z_data(dependent):
    # Centered, exactly orthogonal columns give an uncorrelated fixture.
    # Adding the first column gives a strong, nonsingular alternative.
    x = np.tile([-1.0, -1.0, 1.0, 1.0], 32)
    noise = np.tile([-1.0, 1.0, -1.0, 1.0], 32)
    y = 0.9 * x + 0.1 * noise if dependent else noise
    return pd.DataFrame({0: x, 1: y})


class RecordingBackend:
    """A cheap backend double for testing binding, separate from Fisher-Z tests."""

    def __init__(self, data, method):
        self.data = np.array(data, copy=True)
        self.method = method
        self.calls = []

    def __call__(self, X, Y, cond_set):
        self.calls.append((X, Y, tuple(cond_set)))
        return 1.0


class RunnerResetTests(unittest.TestCase):
    def assert_graph(self, result, dependent):
        graph, sepsets = result
        edge = int(dependent)
        np.testing.assert_array_equal(graph, [[0, edge], [edge, 0]])
        self.assertEqual(sepsets, {} if dependent else {(0, 1): ()})

    def test_df_runner_uses_current_input_after_previous_independence(self):
        tester = DataFlagTester()
        runner = PCStable(0.05, tester, use_deduction=True)
        self.assert_graph(runner.run(flagged_data(True)), dependent=False)
        computations = tester.computations

        result = runner.run(flagged_data(False))

        self.assert_graph(result, dependent=True)
        self.assertGreater(tester.computations, computations)
        self.assertEqual(tester.n_actual_calls, 1)

    def test_same_input_rerun_has_fresh_counts_time_and_deduction_stats(self):
        for use_deduction in (False, True):
            with self.subTest(use_deduction=use_deduction):
                data = flagged_data(False, variables=3)
                tester = DataFlagTester()
                runner = PCStable(0.05, tester, use_deduction=use_deduction)
                # Each CI computation takes one artificial second, so timing
                # accumulation is observable without wall-clock assumptions.
                with patch("cddd.independence.time.time", side_effect=itertools.count()):
                    graph, sepsets = runner.run(data)
                    initial_computations = tester.computations
                    initial_count = tester.n_actual_calls
                    initial_time = tester.total_test_time
                    initial_history = tester.history.copy()
                    initial_requests = runner.total_pc_requests
                    if use_deduction:
                        deductor = runner.deductor
                        initial_stats = deductor.stats.copy()
                        initial_cache = deductor.cache.copy()
                        initial_provenance = deductor.cit_provenance.copy()

                    repeated_graph, repeated_sepsets = runner.run(data)

                np.testing.assert_array_equal(repeated_graph, graph)
                self.assertEqual(repeated_sepsets, sepsets)
                self.assertEqual(tester.computations, 2 * initial_computations)
                self.assertEqual(tester.n_actual_calls, initial_count)
                self.assertEqual(tester.total_test_time, initial_time)
                self.assertEqual(tester.history, initial_history)
                self.assertEqual(runner.total_pc_requests, initial_requests)
                if use_deduction:
                    self.assertIs(runner.deductor, deductor)
                    self.assertEqual(deductor.stats, initial_stats)
                    self.assertEqual(deductor.cache, initial_cache)
                    self.assertEqual(deductor.cit_provenance, initial_provenance)
                    self.assertIsNone(deductor._current_root_query)

    def test_smaller_input_drops_old_queries_and_provenance(self):
        tester = DataFlagTester()
        runner = PCStable(0.05, tester, use_deduction=True)
        runner.run(flagged_data(True, variables=3))
        self.assertEqual(tester.n_actual_calls, 3)
        self.assertTrue(any(2 in key[0] for key in runner.deductor.cit_provenance))
        deductor = runner.deductor

        data = flagged_data(True)
        result = runner.run(data)
        fresh_tester = DataFlagTester()
        fresh_runner = PCStable(0.05, fresh_tester, use_deduction=True)
        fresh_result = fresh_runner.run(data)

        self.assert_graph(result, dependent=False)
        np.testing.assert_array_equal(result[0], fresh_result[0])
        self.assertEqual(result[1], fresh_result[1])
        self.assertEqual(tester.n_actual_calls, 1)
        self.assertEqual(tester.history, fresh_tester.history)
        self.assertIs(runner.deductor, deductor)
        self.assertEqual(deductor.cache, fresh_runner.deductor.cache)
        self.assertEqual(deductor.stats, fresh_runner.deductor.stats)
        self.assertEqual(deductor.cit_provenance, fresh_runner.deductor.cit_provenance)

    def test_interrupted_first_run_is_reset_before_retry(self):
        for use_deduction in (False, True):
            with self.subTest(use_deduction=use_deduction):
                tester = InterruptedTester()
                runner = PCStable(0.05, tester, use_deduction=use_deduction)
                with self.assertRaisesRegex(RuntimeError, "interrupted CI computation"):
                    runner.run(flagged_data(True, variables=3))
                self.assertEqual(tester.computations, 1)

                data = flagged_data(False)
                result = runner.run(data)
                fresh_tester = DataFlagTester()
                fresh_runner = PCStable(0.05, fresh_tester, use_deduction=use_deduction)
                fresh_result = fresh_runner.run(data)

                self.assert_graph(result, dependent=True)
                np.testing.assert_array_equal(result[0], fresh_result[0])
                self.assertEqual(result[1], fresh_result[1])
                self.assertEqual(tester.n_actual_calls, fresh_tester.n_actual_calls)
                self.assertEqual(tester.history, fresh_tester.history)
                self.assertEqual(runner.total_pc_requests, fresh_runner.total_pc_requests)
                if use_deduction:
                    self.assertEqual(runner.deductor.stats, fresh_runner.deductor.stats)
                    self.assertEqual(runner.deductor.cache, fresh_runner.deductor.cache)
                    self.assertEqual(
                        runner.deductor.cit_provenance, fresh_runner.deductor.cit_provenance
                    )
                    self.assertIsNone(runner.deductor._current_root_query)

    def test_legacy_tester_without_reset_support_has_explicit_reuse_error(self):
        for use_deduction in (False, True):
            with self.subTest(use_deduction=use_deduction):
                runner = PCStable(0.05, LegacyDuckTester(), use_deduction=use_deduction)
                data = flagged_data(True)
                self.assert_graph(runner.run(data), dependent=False)

                with self.assertRaisesRegex(TypeError, "reset_for_data|fresh"):
                    runner.run(data)

    def test_fisher_z_new_dataframe_matches_fresh_runner(self):
        for use_deduction in (False, True):
            with self.subTest(use_deduction=use_deduction):
                initial_data = fisher_z_data(dependent=False)
                tester = PartialCorrelation(data=initial_data)
                runner = PCStable(0.05, tester, use_deduction=use_deduction)
                self.assert_graph(runner.run(initial_data), dependent=False)
                initial_backend = tester.cl_cit
                new_data = fisher_z_data(dependent=True)
                fresh_runner = PCStable(
                    0.05, PartialCorrelation(data=new_data), use_deduction=use_deduction
                )
                fresh_result = fresh_runner.run(new_data)
                self.assert_graph(fresh_result, dependent=True)

                result = runner.run(new_data)

                self.assert_graph(result, dependent=True)
                np.testing.assert_array_equal(result[0], fresh_result[0])
                self.assertEqual(result[1], fresh_result[1])
                self.assertIs(tester.data, new_data)
                self.assertIsNot(tester.cl_cit, initial_backend)

    def test_fisher_z_in_place_mutation_matches_fresh_runner(self):
        for use_deduction in (False, True):
            with self.subTest(use_deduction=use_deduction):
                data = fisher_z_data(dependent=False)
                tester = PartialCorrelation(data=data)
                runner = PCStable(0.05, tester, use_deduction=use_deduction)
                self.assert_graph(runner.run(data), dependent=False)
                initial_backend = tester.cl_cit
                data.iloc[:, :] = fisher_z_data(dependent=True).to_numpy()
                fresh_runner = PCStable(
                    0.05, PartialCorrelation(data=data), use_deduction=use_deduction
                )
                fresh_result = fresh_runner.run(data)
                self.assert_graph(fresh_result, dependent=True)

                result = runner.run(data)

                self.assert_graph(result, dependent=True)
                np.testing.assert_array_equal(result[0], fresh_result[0])
                self.assertEqual(result[1], fresh_result[1])
                self.assertIs(tester.data, data)
                self.assertIsNot(tester.cl_cit, initial_backend)

    def test_data_bound_testers_refresh_backend_only_after_first_run(self):
        testers = (
            (G2Tester, "gsq"),
            (ChiSquareTester, "chisq"),
            (PartialCorrelation, "fisherz"),
            (KernelCITest, "kci"),
        )
        for tester_type, method in testers:
            for use_deduction in (False, True):
                with self.subTest(tester=tester_type.__name__, use_deduction=use_deduction):
                    with patch("cddd.independence.CIT", side_effect=RecordingBackend) as factory:
                        data = pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]])
                        tester = tester_type(data=data)
                        initial_backend = tester.cl_cit
                        runner = PCStable(0.05, tester, use_deduction=use_deduction)
                        self.assert_graph(runner.run(data), dependent=False)
                        # Experiment drivers initialize CIT before timing run().
                        self.assertIs(tester.cl_cit, initial_backend)
                        self.assertEqual(factory.call_count, 1)
                        self.assertEqual(initial_backend.method, method)
                        first_backend_calls = len(initial_backend.calls)

                        self.assert_graph(runner.run(data), dependent=False)
                        repeated_backend = tester.cl_cit
                        self.assertIsNot(repeated_backend, initial_backend)
                        self.assertEqual(factory.call_count, 2)
                        self.assertEqual(len(initial_backend.calls), first_backend_calls)
                        self.assertEqual(len(repeated_backend.calls), first_backend_calls)
                        self.assertEqual(tester.n_actual_calls, 1)

                        new_data = data.copy()
                        new_data[1] = [1, 1, 0, 0]
                        runner.run(new_data)
                        self.assertIs(tester.data, new_data)
                        self.assertIsNot(tester.cl_cit, repeated_backend)
                        self.assertEqual(factory.call_count, 3)
                        np.testing.assert_array_equal(tester.cl_cit.data, new_data.to_numpy())

                        previous_backend = tester.cl_cit
                        new_data.iloc[0, 0] = 7
                        runner.run(new_data)
                        self.assertIsNot(tester.cl_cit, previous_backend)
                        self.assertEqual(factory.call_count, 4)
                        np.testing.assert_array_equal(tester.cl_cit.data, new_data.to_numpy())


if __name__ == "__main__":
    unittest.main()
