"""Runner reuse across search settings, with fixed small-graph expectations."""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from cddd.algorithms import PCStable
from cddd.independence import CITester


class TableTester(CITester):
    def __init__(self, fail_once=False):
        super().__init__()
        self.trace = []
        self.fail_once = fail_once
        self.has_failed = False

    def reset_for_data(self, data):
        super().reset_for_data(data)
        self.trace.clear()

    def _compute_p_value(self, data, x, y, cond_set):
        key = tuple(sorted((x, y))), tuple(sorted(cond_set))
        self.trace.append(key)
        if self.fail_once and not self.has_failed and len(self.trace) == 2:
            self.has_failed = True
            raise RuntimeError("interrupted CI computation")
        return (1.0 if key in data.attrs["independent"] else 0.0), 0.0


def table_data(variables, independent=()):
    data = pd.DataFrame(np.zeros((4, variables)))
    data.attrs["independent"] = set(independent)
    return data


# CI tables for faithful directed chains, plus a three-node collider. These
# answers and skeleton edges are fixed independently of either runner's output.
CHAIN_FOUR = {
    ((0, 2), (1,)), ((0, 2), (1, 3)),
    ((0, 3), (1,)), ((0, 3), (2,)), ((0, 3), (1, 2)),
    ((1, 3), (2,)), ((1, 3), (0, 2)),
}
CHAIN_THREE = {((0, 2), (1,))}
COLLIDER_THREE = {((0, 2), ())}


def assert_expected_graph(result, variables, edges):
    expected = np.zeros((variables, variables), dtype=int)
    for x, y in edges:
        expected[x, y] = expected[y, x] = 1
    np.testing.assert_array_equal(result[0], expected)


def assert_same_run_state(actual, fresh):
    assert actual.total_pc_requests == fresh.total_pc_requests
    assert actual.ci_tester.n_actual_calls == fresh.ci_tester.n_actual_calls
    assert actual.ci_tester.history == fresh.ci_tester.history
    assert actual.ci_tester.trace == fresh.ci_tester.trace
    assert actual.early_stopping == fresh.early_stopping
    if actual.use_deduction:
        assert actual.deductor.priority == fresh.deductor.priority
        assert actual.deductor.pure == fresh.deductor.pure
        assert actual.deductor.cache == fresh.deductor.cache
        assert actual.deductor.stats == fresh.deductor.stats
        assert actual.deductor.cit_provenance == fresh.deductor.cit_provenance
        assert actual.deductor._current_root_query is None


SETTINGS = list(product((False, True), (False, True), (False, True), ("dep", "ind")))


@pytest.mark.parametrize("use_deduction,early_stopping,pure,priority", SETTINGS)
def test_reused_runner_matches_fresh_across_search_settings(
        use_deduction, early_stopping, pure, priority):
    options = dict(use_deduction=use_deduction, early_stopping=early_stopping,
                   deduction_pure=pure, deduction_priority=priority)
    tester = TableTester()
    runner = PCStable(.05, tester, **options)
    deductor = runner.deductor
    # Populate higher-order queries before changing both answers and dimensions.
    runner.run(table_data(4))
    chain_data = table_data(4, CHAIN_FOUR)
    cases = (
        (chain_data, ((0, 1), (1, 2), (2, 3))),
        (chain_data, ((0, 1), (1, 2), (2, 3))),
        (table_data(3, CHAIN_THREE), ((0, 1), (1, 2))),
        (table_data(3, COLLIDER_THREE), ((0, 1), (1, 2))),
        (table_data(2, {((0, 1), ())}), ()),
        (table_data(2), ((0, 1),)),
    )
    for data, expected_edges in cases:
        fresh = PCStable(.05, TableTester(), **options)
        expected = fresh.run(data)
        actual = runner.run(data)

        assert_expected_graph(actual, data.shape[1], expected_edges)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1] == expected[1]
        assert runner.ci_tester is tester
        assert runner.deductor is deductor
        assert_same_run_state(runner, fresh)


@pytest.mark.parametrize("use_deduction,early_stopping,pure,priority", SETTINGS)
def test_failed_run_retry_matches_fresh_across_search_settings(
        use_deduction, early_stopping, pure, priority):
    options = dict(use_deduction=use_deduction, early_stopping=early_stopping,
                   deduction_pure=pure, deduction_priority=priority)
    tester = TableTester(fail_once=True)
    runner = PCStable(.05, tester, **options)
    with pytest.raises(RuntimeError, match="interrupted CI computation"):
        runner.run(table_data(4))
    assert tester.n_actual_calls == 2

    data = table_data(3, CHAIN_THREE)
    fresh = PCStable(.05, TableTester(), **options)
    expected = fresh.run(data)
    actual = runner.run(data)

    assert_expected_graph(actual, 3, ((0, 1), (1, 2)))
    assert actual[1] == {(0, 2): (1,)}
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1] == expected[1]
    assert_same_run_state(runner, fresh)
