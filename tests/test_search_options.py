"""Search-policy regressions, with fixed CI answers and an independent DAG oracle."""

import inspect
from itertools import combinations, permutations

import networkx as nx
import numpy as np
import pytest

from cddd.algorithms import PCStable


def query_key(x, y, conditioning_set):
    return tuple(sorted((x, y))), tuple(sorted(conditioning_set))


class FixedCITester:
    """Relabel a fixed answer table; record every call, including cache reuse."""

    def __init__(self, independent_queries, order):
        self.independent_queries = set(independent_queries)
        self.order = tuple(order)
        self.trace = []

    def ci_test(self, data, x, y, conditioning_set):
        key = query_key(self.order[x], self.order[y],
                        (self.order[z] for z in conditioning_set))
        self.trace.append(key)
        return (1.0 if key in self.independent_queries else 0.0), 0.0


# A four-variable finite-sample answer pattern. The six independent answers
# are the Fisher-Z decisions at alpha=.05 from the saved 100-sample seed-1036
# regression (retained columns 0, 1, 2, 4), not a claimed DAG CI oracle.
# Freeze decisions so this control-flow test does not depend on BLAS rounding.
ORDER_WITNESS = {
    ((0, 2), (1, 3)),
    ((0, 3), (1,)),
    ((0, 3), (2,)),
    ((0, 3), (1, 2)),
    ((1, 3), (2,)),
    ((1, 3), (0, 2)),
}


def run_fixed(independent_queries, order=(0, 1, 2, 3), **kwargs):
    tester = FixedCITester(independent_queries, order)
    runner = PCStable(.05, tester, **kwargs)
    graph, sepsets = runner.run(np.zeros((1, len(order))))
    return runner, graph, sepsets


def mapped_edges(graph, order):
    return frozenset(tuple(sorted((order[x], order[y])))
                     for x, y in combinations(range(len(order)), 2)
                     if graph[x, y])


def test_original_constructor_defaults_and_positional_arguments():
    parameters = inspect.signature(PCStable).parameters
    assert parameters['use_deduction'].default is False
    assert parameters['deduction_priority'].default == 'dep'
    assert parameters['deduction_pure'].default is True
    assert parameters['early_stopping'].default is True
    assert parameters['early_stopping'].kind is inspect.Parameter.KEYWORD_ONLY

    tester = FixedCITester(ORDER_WITNESS, range(4))
    runner = PCStable(.05, tester, True, 'ind', False)
    assert runner.early_stopping is True
    assert runner.deductor.priority == 'ind'
    assert runner.deductor.pure is False
    with pytest.raises(TypeError):
        PCStable(.05, tester, True, 'dep', True, False)


@pytest.mark.parametrize('use_deduction', [False, True])
@pytest.mark.parametrize('deduction_pure', [False, True])
@pytest.mark.parametrize('deduction_priority', ['dep', 'ind'])
def test_explicit_early_stopping_matches_default(
        use_deduction, deduction_pure, deduction_priority):
    kwargs = dict(use_deduction=use_deduction,
                  deduction_pure=deduction_pure,
                  deduction_priority=deduction_priority)
    default, expected, expected_sepsets = run_fixed(ORDER_WITNESS, **kwargs)
    explicit, actual, actual_sepsets = run_fixed(
        ORDER_WITNESS, early_stopping=True, **kwargs)
    np.testing.assert_array_equal(actual, expected)
    assert actual_sepsets == expected_sepsets
    assert explicit.total_pc_requests == default.total_pc_requests
    assert explicit.ci_tester.trace == default.ci_tester.trace
    if use_deduction:
        assert explicit.deductor.cache == default.deductor.cache
        assert explicit.deductor.stats == default.deductor.stats
        assert explicit.deductor.cit_provenance == default.deductor.cit_provenance


@pytest.mark.parametrize('use_deduction', [False, True])
def test_no_break_finishes_eligible_queries_before_level_end_removal(use_deduction):
    independent = {((0, 1), (2,)), ((0, 1), (3,))}
    stopping, _, stopping_sepsets = run_fixed(
        independent, use_deduction=use_deduction, early_stopping=True)
    exhaustive, graph, sepsets = run_fixed(
        independent, use_deduction=use_deduction, early_stopping=False)

    assert ((0, 1), (2,)) in stopping.ci_tester.trace
    assert ((0, 1), (3,)) not in stopping.ci_tester.trace
    assert ((0, 1), (3,)) in exhaustive.ci_tester.trace
    assert exhaustive.total_pc_requests > stopping.total_pc_requests
    assert graph[0, 1] == graph[1, 0] == 0
    assert stopping_sepsets[(0, 1)] == (2,)
    assert sepsets[(0, 1)] == (3,)
    # Both endpoint directions are still processed before deleting the edge.
    if not use_deduction:
        level_one = [key for key in exhaustive.ci_tester.trace
                     if key[0] == (0, 1) and len(key[1]) == 1]
        assert level_one == [((0, 1), (2,)), ((0, 1), (3,))] * 2
    # An edge deleted at level one is not queried at later levels.
    assert not any(pair == (0, 1) and len(z) > 1
                   for pair, z in exhaustive.ci_tester.trace)


@pytest.mark.parametrize('priority', ['dep', 'ind'])
def test_fixed_answer_order_witness(priority):
    signatures = {}
    for use_deduction, early_stopping in (
            (False, True), (False, False), (True, True), (True, False)):
        outcomes = set()
        for order in permutations(range(4)):
            _, graph, _ = run_fixed(
                ORDER_WITNESS, order, use_deduction=use_deduction,
                deduction_priority=priority, deduction_pure=True,
                early_stopping=early_stopping)
            outcomes.add(mapped_edges(graph, order))
        signatures[use_deduction, early_stopping] = outcomes

    pc_edges = frozenset({(0, 1), (1, 2), (2, 3)})
    assert signatures[False, True] == {pc_edges}
    assert signatures[False, False] == {pc_edges}
    assert len(signatures[True, True]) == 2
    assert len(signatures[True, False]) == 1
    # This finite answer-table regression is not a general order-invariance
    # theorem or an assertion that no-break improves accuracy or sepsets.


def d_separated(graph, x, y, conditioning_set):
    """Ancestral moralization oracle, independent of DF-PC's deduction rules."""
    conditioned = set(conditioning_set)
    ancestors = {x, y} | conditioned
    for node in tuple(ancestors):
        ancestors.update(nx.ancestors(graph, node))
    ancestral_graph = graph.subgraph(ancestors)
    moral_graph = ancestral_graph.to_undirected()
    for node in ancestral_graph:
        moral_graph.add_edges_from(combinations(ancestral_graph.predecessors(node), 2))
    moral_graph.remove_nodes_from(conditioned)
    return not nx.has_path(moral_graph, x, y)


def oracle_table(graph):
    independent = set()
    for x, y in combinations(graph.nodes, 2):
        remaining = sorted(set(graph.nodes) - {x, y})
        for size in range(len(remaining) + 1):
            for z in combinations(remaining, size):
                if d_separated(graph, x, y, z):
                    independent.add(query_key(x, y, z))
    return independent


def test_oracle_has_expected_chain_and_collider_answers():
    chain = nx.DiGraph([(0, 1), (1, 2)])
    collider = nx.DiGraph([(0, 1), (2, 1), (1, 3)])
    assert not d_separated(chain, 0, 2, ())
    assert d_separated(chain, 0, 2, (1,))
    assert d_separated(collider, 0, 2, ())
    assert not d_separated(collider, 0, 2, (1,))
    assert not d_separated(collider, 0, 2, (3,))


@pytest.mark.parametrize('early_stopping', [False, True])
@pytest.mark.parametrize('use_deduction', [False, True])
@pytest.mark.parametrize('deduction_pure', [False, True])
@pytest.mark.parametrize('priority', ['dep', 'ind'])
def test_small_dags_against_independent_oracle(
        early_stopping, use_deduction, deduction_pure, priority):
    possible_edges = list(combinations(range(4), 2))
    # All 64 DAGs compatible with this topological order, including the empty
    # graph, chains, forks, colliders, diamonds and the complete DAG.
    for mask in range(1 << len(possible_edges)):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(4))
        graph.add_edges_from(edge for bit, edge in enumerate(possible_edges)
                             if mask & (1 << bit))
        runner, actual, sepsets = run_fixed(
            oracle_table(graph), use_deduction=use_deduction,
            deduction_pure=deduction_pure, deduction_priority=priority,
            early_stopping=early_stopping)
        # Expected output comes directly from the specified DAG, not from
        # comparing two implementations that might share the same error.
        np.testing.assert_array_equal(
            actual, nx.to_numpy_array(graph.to_undirected(), nodelist=range(4)))
        missing_edges = {edge for edge in possible_edges if not graph.has_edge(*edge)}
        assert set(sepsets) == missing_edges
        for (x, y), z in sepsets.items():
            assert d_separated(graph, x, y, z)
        if use_deduction:
            for ((x, y), z), answer in runner.deductor.cache.items():
                if answer != 'unknown':
                    assert (answer == 'ind') == d_separated(graph, x, y, z)
