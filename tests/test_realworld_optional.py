"""Small sampling checks; skipped unless Experiment 6 dependencies are installed."""

import numpy as np
import pytest


pytest.importorskip("pgmpy")


@pytest.mark.parametrize("name,variables", [("barley", 48), ("mildew", 35)])
def test_example_network_can_be_sampled(name, variables):
    from pgmpy.utils import get_example_model
    from experiment_realworld import get_pgmpy_gt

    model = get_example_model(name)
    columns = sorted(model.nodes())
    state = np.random.get_state()
    try:
        np.random.seed(0)
        data = model.simulate(n_samples=8, show_progress=False)
    finally:
        np.random.set_state(state)
    assert len(columns) == variables
    assert data.shape == (8, variables)
    assert set(data.columns) == set(columns)
    assert not data.isna().any().any()
    adjacency = get_pgmpy_gt(model, columns)
    assert adjacency.shape == (variables, variables)
    assert int(adjacency.sum()) == model.number_of_edges()
