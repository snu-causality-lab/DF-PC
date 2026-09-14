"""Exercise the user-facing example and imports without benchmark sweeps."""

import importlib
from pathlib import Path
import re

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_readme_example_is_self_contained():
    examples = re.findall(r"```python\n(.*?)\n```", (ROOT / "README.md").read_text(), re.S)
    assert examples, "README must contain an executable Python quickstart"
    namespace = {}
    exec(compile(examples[0], "README.md quickstart", "exec"), namespace)
    skeleton = namespace["skeleton"]
    assert skeleton.shape == (3, 3)
    np.testing.assert_array_equal(skeleton, skeleton.T)
    assert not np.diag(skeleton).any()
    assert namespace["runner"].use_deduction is True
    assert namespace["runner"].early_stopping is True
    assert namespace["runner"].deductor.pure is True
    assert namespace["runner"].deductor.priority == "dep"


@pytest.mark.parametrize("script", sorted(
    path.stem for path in ROOT.glob("experiment_*.py")
    if path.stem != "experiment_realworld"
))
def test_base_experiment_import(script, tmp_path, monkeypatch):
    # Importing a driver must not start a sweep or create result files.
    monkeypatch.chdir(tmp_path)
    module = importlib.import_module(script)
    assert Path(module.__file__).name == f"{script}.py"
    assert list(tmp_path.iterdir()) == []


def test_legacy_hiton_deduction_constructor():
    from cddd.algorithms import HitonPC
    from cddd.independence import CITester
    from cddd.inference import DeductiveReasoning

    runner = HitonPC(.01, CITester(), is_deduce_dep=True)
    assert isinstance(runner.deductor, DeductiveReasoning)
