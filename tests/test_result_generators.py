"""Check result generation without rerunning the scientific experiments."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATORS = [
    ("generate_paper_tables.py", "final_paper_tables.tex"),
    ("generate_paper_plots.py", "supplementary_tables.tex"),
]


def run_generator(script, cwd):
    env = os.environ.copy()
    env.update(MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / script)], cwd=cwd, env=env,
        capture_output=True, text=True, timeout=120,
    )


def copy_results(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    for source in (REPO_ROOT / "results").glob("*.csv"):
        shutil.copyfile(source, results / source.name)
    return results


def test_importing_plot_generator_preserves_outputs(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    output = results / "supplementary_tables.tex"
    output.write_text("previous tables")
    env = os.environ.copy()
    env.update(
        MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO_ROOT),
    )
    result = subprocess.run(
        [sys.executable, "-c", "import generate_paper_plots"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text() == "previous tables"
    assert not (results / "plots").exists()


@pytest.mark.parametrize("script, output_name", GENERATORS)
def test_missing_inputs_fail_without_overwriting_tables(tmp_path, script, output_name):
    results = tmp_path / "results"
    results.mkdir()
    output = results / output_name
    output.write_text("previous tables")
    result = run_generator(script, tmp_path)
    assert result.returncode != 0
    assert "success" not in result.stdout.lower()
    assert output.read_text() == "previous tables"


@pytest.mark.parametrize("script, output_name", GENERATORS)
def test_malformed_late_input_preserves_tables(tmp_path, script, output_name):
    results = copy_results(tmp_path)
    input_name = (
        "experiment_realworld_summary.csv" if script == "generate_paper_tables.py"
        else "experiment_noise_robustness_summary.csv"
    )
    (results / input_name).write_text("unexpected_column\n1\n")
    output = results / output_name
    output.write_text("previous tables")
    result = run_generator(script, tmp_path)
    assert result.returncode != 0
    assert "success" not in result.stdout.lower()
    assert output.read_text() == "previous tables"


def test_full_main_tables_match_preserved_results(tmp_path):
    results = copy_results(tmp_path)
    result = run_generator("generate_paper_tables.py", tmp_path)
    assert result.returncode == 0, result.stderr
    assert (results / "final_paper_tables.tex").read_bytes() == (
        REPO_ROOT / "results" / "final_paper_tables.tex"
    ).read_bytes()


def test_plots_accept_legacy_csv_names_and_single_oracle_facets(tmp_path):
    results = copy_results(tmp_path)
    for kind in ("raw", "summary"):
        oracle_path = results / f"experiment_4_{kind}.csv"
        oracle = pd.read_csv(oracle_path)
        keep = (
            ((oracle["Setup"] == "SetupA_Nodes") & (oracle["Nodes"] == 10))
            | ((oracle["Setup"] == "SetupB_Density") & (oracle["Deg"] == 2))
        )
        oracle.loc[keep].to_csv(oracle_path, index=False)
        for stem in (
            "baseline_comparison", "high_density", "runtime_scaling",
            "oracle_saved_cits", "noise_robustness",
        ):
            (results / f"experiment_{stem}_{kind}.csv").unlink()
    result = run_generator("generate_paper_plots.py", tmp_path)
    assert result.returncode == 0, result.stderr
    assert (results / "supplementary_tables.tex").read_text().count("subsection*") == 5
    for experiment in (4, 10):
        for setup in ("nodes", "density"):
            for extension in ("png", "pdf"):
                output = results / "plots" / f"exp{experiment}_save_CIT_{setup}.{extension}"
                assert output.stat().st_size > 0
