"""Test launcher control flow with fake commands, without scientific experiments."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    "experiment_baseline_comparison.py",
    "experiment_high_density.py",
    "experiment_runtime_scaling.py",
    "experiment_oracle_saved_cits.py",
    "experiment_noise_robustness.py",
    "generate_paper_plots.py",
]
THREAD_VARS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dfpc-launcher-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "project with spaces"
        self.repo.mkdir()
        for name in ("run_all_local.sh", "run_all_slurm.sh"):
            shutil.copyfile(REPO_ROOT / name, self.repo / name)
        for name in STEPS:
            (self.repo / name).touch()
        self.log = self.root / "calls.tsv"
        self.conda_log = self.root / "conda.log"
        self.fake_python = self.root / "fake python"
        self.fake_python.write_text(
            "#!/bin/bash\n"
            "printf '%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' "
            '"$1" "$PWD" "${PYTHONPATH-}" "${OMP_NUM_THREADS-}" '
            '"${MKL_NUM_THREADS-}" "${OPENBLAS_NUM_THREADS-}" '
            '"${NUMEXPR_NUM_THREADS-}" "${SLURM_CPUS_PER_TASK-}" '
            '"${FAKE_CONDA_ACTIVE-}" >> "$FAKE_LOG"\n'
            'if [[ "$1" == "${FAIL_SCRIPT-}" ]]; then exit 23; fi\n'
        )
        self.fake_python.chmod(0o755)
        self.conda_sh = self.root / "conda bootstrap.sh"
        self.conda_sh.write_text(
            "conda() {\n"
            '  [[ "$1" == activate ]] || return 42\n'
            '  printf "%s\\n" "$2" >> "$FAKE_CONDA_LOG"\n'
            '  export FAKE_CONDA_ACTIVE="$2"\n'
            '  export PATH="$FAKE_CONDA_BIN:$PATH"\n'
            "}\n"
        )
        self.conda_bin = self.root / "conda bin"
        self.conda_bin.mkdir()
        (self.conda_bin / "python").symlink_to(self.fake_python)
        self.fallback_bin = self.root / "fallback bin"
        self.fallback_bin.mkdir()
        self.unexpected_python = self.fallback_bin / "python"
        self.unexpected_python.write_text("#!/bin/bash\nexit 88\n")
        self.unexpected_python.chmod(0o755)
        # Old launchers must not activate a real environment or create NFS paths.
        for name in ("conda", "mkdir"):
            stub = self.fallback_bin / name
            stub.write_text("#!/bin/bash\nexit 97\n")
            stub.chmod(0o755)
        self.env = os.environ.copy()
        for key in list(self.env):
            if (key.startswith(("DFPC_", "SLURM_", "FAKE_"))
                    or key in THREAD_VARS or key == "FAIL_SCRIPT"):
                del self.env[key]
        self.env.update(
            DFPC_PYTHON=str(self.fake_python),
            FAKE_LOG=str(self.log),
            FAKE_CONDA_LOG=str(self.conda_log),
            FAKE_CONDA_BIN=str(self.conda_bin),
            PYTHONPATH="/existing/modules",
            PATH=f"{self.fallback_bin}:{self.env['PATH']}",
        )

    def run_launcher(self, name, env=None):
        return subprocess.run(
            ["bash", str(self.repo / name)],
            cwd=self.root,
            env=self.env if env is None else env,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )

    def calls(self):
        if not self.log.exists():
            return []
        return [line.split("\t") for line in self.log.read_text().splitlines()]

    def slurm_env(self):
        env = self.env.copy()
        env.update(
            DFPC_PROJECT_DIR=str(self.repo),
            DFPC_CONDA_SH=str(self.conda_sh),
            DFPC_CONDA_ENV="dfpc-test-env",
            SLURM_JOB_ID="12345",
            SLURM_CPUS_PER_TASK="4",
        )
        return env

    def assert_no_work(self, result):
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])
        self.assertNotIn("ALL COMPLETED", result.stdout)

    def test_local_uses_selected_interpreter_and_repo_root(self):
        result = self.run_launcher("run_all_local.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual([call[0] for call in calls], STEPS)
        for call in calls:
            self.assertEqual(Path(call[1]).resolve(), self.repo.resolve())
            self.assertEqual(call[2], f"{self.repo.resolve()}:/existing/modules")
            self.assertEqual(call[3:7], ["1"] * 4)
        self.assertIn("ALL COMPLETED", result.stdout)
        self.assertFalse(self.conda_log.exists())

    def test_local_defaults_to_python_from_active_environment(self):
        env = self.env.copy()
        del env["DFPC_PYTHON"]
        env["PATH"] = f"{self.conda_bin}:{env['PATH']}"
        result = self.run_launcher("run_all_local.sh", env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call[0] for call in self.calls()], STEPS)

    def test_local_resolves_relative_interpreter_before_changing_directory(self):
        env = self.env.copy()
        env["DFPC_PYTHON"] = "./fake python"
        result = self.run_launcher("run_all_local.sh", env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call[0] for call in self.calls()], STEPS)

    def test_local_resolves_relative_path_entry_before_changing_directory(self):
        env = self.env.copy()
        del env["DFPC_PYTHON"]
        env["PATH"] = f"conda bin:{env['PATH']}"
        result = self.run_launcher("run_all_local.sh", env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call[0] for call in self.calls()], STEPS)

    def test_local_preserves_explicit_thread_limits(self):
        env = self.env.copy()
        env.update(zip(THREAD_VARS, ["2", "3", "4", "5"]))
        result = self.run_launcher("run_all_local.sh", env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls()[0][3:7], ["2", "3", "4", "5"])

    def test_local_handles_unset_pythonpath_without_empty_search_entry(self):
        env = self.env.copy()
        del env["PYTHONPATH"]
        result = self.run_launcher("run_all_local.sh", env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls()[0][2], str(self.repo.resolve()))

    def test_local_stops_at_first_failed_step(self):
        env = self.env.copy()
        env["FAIL_SCRIPT"] = STEPS[2]
        result = self.run_launcher("run_all_local.sh", env)
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertEqual([call[0] for call in self.calls()], STEPS[:3])
        self.assertNotIn("ALL COMPLETED", result.stdout)

    def test_local_rejects_missing_interpreter(self):
        env = self.env.copy()
        env["DFPC_PYTHON"] = str(self.root / "missing-python")
        self.assert_no_work(self.run_launcher("run_all_local.sh", env))

    def test_local_rejects_incomplete_project_before_first_step(self):
        (self.repo / STEPS[-1]).unlink()
        self.assert_no_work(self.run_launcher("run_all_local.sh"))

    def test_slurm_activates_conda_then_runs_all_steps_with_one_thread(self):
        env = self.slurm_env()
        del env["DFPC_PYTHON"]
        env.update({key: "24" for key in THREAD_VARS})
        result = self.run_launcher("run_all_slurm.sh", env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.conda_log.read_text(), "dfpc-test-env\n")
        calls = self.calls()
        self.assertEqual([call[0] for call in calls], STEPS)
        for call in calls:
            self.assertEqual(Path(call[1]).resolve(), self.repo.resolve())
            self.assertEqual(call[3:7], ["1"] * 4)
            self.assertEqual(call[7:], ["4", "dfpc-test-env"])
        self.assertIn("ALL COMPLETED", result.stdout)

    def test_slurm_supports_explicit_interpreter_override(self):
        (self.conda_bin / "python").unlink()
        (self.conda_bin / "python").symlink_to(self.unexpected_python)
        result = self.run_launcher("run_all_slurm.sh", self.slurm_env())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call[0] for call in self.calls()], STEPS)

    def test_slurm_stops_at_first_failed_step(self):
        env = self.slurm_env()
        env["FAIL_SCRIPT"] = STEPS[1]
        result = self.run_launcher("run_all_slurm.sh", env)
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertEqual([call[0] for call in self.calls()], STEPS[:2])
        self.assertNotIn("ALL COMPLETED", result.stdout)

    def test_slurm_requires_explicit_configuration_and_allocation(self):
        for key in (
            "DFPC_PROJECT_DIR", "DFPC_CONDA_SH", "DFPC_CONDA_ENV",
            "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK",
        ):
            with self.subTest(key=key):
                env = self.slurm_env()
                del env[key]
                self.assert_no_work(self.run_launcher("run_all_slurm.sh", env))
                self.assertFalse(self.conda_log.exists())

    def test_slurm_rejects_invalid_cpu_allocation(self):
        for value in ("0", "-1", "1.5", "four", "", "4(x2)"):
            with self.subTest(value=value):
                env = self.slurm_env()
                env["SLURM_CPUS_PER_TASK"] = value
                self.assert_no_work(self.run_launcher("run_all_slurm.sh", env))
                self.assertFalse(self.conda_log.exists())

    def test_slurm_rejects_invalid_project_and_conda_paths(self):
        for key, value in (
            ("DFPC_PROJECT_DIR", "relative/project"),
            ("DFPC_PROJECT_DIR", str(self.root / "missing-project")),
            ("DFPC_CONDA_SH", "relative/conda.sh"),
            ("DFPC_CONDA_SH", str(self.root / "missing-conda.sh")),
        ):
            with self.subTest(key=key, value=value):
                env = self.slurm_env()
                env[key] = value
                self.assert_no_work(self.run_launcher("run_all_slurm.sh", env))
                self.assertFalse(self.conda_log.exists())

    def test_slurm_rejects_incomplete_project_before_conda_activation(self):
        (self.repo / STEPS[-1]).unlink()
        self.assert_no_work(self.run_launcher("run_all_slurm.sh", self.slurm_env()))
        self.assertFalse(self.conda_log.exists())

    def test_slurm_conda_failure_stops_before_experiments(self):
        self.conda_sh.write_text("conda() { return 19; }\n")
        result = self.run_launcher("run_all_slurm.sh", self.slurm_env())
        self.assert_no_work(result)
        self.assertEqual(result.returncode, 19, result.stderr)

    def test_slurm_bootstrap_failure_stops_before_experiments(self):
        self.conda_sh.write_text("return 17\n")
        result = self.run_launcher("run_all_slurm.sh", self.slurm_env())
        self.assert_no_work(result)
        self.assertEqual(result.returncode, 17, result.stderr)


if __name__ == "__main__":
    unittest.main()
