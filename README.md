# Don’t Test What You Can Deduce: Causal Discovery with Logical Inference

This repository contains the official implementation of [Don’t Test What You Can Deduce: Causal Discovery with Logical Inference](https://proceedings.mlr.press/v337/kim26a.html), by Jonghwan Kim and Sanghack Lee (UAI 2026).

> **Note:** Our implementation and experiments focus on skeleton discovery; CPDAG orientation is not included in the released implementation.

---

## 🛠️ Installation

Use Python 3.11. From a terminal, clone the repository and create a dedicated environment:

```bash
git clone https://github.com/snu-causality-lab/DF-PC.git
cd DF-PC
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead; the batch launchers require Bash. Run the commands below from the repository root in the activated environment. The requirements retain `causal-learn==0.1.3.8` and the numerical-library version ranges used by the experiment code. Experiment 6 has [additional dependencies](#6-semi-synthetic-data-dependency).

## Using DF-PC and choosing the search mode

`PCStable` implements both PC-stable and DF-PC. Pass `use_deduction=True` for DF-PC; the constructor's existing default `use_deduction=False` selects PC-stable.

This self-contained example generates a small linear-Gaussian dataset and uses the Fisher-Z partial-correlation tester. Input is a numeric array or DataFrame with samples in rows and variables in columns. Categorical and nonlinear experiments use different testers in their scripts.

```python
import numpy as np

from cddd.algorithms import PCStable
from cddd.independence import PartialCorrelation

rng = np.random.default_rng(42)
x = rng.normal(size=1000)
y = 0.8 * x + rng.normal(size=1000)
z = 0.8 * y + rng.normal(size=1000)
data = np.column_stack([x, y, z])

tester = PartialCorrelation(data=data)
runner = PCStable(
    alpha=0.01,
    ci_tester=tester,
    use_deduction=True,
    deduction_priority="dep",
    deduction_pure=True,
    early_stopping=True,
)
skeleton, separating_sets = runner.run(data)
print(skeleton)
print(separating_sets)
```

The returned adjacency matrix is an undirected skeleton, not a CPDAG.

Variable indices follow the input column order. For example, a `separating_sets` entry `{(0, 2): (1,)}` records that columns 0 and 2 were judged independent given column 1.

| DF-PC mode | `deduction_pure` | `early_stopping` | Behavior |
| --- | --- | --- | --- |
| Paper DF-PC settings | `True` | `True` | Pure logical deduction before a top-level CI fallback; stop the current conditioning-set search at the first independence. |
| No-break option | `True` | `False` | Process all eligible conditioning sets at each level, even after finding an independence. |
| Recursive CI option | `False` | `True` | Keep early stopping, but allow recursive requests to perform missing lower-order CI tests. |

`early_stopping` also applies to PC-stable (`use_deduction=False`). In either mode, edge removals occur at the end of the level. No-break retains the existing separator-recording rule: the last recorded independence for a pair supplies its separating set. It does not preserve all separating sets.

The paper's DF-PC mode remains pure deduction, dependence priority (`"dep"`), and early stopping. The experiment scripts retain their original settings; the new option is available through the core Python API, not a new command-line switch for every experiment wrapper.

In finite samples, early stopping can leave different lower-order CI histories for later deduction, so default DF-PC skeletons can depend on query order despite level-wise adjacency updates. No-break processes additional queries and can cost more CI tests and runtime. It does not guarantee better accuracy; separating-set selection and CPDAG orientation are not covered by a skeleton-order comparison. Changing `deduction_pure` is distinct from turning off early stopping.

### Reusing a runner

Each repeated call to `PCStable.run(data)` resets its deduction cache, counters,
and provenance, and rebinds built-in CI testers to the supplied data with a fresh
backend. This also handles data modified in place. For the first call, initialize
a fresh statistical tester with the same data object you pass to `run`, as in the
example above. That first call retains the preinitialized backend.

Custom testers must implement `reset_for_data(data)` to support repeated calls.
`CITester` provides this hook for its built-in state; subclasses with additional
run-specific caches or counters must override it and call `super()`. Testers
without this hook remain usable for one run. Experiment wrappers that override
`run` are unchanged and continue to construct fresh objects for their trials.

The reset is independent of `early_stopping`; both break and no-break support reuse. An `OracleCITester` remains tied to its supplied `true_graph`, so create a new oracle tester when the ground-truth graph changes. These runner guarantees apply to `PCStable`, not the legacy `HitonPC` class.

---

## 📊 Running Experiments

Run the benchmark scripts from the repository root in the activated environment. They execute full experiment grids, not smoke tests, and write to `results/`; use a separate checkout or preserve existing outputs before rerunning. Several scripts use parallel workers (including fixed counts of 24 or all allocated/available CPUs), so check their settings before running on a shared machine.

The repository includes saved CSVs for regenerating tables and figures without rerunning the experiments. These are supplied result snapshots, not a guarantee that every timing or manuscript cell will reproduce exactly. Record the code commit and environment for new runs; runtime depends on hardware and concurrent load.

### 1. Core Benchmarks (Experiments 1–6)
Individual scripts for the primary evaluations in the main paper:

1. **Experiment 1: Synthetic Benchmarks ($N \in \{10, 20, 30\}$)**
   ```bash
   python experiment_synthetic_benchmarks.py
   ```
2. **Experiment 2: Nonlinear Causal Discovery (KCI Test)**
   ```bash
   python experiment_nonlinear.py
   ```
3. **Experiment 3: Extreme Case Analysis (Collider & Source Hubs)**
   ```bash
   python experiment_extreme_cases.py
   ```
4. **Experiment 4: Alpha Significance Sensitivity**
   ```bash
   python experiment_alpha_sensitivity.py
   ```
5. **Experiment 5: High-Dimensional Scalability ($N=100$)**
   ```bash
   python experiment_scalability.py
   ```
6. **Experiment 6: Semi-Synthetic Benchmarks (Barley & Mildew)**
   ```bash
   python experiment_realworld.py
   ```

#### 6. Semi-synthetic data dependency

Before Experiment 6, install `python -m pip install -r requirements-realworld.txt`. It pins `pgmpy==0.1.25` for the Barley/Mildew example networks and includes the base dependencies. These are sampled from known Bayesian networks, not supplied observational datasets. The other experiments and the regression tests do not need this extra installation.

#### Generate Main Paper Tables
To consolidate and generate the LaTeX tables for the core experiments:
```bash
python generate_paper_tables.py
```
Output is saved to `results/final_paper_tables.tex`.

---

### 2. Extended Benchmarks (Experiments 7–11)
Individual scripts for supplementary and extended evaluations:

7. **Experiment 7: Baseline Comparison (DF-PC vs PC-stable vs Deduce-Dep)**
   ```bash
   python experiment_baseline_comparison.py
   ```
8. **Experiment 8: High-Density Network Regime ($D=6$)**
   ```bash
   python experiment_high_density.py
   ```
9. **Experiment 9: Runtime Scaling across Nodes and Density**
   ```bash
   python experiment_runtime_scaling.py
   ```
10. **Experiment 10: Oracle Saved CITs Breakdown by Conditioning Set Size ($|\mathbf{Z}|$)**
    ```bash
    python experiment_oracle_saved_cits.py
    ```
11. **Experiment 11: Algorithmic Resilience to CIT Errors**
    ```bash
    python experiment_noise_robustness.py
    ```

#### Batch Execution
To run Experiments 7–11 and their plotting step sequentially in your activated environment:

```bash
bash run_all_local.sh
```

The local launcher resolves the repository directory and stops on the first failed command. It defaults BLAS thread counts to one without overriding explicit local settings. `DFPC_PYTHON` can select a particular Python executable; it must be a single executable name or path, not a command with arguments.

`run_all_slurm.sh` is a CPU-only Slurm example. Set these values for **your own cluster** before submission; the project and environment must be accessible on the compute nodes. The conda environment is activated inside the job.

```bash
export DFPC_PROJECT_DIR="/absolute/nfs/path/to/DF-PC"
export DFPC_CONDA_SH="/absolute/nfs/path/to/miniforge3/etc/profile.d/conda.sh"
export DFPC_CONDA_ENV="your-dfpc-environment"
sbatch --chdir="/absolute/nfs/path/to/DF-PC" run_all_slurm.sh
```

Use the same literal project path for `--chdir` and `DFPC_PROJECT_DIR`; shell variables in `#SBATCH` directives are not expanded. Adjust the example's partition and CPU allocation to your cluster. The job requires a valid allocation, uses `SLURM_CPUS_PER_TASK` for the experiments' parallel workers, and sets each worker's BLAS thread count to one. Leave `DFPC_PYTHON` unset to use the activated conda environment's Python, or explicitly select a compatible interpreter. Neither launcher changes the paper's algorithm settings.

#### Generate Supplementary Figures & Tables
To generate publication-ready plots (PDF & PNG) and LaTeX tables for extended benchmarks:
```bash
python generate_paper_plots.py
```
Outputs are saved in `results/plots/` and `results/supplementary_tables.tex`.

Both generators require their input CSVs and exit with an error for missing files or generation failures. These checks do not verify complete experiment-grid coverage. Input-loading failures do not replace existing tables; later plotting failures can leave partial outputs, so preserve any outputs you need before regeneration.

Extended-result filenames `experiment_1_*` through `experiment_5_*` are legacy aliases for Experiments 7–11, not the core Experiments 1–5. Canonical descriptive filenames take precedence. The retained `results/supplementary_figures.tex` is a legacy figure-inclusion snippet; its old “logical deduction” wording for bypassed tests should be read using the cache-inclusive definition below.

### Reading the CI counts

- `Requests`: primary CI queries requested by the skeleton search, including repeated requests.
- `Actual_CIT` / `Performed_CITs`: the statistical testers' count of distinct tested queries. The experiment-specific oracle testers in Experiments 10–11 instead count calls to their oracle.
- `Bypassed_CITs`: requests minus the reported performed count in the supplied pure-mode experiments. This includes cache reuse as well as deduction; it is not a count of newly applied logical inferences.

The statistical PC baseline also has CI-backend caching. PC-versus-DF-PC test-count differences measure the overall algorithm effect, not an isolated deduction-only effect. For recursive-CI variants, helper-internal tests must be counted separately; the same subtraction is not a general measure of top-level deductions. Fewer CI tests need not imply lower runtime or better accuracy.

## Regression tests

The small tests do not run the paper's experiment grids:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests
```

GitHub Actions runs this suite with Python 3.11 and the base dependencies. It exercises the README example, runner reuse, search options, launchers and generation from copied saved results. It does not rerun the full experiment grids or install the optional Experiment 6 dependencies. The small Experiment 6 sampling tests run when those dependencies are installed and otherwise skip.

---

## 📂 Project Structure

```text
├── cddd/                               # Core DF-PC package & deductive inference engine
│   ├── algorithms.py                   # Constraint-based discovery algorithms
│   ├── data.py                         # Graph models, Bayesian networks & synthetic DAG generators
│   ├── independence.py                 # Conditional independence test wrappers (Fisher-Z, Chi-Square, KCI)
│   ├── inference.py                    # Logical deduction rules (Soundness & Completeness)
│   └── metrics.py                      # Evaluation metrics (F1, SHD, Precision, Recall)
├── experiment_synthetic_benchmarks.py  # Exp 1: Standard synthetic benchmarks
├── experiment_nonlinear.py             # Exp 2: Nonlinear discovery (KCI)
├── experiment_extreme_cases.py         # Exp 3: Extreme hub topologies
├── experiment_alpha_sensitivity.py     # Exp 4: Significance threshold sensitivity
├── experiment_scalability.py           # Exp 5: Scalability to N=100
├── experiment_realworld.py             # Exp 6: Real-world semi-synthetic DAGs
├── experiment_baseline_comparison.py   # Exp 7: Comparison against PC-stable & Deduce-Dep
├── experiment_high_density.py          # Exp 8: High-density graph regime (D=6)
├── experiment_runtime_scaling.py       # Exp 9: Runtime scaling across N & D
├── experiment_oracle_saved_cits.py     # Exp 10: Oracle CIT breakdown by |Z|
├── experiment_noise_robustness.py      # Exp 11: Resilience to low-order CIT noise
├── generate_paper_tables.py            # Generates main paper LaTeX tables
├── generate_paper_plots.py             # Generates supplementary figures and LaTeX tables
├── results/                            # Consolidated results directory
│   ├── *.csv                           # Raw data and summary metrics
│   ├── plots/                          # High-resolution PDF and PNG figures
│   ├── final_paper_tables.tex          # Consolidated main paper tables
│   └── supplementary_tables.tex        # Supplementary tables
├── tests/                             # Runner lifecycle, search-mode and release regressions
├── run_all_local.sh                    # Batch execution script (local workstation)
├── run_all_slurm.sh                    # SLURM batch execution script (cluster)
├── requirements.txt                    # Base experiment and plotting dependencies
├── requirements-realworld.txt          # Extra dependencies for Experiment 6
└── requirements-dev.txt                # Base dependencies plus pytest
```

## License status

Public-release licensing is pending; no repository-wide license has been added. The existing third-party notice in [cddd/independence.py](cddd/independence.py) is retained.

## Citation

Please cite the [published paper](https://proceedings.mlr.press/v337/kim26a.html) when using DF-PC:

```bibtex
@inproceedings{pmlr-v337-kim26a,
  title = {Don't Test What You Can Deduce: Causal Discovery with Logical Inference},
  author = {Kim, Jonghwan and Lee, Sanghack},
  booktitle = {Proceedings of the 42nd Conference on Uncertainty in Artificial Intelligence},
  pages = {2958--2993},
  year = {2026},
  volume = {337},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v337/kim26a.html}
}
```
