# Don’t Test What You Can Deduce: Causal Discovery with Logical Inference

This repository contains the official implementation for the paper **"Don’t Test What You Can Deduce: Causal Discovery with Logical Inference"** (Accepted to UAI 2026).

The codebase is structured for reproducibility, clarity, and ease of experimentation.

> **Note:** Our implementation and experiments focus on skeleton discovery; CPDAG orientation is not included in the released implementation.

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

---

## 📊 Running Experiments

All benchmark experiments are runnable directly from the root directory.

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
To run all extended experiments sequentially:
```bash
bash run_all_local.sh
# Or submit via SLURM:
sbatch run_all_slurm.sh
```

#### Generate Supplementary Figures & Tables
To generate publication-ready plots (PDF & PNG) and LaTeX tables for extended benchmarks:
```bash
python generate_paper_plots.py
```
Outputs are saved in `results/plots/`, `results/supplementary_tables.tex`, and `results/supplementary_figures.tex`.

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
│   ├── supplementary_tables.tex        # Supplementary tables
│   └── supplementary_figures.tex       # Supplementary figure snippets
├── run_all_local.sh                    # Batch execution script (local workstation)
├── run_all_slurm.sh                    # SLURM batch execution script (cluster)
└── requirements.txt                    # Environment dependencies
```
