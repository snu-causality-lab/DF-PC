# Don’t Test What You Can Deduce: Causal Discovery with Logical Inference

This repository contains the official implementation for the paper **"Don’t Test What You Can Deduce: Causal Discovery with Logical Inference"** (Accepted to UAI 2026).

The codebase is refactored for simplicity, reproducibility, and clarity.

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

---

## 📊 Running Experiments

### 1. Main Paper Experiments (Core Benchmarks)
Run each script individually from the root directory:

1. **Synthetic Benchmarks ($N \in \{10, 20, 30\}$)**:
   ```bash
   python experiment_synthetic_benchmarks.py
   ```
2. **Nonlinear Causal Discovery (KCI Test)**:
   ```bash
   python experiment_nonlinear.py
   ```
3. **Extreme Case Analysis (Collider/Source Hubs)**:
   ```bash
   python experiment_extreme_cases.py
   ```
4. **Alpha Sensitivity Analysis**:
   ```bash
   python experiment_alpha_sensitivity.py
   ```
5. **Scalability Analysis ($N=100$)**:
   ```bash
   python experiment_scalability.py
   ```
6. **Real-World Benchmarks (Barley, Mildew)**:
   ```bash
   python experiment_realworld.py
   ```

#### Generate Main Tables
```bash
python generate_paper_tables.py
```
Consolidated results are saved in `results/final_paper_tables.tex`.

---

### 2. Rebuttal Experiments (Appendix Experiments 7–11)
The rebuttal suite provides extended analyses under dense graphs, runtime scaling, oracle deduction limits, and robustness against noise.

Run individual experiments from `rebuttal_experiments/`:
* **Experiment 7 (Baseline Comparison with Deduce-Dep)**:
  ```bash
  python rebuttal_experiments/exp1_baseline/experiment_rebuttal_1_baseline.py
  ```
* **Experiment 8 (High-Density Graphs, $D=6$)**:
  ```bash
  python rebuttal_experiments/exp2_density/experiment_rebuttal_2_density.py
  ```
* **Experiment 9 (Runtime Scaling across Nodes & Density)**:
  ```bash
  python rebuttal_experiments/exp3_runtime/experiment_rebuttal_3_runtime.py
  ```
* **Experiment 10 (Oracle CIT Skip Breakdown across $|\mathbf{Z}|$)**:
  ```bash
  python rebuttal_experiments/exp4_saved_cits/experiment_rebuttal_4_saved_cits.py
  ```
* **Experiment 11 (Robustness against Low-Order CIT Errors)**:
  ```bash
  python rebuttal_experiments/exp5_robustness/experiment_rebuttal_5_robustness.py
  ```

#### Run All Rebuttal Experiments
```bash
bash run_all_local.sh
```

#### Generate Rebuttal Tables & Visualizations
```bash
python generate_rebuttal_tables_plots.py
```
Outputs (summary LaTeX tables and high-resolution PDF/PNG figures) are saved in `results_rebuttal/`.

---

## 📂 Project Structure

```text
├── cddd/                     # Core DF-PC algorithm & deductive inference engine
├── experiment_*.py           # Main paper experiment scripts
├── generate_paper_tables.py  # Script to generate main paper LaTeX tables
├── rebuttal_experiments/     # Additional experiments added during rebuttal (Exp 7–11)
│   ├── exp1_baseline/
│   ├── exp2_density/
│   ├── exp3_runtime/
│   ├── exp4_saved_cits/
│   └── exp5_robustness/
├── generate_rebuttal_tables_plots.py # Script to generate rebuttal tables & plots
├── results/                  # Main experiment CSV summaries and tables
├── results_rebuttal/         # Rebuttal experiment data, tables, and plots
├── requirements.txt          # Python dependencies
└── run_all_local.sh          # Batch script for rebuttal experiments
```
