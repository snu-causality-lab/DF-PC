# Don’t Test What You Can Deduce: Causal Discovery with Logical Inference

This repository contains the implementation for the paper **"Don’t Test What You Can Deduce: Causal Discovery with Logical Inference"** (Submitted to UAI 2026).

The codebase is refactored for simplicity and reproducibility.

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 📊 Running Experiments

The project consists of 6 core experiments. Run them individually from the root directory:

1. **Synthetic Benchmarks (N=10, 20, 30)**:
   ```bash
   python experiment_synthetic_benchmarks.py
   ```
2. **Nonlinear Causal Discovery (KCI)**:
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
5. **Scalability Analysis (N=100)**:
   ```bash
   python experiment_scalability.py
   ```
6. **Real-World Benchmarks (Barley, Mildew)**:
   ```bash
   python experiment_realworld.py
   ```

### Generate Results
After running the experiments, generate the LaTeX tables using:
```bash
python generate_paper_tables.py
```
Consolidated results are saved in `results/final_paper_tables.tex`.

## 📂 Project Structure
- `cddd/`: Core algorithm logic and utilities.
- `results/`: Output directory for CSV and LaTeX results.
- `experiment_*.py`: Individual experiment scripts.
