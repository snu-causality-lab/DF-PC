#!/bin/bash
source /mnt/nfs/colossal/jonghwan/miniforge3/etc/profile.d/conda.sh
conda activate DF-PC
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Starting Exp 7 (Baseline Comparison)..."
python experiment_baseline_comparison.py
echo "Starting Exp 8 (High Density)..."
python experiment_high_density.py
echo "Starting Exp 9 (Runtime Scaling)..."
python experiment_runtime_scaling.py
echo "Starting Exp 10 (Oracle Saved CITs)..."
python experiment_oracle_saved_cits.py
echo "Starting Exp 11 (Noise Robustness)..."
python experiment_noise_robustness.py

echo "Generating LaTeX and Plots..."
python generate_paper_plots.py

echo "ALL COMPLETED!"
