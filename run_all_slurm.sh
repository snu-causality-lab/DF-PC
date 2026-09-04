#!/bin/bash
#SBATCH --job-name=dfpc_all_exps
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --chdir=/mnt/nfs/colossal/jonghwan/DF-PC
#SBATCH --output=/mnt/nfs/colossal/jonghwan/slurm-logs/dfpc-all-%j.out
#SBATCH --error=/mnt/nfs/colossal/jonghwan/slurm-logs/dfpc-all-%j.err

mkdir -p /mnt/nfs/colossal/jonghwan/slurm-logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /mnt/nfs/colossal/jonghwan/miniforge3/etc/profile.d/conda.sh
conda activate DF-PC

export PYTHONPATH=/mnt/nfs/colossal/jonghwan/DF-PC:$PYTHONPATH

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

echo "Generating Results..."
python generate_paper_plots.py

echo "ALL COMPLETED!"
