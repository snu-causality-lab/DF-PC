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

echo "Starting Exp 1..."
python rebuttal_experiments/exp1_baseline/experiment_rebuttal_1_baseline.py
echo "Starting Exp 2..."
python rebuttal_experiments/exp2_density/experiment_rebuttal_2_density.py
echo "Starting Exp 3..."
python rebuttal_experiments/exp3_runtime/experiment_rebuttal_3_runtime.py
echo "Starting Exp 4..."
python rebuttal_experiments/exp4_saved_cits/experiment_rebuttal_4_saved_cits.py
echo "Starting Exp 5..."
python rebuttal_experiments/exp5_robustness/experiment_rebuttal_5_robustness.py

echo "Generating Results..."
python generate_rebuttal_tables_plots.py

echo "ALL COMPLETED!"
