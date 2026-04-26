#!/bin/bash
#SBATCH --job-name=dfpc_exp5
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --chdir=/mnt/nfs/colossal/jonghwan/DF-PC
#SBATCH --output=/mnt/nfs/colossal/jonghwan/slurm-logs/exp5-%j.out
#SBATCH --error=/mnt/nfs/colossal/jonghwan/slurm-logs/exp5-%j.err

mkdir -p /mnt/nfs/colossal/jonghwan/slurm-logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /mnt/nfs/colossal/jonghwan/miniforge3/etc/profile.d/conda.sh
conda activate DF-PC
export PYTHONPATH=/mnt/nfs/colossal/jonghwan/DF-PC:$PYTHONPATH

python rebuttal_experiments/exp5_robustness/experiment_rebuttal_5_robustness.py
