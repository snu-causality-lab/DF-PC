#!/bin/bash
#SBATCH --job-name=dfpc_exp1_baseline
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --output=rebuttal_experiments/exp1_baseline/slurm-%j.out

source /mnt/nfs/colossal/jonghwan/miniforge3/etc/profile.d/conda.sh
conda activate DF-PC
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH
cd $SLURM_SUBMIT_DIR
python rebuttal_experiments/exp1_baseline/experiment_rebuttal_1_baseline.py
