#!/bin/bash
#SBATCH --job-name=dfpc_exp3_runtime
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --output=rebuttal_experiments/exp3_runtime/slurm-%j.out

source /mnt/nfs/colossal/jonghwan/miniforge3/etc/profile.d/conda.sh
conda activate DF-PC
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH
cd $SLURM_SUBMIT_DIR
python rebuttal_experiments/exp3_runtime/experiment_rebuttal_3_runtime.py
