#!/bin/bash
#SBATCH --job-name=dfpc_exp2_density
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --output=rebuttal_experiments/exp2_density/slurm-%j.out

source /mnt/nfs/colossal/jonghwan/miniforge3/etc/profile.d/conda.sh
conda activate DF-PC
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH
cd $SLURM_SUBMIT_DIR
python rebuttal_experiments/exp2_density/experiment_rebuttal_2_density.py
