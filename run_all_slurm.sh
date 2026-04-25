#!/bin/bash
#SBATCH --job-name=dfpc_all_exps
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --output=rebuttal_experiments/slurm_all_exps-%j.out

source /mnt/nfs/colossal/jonghwan/miniforge3/etc/profile.d/conda.sh
conda activate DF-PC
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH
cd $SLURM_SUBMIT_DIR

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

echo "ALL EXPERIMENTS COMPLETED!"
