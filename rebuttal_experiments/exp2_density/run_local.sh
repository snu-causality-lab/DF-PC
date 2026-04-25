#!/bin/bash
source /mnt/nfs/colossal/jonghwan/miniforge3/etc/profile.d/conda.sh
conda activate DF-PC
export PYTHONPATH=$PWD:$PYTHONPATH
python rebuttal_experiments/exp2_density/experiment_rebuttal_2_density.py
