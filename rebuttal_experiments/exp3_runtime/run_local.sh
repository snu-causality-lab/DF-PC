#!/bin/bash
source /mnt/nfs/colossal/jonghwan/miniforge3/etc/profile.d/conda.sh
conda activate DF-PC
export PYTHONPATH=$PWD:$PYTHONPATH
python rebuttal_experiments/exp3_runtime/experiment_rebuttal_3_runtime.py
