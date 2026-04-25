#!/bin/bash
source /mnt/nfs/colossal/jonghwan/miniforge3/etc/profile.d/conda.sh
conda activate DF-PC
export PYTHONPATH=$PWD:$PYTHONPATH
python rebuttal_experiments/exp4_saved_cits/experiment_rebuttal_4_saved_cits.py
