#!/bin/bash
#SBATCH --job-name=dfpc_all_exps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=all
#SBATCH --output=dfpc-all-%j.out
#SBATCH --error=dfpc-all-%j.err

set -euo pipefail

# CPU-only example. Export these variables explicitly and submit with
# sbatch --chdir=/absolute/shared/project run_all_slurm.sh (see README).
# Slurm does not expand shell variables in #SBATCH directives.
: "${DFPC_PROJECT_DIR:?Set DFPC_PROJECT_DIR to the absolute shared project path}"
: "${DFPC_CONDA_SH:?Set DFPC_CONDA_SH to an absolute conda.sh path}"
: "${DFPC_CONDA_ENV:?Set DFPC_CONDA_ENV to the conda environment name or prefix}"
: "${SLURM_JOB_ID:?Submit this script through sbatch}"
: "${SLURM_CPUS_PER_TASK:?Request CPUs with sbatch --cpus-per-task}"

if [[ ! "$SLURM_CPUS_PER_TASK" =~ ^[1-9][0-9]*$ ]]; then
    echo "SLURM_CPUS_PER_TASK must be a positive integer" >&2
    exit 1
fi
if [[ "$DFPC_PROJECT_DIR" != /* || ! -d "$DFPC_PROJECT_DIR" ]]; then
    echo "DFPC_PROJECT_DIR must be an existing absolute directory" >&2
    exit 1
fi
for dfpc_script in \
    run_all_local.sh \
    experiment_baseline_comparison.py \
    experiment_high_density.py \
    experiment_runtime_scaling.py \
    experiment_oracle_saved_cits.py \
    experiment_noise_robustness.py \
    generate_paper_plots.py; do
    if [[ ! -f "$DFPC_PROJECT_DIR/$dfpc_script" ]]; then
        echo "Missing project script: $DFPC_PROJECT_DIR/$dfpc_script" >&2
        exit 1
    fi
done
if [[ "$DFPC_CONDA_SH" != /* || ! -f "$DFPC_CONDA_SH" || ! -r "$DFPC_CONDA_SH" ]]; then
    echo "DFPC_CONDA_SH must be an absolute, readable conda.sh file" >&2
    exit 1
fi

# Activate inside the job, even if the submission shell has an active environment.
source "$DFPC_CONDA_SH"
conda activate "$DFPC_CONDA_ENV"

# Experiments already use SLURM_CPUS_PER_TASK joblib workers. Pin each worker's
# numerical libraries to one thread instead of nesting N workers x N threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd -- "$DFPC_PROJECT_DIR"
exec bash "$DFPC_PROJECT_DIR/run_all_local.sh"
