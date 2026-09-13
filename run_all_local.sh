#!/bin/bash
set -euo pipefail

# Activate the desired environment before launching this script. DFPC_PYTHON
# may select a different interpreter executable (not a command with arguments).
dfpc_python="${DFPC_PYTHON:-python}"
if ! dfpc_interpreter="$(command -v -- "$dfpc_python")"; then
    echo "Python interpreter not found: $dfpc_python" >&2
    exit 1
fi
# Keep the selected executable fixed when a relative path or PATH entry was used.
dfpc_python="$dfpc_interpreter"
if [[ "$dfpc_python" != /* && "$dfpc_python" == */* ]]; then
    dfpc_python="$PWD/$dfpc_python"
fi

dfpc_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$dfpc_project_dir"
for dfpc_script in \
    experiment_baseline_comparison.py \
    experiment_high_density.py \
    experiment_runtime_scaling.py \
    experiment_oracle_saved_cits.py \
    experiment_noise_robustness.py \
    generate_paper_plots.py; do
    if [[ ! -f "$dfpc_script" ]]; then
        echo "Missing project script: $dfpc_project_dir/$dfpc_script" >&2
        exit 1
    fi
done
export PYTHONPATH="${dfpc_project_dir}${PYTHONPATH:+:$PYTHONPATH}"

# Each experiment uses joblib workers. Avoid nested BLAS parallelism by default;
# explicit local thread settings are preserved.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "Starting Exp 7 (Baseline Comparison)..."
"$dfpc_python" experiment_baseline_comparison.py
echo "Starting Exp 8 (High Density)..."
"$dfpc_python" experiment_high_density.py
echo "Starting Exp 9 (Runtime Scaling)..."
"$dfpc_python" experiment_runtime_scaling.py
echo "Starting Exp 10 (Oracle Saved CITs)..."
"$dfpc_python" experiment_oracle_saved_cits.py
echo "Starting Exp 11 (Noise Robustness)..."
"$dfpc_python" experiment_noise_robustness.py

echo "Generating LaTeX and Plots..."
"$dfpc_python" generate_paper_plots.py

echo "ALL COMPLETED!"
