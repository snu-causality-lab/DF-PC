# Rebuttal Snippet

## dQqc
This section tracks the execution configuration and cluster logs for the DF-PC rebuttal experiments locally orchestrated through the `DF-PC` conda environment. The parallel scaling strategy relies on `n_jobs=24` across 1 computational node dynamically queued by Slurm `sbatch`. Data sets traverse `linear_sem` and `discrete` regimes bounded by degree caps tightly.

## eePS
The structure successfully encapsulates:
- `exp01_baseline`: Validating DF-PC vs Standard PC vs Deduce-Dep
- `exp02_density`: Extreme connectivity ($d=6$)
- `exp03_runtime`: Disentangled Node and Density scaling runtime matrices
- `exp04_saved_cits`: Explicit topological breakdown by conditioning set dimension $|Z|$
- `exp05_robustness`: Native algorithmic resilience against statistically injected local false correlations
