
import pandas as pd
import numpy as np
import networkx as nx
import time
import os
import random
import warnings
from itertools import product
from pathlib import Path
from joblib import Parallel, delayed

from cddd.data import Graph, BayesianNetwork, DataGenerator
from cddd.independence import PartialCorrelation, ChiSquareTester
from cddd.algorithms import PCStable
from cddd.metrics import Evaluator

warnings.filterwarnings('ignore')

def calculate_ci(data):
    if len(data) < 2:
        return 0.0
    return 1.96 * (np.std(data, ddof=1) / np.sqrt(len(data)))

def run_single_alpha(n, m, d, gt, dt, alpha, rep, algo_modes):
    seed = 555 + rep
    random.seed(seed)
    np.random.seed(seed)
    
    if gt == 'er':
        G = Graph.erdos_renyi_DAG(n, int(n * d / 2), max_in_degree=10)
    else:
        G = Graph.scale_free_DAG(n, int(max(1, d / 2)), max_in_degree=10)
    
    true_adj = nx.to_numpy_array(G)
    bn = BayesianNetwork(G, model_type=dt)
    gen = DataGenerator(bn)
    data = gen.sample(m, seed=seed)
    
    results = []
    for algo_name, use_ded, is_pure in algo_modes:
        tester = PartialCorrelation() if dt == 'linear_sem' else ChiSquareTester()
        tester._set_data(data)
        runner = PCStable(alpha=alpha, ci_tester=tester, use_deduction=use_ded, deduction_pure=is_pure)
        
        start_t = time.time()
        adj_est, _ = runner.run(data)
        total_t = time.time() - start_t
        
        evaluator = Evaluator(true_adj)
        prec, rec, f1 = evaluator.global_skeleton_metric_evaluation(adj_est)
        
        results.append({
            "Alpha": alpha, "Rep": rep, "Algo": algo_name,
            "F1": f1, "CIT": tester.n_actual_calls, "Time": total_t
        })
    return results

def main():
    # Setup
    N = 40
    M = 5000
    D = 2
    GT = 'er'
    DT = 'linear_sem'
    REPS = 10
    ALPHAS = [0.05, 0.01, 0.001, 0.0001]
    
    ALGO_MODES = [
        ("Standard PC", False, False),
        ("DF-PC (Pure)", True, True)
    ]
    
    print(f"Running Alpha Sensitivity Experiment (N={N}, M={M}, Data={DT})...")
    
    tasks = list(product(ALPHAS, range(REPS)))
    results_nested = Parallel(n_jobs=20, verbose=10)(
        delayed(run_single_alpha)(N, M, D, GT, DT, a, r, ALGO_MODES)
        for a, r in tasks
    )
    
    all_data = [item for sublist in results_nested for item in sublist]
    df = pd.DataFrame(all_data)
    
    summary = df.groupby(["Alpha", "Algo"])[["F1", "CIT", "Time"]].mean().reset_index()
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "experiment_alpha_sensitivity.csv"
    summary.to_csv(csv_path, index=False)
    
    print("\nAlpha Sensitivity Summary:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
