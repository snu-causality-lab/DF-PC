
import pandas as pd
import numpy as np
import networkx as nx
import time
import os
import random
import warnings
from pathlib import Path
from joblib import Parallel, delayed
from scipy import stats

from cddd.data import Graph, BayesianNetwork, DataGenerator
from cddd.independence import PartialCorrelation
from cddd.algorithms import PCStable
from cddd.metrics import Evaluator

warnings.filterwarnings('ignore')

def calculate_ci(data):
    if len(data) < 2:
        return 0.0
    return 1.96 * (np.std(data, ddof=1) / np.sqrt(len(data)))

def run_scalability_trial(n, m, d, g_type, rep, alpha=0.001):
    seed = 500 + rep
    random.seed(seed)
    np.random.seed(seed)
    
    if g_type == 'er':
        G = Graph.erdos_renyi_DAG(n, int(n * d / 2), max_in_degree=10)
    else:
        G = Graph.scale_free_DAG(n, int(max(1, d / 2)), max_in_degree=10)
    
    true_adj = nx.to_numpy_array(G)
    bn = BayesianNetwork(G, model_type='linear_sem')
    gen = DataGenerator(bn)
    data = gen.sample(m, seed=seed)
    
    uncond_calc = (n * (n - 1)) // 2
    results = []
    
    for algo_name, use_ded in [("Standard PC", False), ("DF-PC (Pure)", True)]:
        tester = PartialCorrelation()
        tester._set_data(data)
        runner = PCStable(alpha=alpha, ci_tester=tester, use_deduction=use_ded)
        
        start_t = time.time()
        adj_est, _ = runner.run(data)
        total_t = time.time() - start_t
        
        evaluator = Evaluator(true_adj)
        prec, rec, f1 = evaluator.global_skeleton_metric_evaluation(adj_est)
        shd = evaluator.get_skeleton_SHD(true_adj, adj_est)
        
        actual_cits = tester.n_actual_calls
        
        results.append({
            "Nodes": n,
            "Samples": m,
            "Deg": d,
            "Graph": g_type,
            "Rep": rep,
            "Algo": algo_name,
            "F1": f1,
            "Prec": prec,
            "Rec": rec,
            "SHD": shd,
            "Actual_CIT": actual_cits,
            "Cond_CIT": actual_cits - uncond_calc,
            "Requests": runner.total_pc_requests,
            "Time_Total": total_t
        })
    return results

def main():
    N = 100
    M = 1000
    DEGS = [2, 4]
    GRAPHS = ['er', 'sf']
    REPS = 30
    ALPHA = 0.001
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    summary_path = output_dir / "experiment_scalability_summary.csv"
    raw_path = output_dir / "experiment_scalability_raw.csv"
    
    print(f"Starting Scalability Experiment N={N}, M={M}, Reps={REPS}, Alpha={ALPHA}...")
    tasks = [(N, M, d, g, r, ALPHA) for d in DEGS for g in GRAPHS for r in range(REPS)]
    
    # 100 nodes and 30 reps - use 12 jobs for safety
    results_nested = Parallel(n_jobs=12, verbose=10)(
        delayed(run_scalability_trial)(*t) for t in tasks
    )
    
    all_raw_data = [item for sublist in results_nested for item in sublist]
    df_raw = pd.DataFrame(all_raw_data)
    df_raw.to_csv(raw_path, index=False)
    
    group_cols = ["Nodes", "Samples", "Deg", "Graph", "Algo"]
    metrics = ["F1", "Prec", "Rec", "SHD", "Actual_CIT", "Cond_CIT", "Requests", "Time_Total"]
    
    summary_rows = []
    for keys, group in df_raw.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        for m in metrics:
            row[f"{m}_mean"] = group[m].mean()
            row[f"{m}_ci95"] = calculate_ci(group[m])
        summary_rows.append(row)
    
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(summary_path, index=False)
    print(f"Done. Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
