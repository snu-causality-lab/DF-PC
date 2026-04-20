
import pandas as pd
import numpy as np
import networkx as nx
import time
import os
import warnings
from pathlib import Path
from joblib import Parallel, delayed
from pgmpy.utils import get_example_model

from cddd.independence import ChiSquareTester
from cddd.algorithms import PCStable
from cddd.metrics import Evaluator

warnings.filterwarnings('ignore')

def calculate_ci(data):
    if len(data) < 2:
        return 0.0
    return 1.96 * (np.std(data, ddof=1) / np.sqrt(len(data)))

def get_pgmpy_gt(model, node_names):
    n = len(node_names)
    adj = np.zeros((n, n))
    node_to_idx = {name: i for i, name in enumerate(node_names)}
    for u, v in model.edges():
        adj[node_to_idx[u], node_to_idx[v]] = 1
    return adj

def run_realworld_trial(dataset, m, rep, alpha=0.01):
    seed = 800 + rep
    np.random.seed(seed)
    
    # 1. Load Model
    model = get_example_model(dataset)
    node_names = sorted(list(model.nodes()))
    true_adj = get_pgmpy_gt(model, node_names)
    
    # 2. Sample Data
    data_df = model.simulate(n_samples=m, show_progress=False)
    data_df = data_df[node_names]
    
    # 3. Factorize
    for col in data_df.columns:
        data_df[col] = pd.factorize(data_df[col])[0]
        
    n = len(node_names)
    uncond_calc = (n * (n - 1)) // 2
    results = []
    
    # Use standard DF-PC (Dep-First) as requested.
    for algo_name, use_ded, priority in [("Standard PC", False, 'dep'), ("DF-PC (Pure)", True, 'dep')]:
        tester = ChiSquareTester()
        tester._set_data(data_df)
        
        runner = PCStable(alpha=alpha, ci_tester=tester, use_deduction=use_ded, deduction_priority=priority)
        
        start_t = time.time()
        adj_est, _ = runner.run(data_df)
        total_t = time.time() - start_t
        
        evaluator = Evaluator(true_adj)
        prec, rec, f1 = evaluator.global_skeleton_metric_evaluation(adj_est)
        shd = evaluator.get_skeleton_SHD(true_adj, adj_est)
        
        results.append({
            "Dataset": dataset,
            "Nodes": n,
            "Samples": m,
            "Rep": rep,
            "Algo": algo_name,
            "F1": f1,
            "Prec": prec,
            "Rec": rec,
            "SHD": shd,
            "Actual_CIT": tester.n_actual_calls,
            "Cond_CIT": tester.n_actual_calls - uncond_calc,
            "Requests": runner.total_pc_requests,
            "Time_Total": total_t
        })
    return results

def main():
    DATASETS = ['barley', 'mildew']
    SAMPLES = [2000, 5000]
    REPS = 30
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    summary_path = output_dir / "experiment_realworld_summary.csv"
    raw_path = output_dir / "experiment_realworld_raw.csv"
    
    print(f"Starting Advanced Real-World Benchmarks: {DATASETS}")
    print(f"Configurations: {REPS} reps, Samples {SAMPLES}")
    
    tasks = [(ds, m, r) for ds in DATASETS for m in SAMPLES for r in range(REPS)]
    
    # Parallelize. Hepar2 is 70 nodes, 30 reps - can be heavy.
    results_nested = Parallel(n_jobs=24, verbose=10)(
        delayed(run_realworld_trial)(*t) for t in tasks
    )
    
    all_raw_data = [item for sublist in results_nested for item in sublist]
    df_raw = pd.DataFrame(all_raw_data)
    df_raw.to_csv(raw_path, index=False)
    
    group_cols = ["Dataset", "Nodes", "Samples", "Algo"]
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
    
    print(f"\nFinal summary saved to {summary_path}")
    print(df_summary[["Dataset", "Algo", "F1_mean", "Cond_CIT_mean"]].to_string(index=False))

if __name__ == "__main__":
    main()
