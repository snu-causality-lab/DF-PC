
import pandas as pd
import numpy as np
import networkx as nx
import time
import os
import random
import warnings
from scipy import stats
from itertools import product
from pathlib import Path
from joblib import Parallel, delayed

from cddd.data import Graph, BayesianNetwork, DataGenerator
from cddd.independence import KernelCITest
from cddd.algorithms import PCStable
from cddd.metrics import Evaluator

warnings.filterwarnings('ignore')

def calculate_ci(data):
    if len(data) < 2:
        return 0.0
    return 1.96 * (np.std(data, ddof=1) / np.sqrt(len(data)))

def run_single_rep(n, m, d, g_type, d_type, rep, algo_modes):
    seed = 123 + rep
    random.seed(seed)
    np.random.seed(seed)

    if g_type == 'er':
        G = Graph.erdos_renyi_DAG(n, int(n * d / 2), max_in_degree=5)
    else:
        G = Graph.scale_free_DAG(n, int(max(1, d / 2)), max_in_degree=5)
    
    true_adj = nx.to_numpy_array(G)
    bn_model = BayesianNetwork(G, model_type=d_type)
    gen = DataGenerator(bn_model)
    data = gen.sample(m, seed=seed)
    data.columns = range(n)

    uncond_calc = (n * (n - 1)) // 2
    rep_results = []

    for algo_name, use_ded, is_pure in algo_modes:
        try:
            tester = KernelCITest()
            tester._set_data(data)
            
            runner = PCStable(alpha=0.01, ci_tester=tester, use_deduction=use_ded, deduction_pure=is_pure)
            
            start_t = time.time()
            adj_est, _ = runner.run(data)
            total_t = time.time() - start_t
            
            evaluator = Evaluator(true_adj)
            prec, rec, f1 = evaluator.global_skeleton_metric_evaluation(adj_est)
            shd = evaluator.get_skeleton_SHD(true_adj, adj_est)
            
            actual_cits = tester.n_actual_calls
            cit_t = tester.total_test_time
            overhead_t = total_t - cit_t
            
            rep_results.append({
                "Nodes": n, "Samples": m, "Deg": d, "Graph": g_type, "Data": d_type,
                "Rep": rep, "Algo": algo_name, "F1": f1, "Prec": prec, "Rec": rec, "SHD": shd,
                "Actual_CIT": actual_cits, "Cond_CIT": actual_cits - uncond_calc,
                "Requests": runner.total_pc_requests,
                "Time_Total": total_t, "Time_CIT": cit_t, "Time_Overhead": overhead_t
            })
        except Exception as e:
            print(f"Error in Trial R{rep}_{algo_name}: {e}")
            continue
    return rep_results

def main():
    # Focused nonlinear benchmark
    REPS = 30  # Reduced reps for KCI speed
    NODES = [10]
    SAMPLES = [1000] 
    AVG_DEGREES = [2]
    GRAPH_TYPES = ['er', 'sf']
    DATA_TYPES = ['nonlinear_sem'] 
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "experiment_nonlinear_summary.csv"

    ALGO_MODES = [
        ("Standard PC", False, False),
        ("DF-PC (Pure)", True, True)
    ]

    tasks = list(product(NODES, SAMPLES, AVG_DEGREES, GRAPH_TYPES, DATA_TYPES, range(REPS)))
    print(f"Starting Nonlinear Benchmark: {len(tasks)} trials total.")

    # High parallelism since each KCI call is slow but should be multi-threaded by backend
    # We use fewer jobs if KCI is already using many threads
    n_jobs = 10 
    results_nested = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_rep)(n, m, d, gt, dt, rep, ALGO_MODES) 
        for n, m, d, gt, dt, rep in tasks
    )

    all_raw_data = [item for sublist in results_nested for item in sublist]
    df_raw = pd.DataFrame(all_raw_data)
    
    group_cols = ["Nodes", "Samples", "Deg", "Graph", "Data", "Algo"]
    metrics = ["F1", "Prec", "Rec", "SHD", "Actual_CIT", "Cond_CIT", "Requests", "Time_Total"]
    
    summary_rows = []
    for keys, group in df_raw.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        for m in metrics:
            row[f"{m}_mean"] = group[m].mean()
            row[f"{m}_ci95"] = calculate_ci(group[m])
        summary_rows.append(row)
    
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(csv_path, index=False)
    print(f"Done! Results saved to {csv_path}")
    print(df_summary[["Graph", "Algo", "F1_mean", "Cond_CIT_mean", "Time_Total_mean"]])

if __name__ == "__main__":
    main()
