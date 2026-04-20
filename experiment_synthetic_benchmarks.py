
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
from cddd.independence import ChiSquareTester, PartialCorrelation, KernelCITest
from cddd.algorithms import PCStable
from cddd.metrics import Evaluator

warnings.filterwarnings('ignore')

def calculate_ci(data):
    """Calculate 95% confidence interval."""
    if len(data) < 2:
        return 0.0
    return 1.96 * (np.std(data, ddof=1) / np.sqrt(len(data)))

def run_single_rep(n, m, d, g_type, d_type, rep, algo_modes):
    """
    Executes a single experimental trial (one rep of one setting).
    Returns a list of result dictionaries (one for each algorithm).
    """
    # Seed management for reproducibility in parallel processes
    # We use a base seed plus rep to ensure consistency with the non-parallel version
    seed = 42 + rep
    random.seed(seed)
    np.random.seed(seed)

    # 1. Generate Graph - Cap in-degree to 10 to prevent CPT explosion in discrete settings
    if g_type == 'er':
        G = Graph.erdos_renyi_DAG(n, int(n * d / 2), max_in_degree=10)
    else:
        G = Graph.scale_free_DAG(n, int(max(1, d / 2)), max_in_degree=10)
    
    true_adj = nx.to_numpy_array(G)
    
    # 2. Generate Data
    bn_model = BayesianNetwork(G, model_type=d_type)
    gen = DataGenerator(bn_model)
    data = gen.sample(m, seed=seed)
    data.columns = range(n)

    uncond_calc = (n * (n - 1)) // 2
    rep_results = []

    for algo_name, use_ded, is_pure in algo_modes:
        try:
            # 3. Setup Tester
            if d_type == 'discrete':
                tester = ChiSquareTester()
            elif d_type == 'linear_sem':
                tester = PartialCorrelation()
            else: # nonlinear
                tester = KernelCITest()
            
            tester._set_data(data)
            
            # 4. Run Algorithm 
            runner = PCStable(alpha=0.01, ci_tester=tester, use_deduction=use_ded, deduction_pure=is_pure)
            
            start_t = time.time()
            adj_est, _ = runner.run(data)
            total_t = time.time() - start_t
            
            # 5. Evaluate
            evaluator = Evaluator(true_adj)
            prec, rec, f1 = evaluator.global_skeleton_metric_evaluation(adj_est)
            shd = evaluator.get_skeleton_SHD(true_adj, adj_est)
            
            actual_cits = tester.n_actual_calls
            cit_t = tester.total_test_time
            overhead_t = total_t - cit_t
            
            rep_results.append({
                "Nodes": n,
                "Samples": m,
                "Deg": d,
                "Graph": g_type,
                "Data": d_type,
                "Rep": rep,
                "Algo": algo_name,
                "F1": f1,
                "Prec": prec,
                "Rec": rec,
                "SHD": shd,
                "Actual_CIT": actual_cits,
                "Cond_CIT": actual_cits - uncond_calc,
                "Requests": runner.total_pc_requests,
                "Time_Total": total_t,
                "Time_CIT": cit_t,
                "Time_Overhead": overhead_t
            })
        except Exception as e:
            print(f"Error in Trial N{n}_M{m}_{g_type}_{d_type}_R{rep}_{algo_name}: {e}")
            continue
    
    return rep_results

def conduct_benchmarks():
    # --- Configuration ---
    REPS = 30
    ALPHAS = [0.01]
    
    # Current User Settings (from previous verification)
    NODES = [10, 20, 30] 
    SAMPLES = [1000, 2000, 5000, 10000] 
    AVG_DEGREES = [2, 4]
    GRAPH_TYPES = ['er', 'sf']
    DATA_TYPES = ['discrete', 'linear_sem'] 
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "synthetic_benchmarks_summary.csv"
    raw_results_path = output_dir / "synthetic_benchmarks_raw.csv"

    # Algorithms to compare
    ALGO_MODES = [
        ("Standard PC", False, False),
        ("DF-PC (Pure)", True, True)
    ]

    print(f"Starting Parallel Benchmarks: {REPS} reps per setting.")
    print(f"Settings: Nodes={NODES}, Samples={SAMPLES}, Degrees={AVG_DEGREES}, Graphs={GRAPH_TYPES}, Data={DATA_TYPES}")

    # Prepare all tasks
    tasks = list(product(NODES, SAMPLES, AVG_DEGREES, GRAPH_TYPES, DATA_TYPES, range(REPS)))
    
    # Parallelize across settings and reps
    print(f"Total trials to run: {len(tasks)}")
    
    # Using n_jobs=24 for better RAM safety on 48-core machine with large DAGs
    results_nested = Parallel(n_jobs=24, verbose=10)(
        delayed(run_single_rep)(n, m, d, gt, dt, rep, ALGO_MODES) 
        for n, m, d, gt, dt, rep in tasks
    )

    # Flatten the results list
    all_raw_data = [item for sublist in results_nested for item in sublist]
    
    print("\nBenchmark Execution Complete. Saving raw data...")
    df_raw = pd.DataFrame(all_raw_data)
    df_raw.to_csv(raw_results_path, index=False)

    print("\nProcessing Summary Statistics...")
    # Group by all setting factors and Algo
    group_cols = ["Nodes", "Samples", "Deg", "Graph", "Data", "Algo"]
    metrics = ["F1", "Prec", "Rec", "SHD", "Actual_CIT", "Cond_CIT", "Requests", "Time_Total", "Time_CIT", "Time_Overhead"]
    
    summary_rows = []
    for keys, group in df_raw.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        for m in metrics:
            row[f"{m}_mean"] = group[m].mean()
            row[f"{m}_std"] = group[m].std()
            row[f"{m}_ci95"] = calculate_ci(group[m])
        summary_rows.append(row)
    
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(csv_path, index=False)
    
    print(f"Done! Results saved to {csv_path}")
    print("\nPreview of Summary (F1 and Cond_CIT):")
    cols_to_show = ["Nodes", "Samples", "Graph", "Data", "Algo", "F1_mean", "Cond_CIT_mean", "Time_Total_mean"]
    print(df_summary[cols_to_show].to_string(index=False))

if __name__ == "__main__":
    conduct_benchmarks()
