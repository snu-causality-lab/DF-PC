
import pandas as pd
import numpy as np
import networkx as nx
import time
import os
import random
import warnings
from pathlib import Path

from cddd.data import Graph, BayesianNetwork, DataGenerator
from cddd.independence import PartialCorrelation
from cddd.algorithms import PCStable
from cddd.metrics import Evaluator

warnings.filterwarnings('ignore')

def create_collider_hub(n):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    target = 0
    for i in range(1, n):
        G.add_edge(i, target)
    return G

def create_source_hub(n):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    source = 0
    for i in range(1, n):
        G.add_edge(source, i)
    return G

def run_experiment(name, G, m, reps):
    n = G.number_of_nodes()
    true_adj = nx.to_numpy_array(G)
    results = []
    
    algo_modes = [
        ("Standard PC", False, False),
        ("DF-PC (Pure)", True, True)
    ]
    
    for rep in range(reps):
        seed = 777 + rep
        bn = BayesianNetwork(G, model_type='linear_sem')
        gen = DataGenerator(bn)
        data = gen.sample(m, seed=seed)
        
        for algo_name, use_ded, is_pure in algo_modes:
            tester = PartialCorrelation()
            tester._set_data(data)
            runner = PCStable(alpha=0.01, ci_tester=tester, use_deduction=use_ded, deduction_pure=is_pure)
            
            start_t = time.time()
            adj_est, _ = runner.run(data)
            total_t = time.time() - start_t
            
            evaluator = Evaluator(true_adj)
            prec, rec, f1 = evaluator.global_skeleton_metric_evaluation(adj_est)
            
            results.append({
                "Case": name, "Rep": rep, "Algo": algo_name,
                "F1": f1, "CIT": tester.n_actual_calls, "Time": total_t
            })
            
    return results

def main():
    N = 11
    M = 2000
    REPS = 10
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Running Extreme Case Analysis (N={N}, M={M})...")
    
    # 1. Collider Hub (Best)
    G_collider = create_collider_hub(N)
    res_collider = run_experiment("Collider-Hub", G_collider, M, REPS)
    
    # 2. Source Hub (Worst)
    G_source = create_source_hub(N)
    res_source = run_experiment("Source-Hub", G_source, M, REPS)
    
    df = pd.DataFrame(res_collider + res_source)
    summary = df.groupby(["Case", "Algo"])[["F1", "CIT", "Time"]].mean().reset_index()
    
    csv_path = output_dir / "experiment_extreme_cases.csv"
    summary.to_csv(csv_path, index=False)
    
    print("\nExtreme Case Summary:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
