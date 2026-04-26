import pandas as pd
import numpy as np
import networkx as nx
import time
import os
import random
import warnings
from itertools import product, combinations
from pathlib import Path
from joblib import Parallel, delayed

from cddd.data import Graph
from cddd.algorithms import PCStable

warnings.filterwarnings('ignore')

class OracleCITester:
    def __init__(self, true_graph):
        self.true_graph = true_graph
        self.n_actual_calls = 0
        self.total_test_time = 0
        # Track performed tests strictly by conditioning set size (|Z|)
        self.z_performed = {}
        
    def ci_test(self, data, i, j, cond_set):
        self.n_actual_calls += 1
        k = len(cond_set)
        if k not in self.z_performed:
            self.z_performed[k] = 0
        self.z_performed[k] += 1
        
        st = time.time()
        is_indep = nx.d_separated(self.true_graph, {i}, {j}, set(cond_set))
        self.total_test_time += time.time() - st
        
        # p-value > 0.01 is independent
        pval = 1.0 if is_indep else 0.0
        return pval, 0.0
        
    def _set_data(self, data):
        pass # Data is not needed for Oracle check

class PCStableZTracker(PCStable):
    def __init__(self, alpha: float, ci_tester):
        super().__init__(alpha, ci_tester, use_deduction=True, deduction_pure=True)
        self.z_requested = {}

    def run(self, data):
        self.total_pc_requests = 0
        _, num_of_variables = np.shape(data)
        
        adj_mat = [[1 if i != j else 0 for j in range(num_of_variables)] for i in range(num_of_variables)]
        sepsets = dict()
        
        for k in range(num_of_variables - 1):
            if k not in self.z_requested:
                self.z_requested[k] = 0
                
            marker = []
            for target in range(num_of_variables):
                adj_target = [i for i in range(num_of_variables) if (adj_mat[i][target] == 1) and (i != target)]
                
                for candidate in adj_target:
                    conditioning_set_pool = list(set(adj_target) - {candidate})
                    
                    if len(conditioning_set_pool) >= k:
                        k_length_conditioning_sets = combinations(conditioning_set_pool, k)
                        
                        for cond_set in k_length_conditioning_sets:
                            self.z_requested[k] += 1
                            self.total_pc_requests += 1 
                            
                            is_independent = False
                            pair_key = tuple(sorted([target, candidate]))
                            
                            res = self.deductor.deduce(data, target, candidate, set(cond_set))
                            if res == 'ind':
                                is_independent = True

                            if is_independent:
                                sepsets[pair_key] = cond_set
                                marker.append([pair_key, cond_set])
                                break 

            for pair_tuple, cond_set in marker:
                sepsets[pair_tuple] = cond_set
                var1, var2 = pair_tuple
                adj_mat[var1][var2] = 0
                adj_mat[var2][var1] = 0

        return np.array(adj_mat), sepsets

def run_single_rep(setup_name, n, d, rep):
    seed = 300 + rep
    random.seed(seed)
    np.random.seed(seed)

    # Use Erdos-Renyi graph to prevent combinatorial explosion on hubs
    G = Graph.erdos_renyi_DAG(n, int(n * d / 2), max_in_degree=10)
    
    # We create a dummy dataframe because algorithms.py requires shape[1] mapping
    dummy_data = pd.DataFrame(np.zeros((10, n)))

    tester = OracleCITester(G)
    
    # Warm-up OS OpenBLAS threads to guarantee completely fair profiling
    tester.ci_test(dummy_data, 0, 1, [])
    tester.n_actual_calls = 0
    tester.total_test_time = 0.0
    if hasattr(tester, 'history'):
        tester.history.clear()
        
    runner = PCStableZTracker(alpha=0.01, ci_tester=tester)
    
    runner.run(dummy_data)
    
    # Break down the Z-layered output
    row_list = []
    
    all_k = set(runner.z_requested.keys()).union(set(tester.z_performed.keys()))
    for k in sorted(list(all_k)):
        req = runner.z_requested.get(k, 0)
        perf = tester.z_performed.get(k, 0)
        bypassed = req - perf
        
        row_list.append({
            "Setup": setup_name,
            "Nodes": n,
            "Deg": d,
            "Rep": rep,
            "Z_Size": k,
            "Requested_CITs": req,
            "Performed_CITs": perf,
            "Bypassed_CITs": bypassed
        })
    return row_list

def main():
    REPS = 30
    output_dir = Path("results_rebuttal")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "experiment_4_summary.csv"
    raw_path = output_dir / "experiment_4_raw.csv"

    # Setup A: Node Scaling Impact
    NODES_A = [10, 20, 30, 40]
    DENSITY_A = [3]
    tasks_a = [('SetupA_Nodes', n, d, rep) for n, d, rep in product(NODES_A, DENSITY_A, range(REPS))]

    # Setup B: Density Scaling Impact
    NODES_B = [20]
    DENSITY_B = [2, 3, 4, 5]
    tasks_b = [('SetupB_Density', n, d, rep) for n, d, rep in product(NODES_B, DENSITY_B, range(REPS))]

    tasks = tasks_a + tasks_b
    print(f"Total trials to run: {len(tasks)}")
    
    n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    results = Parallel(n_jobs=n_cores, verbose=10)(
        delayed(run_single_rep)(setup, n, d, rep) 
        for setup, n, d, rep in tasks
    )
    
    all_rows = [row for sublist in results for row in sublist]
    df_raw = pd.DataFrame(all_rows)
    df_raw.to_csv(raw_path, index=False)

    group_cols = ["Setup", "Nodes", "Deg", "Z_Size"]
    
    summary_rows = []
    for keys, group in df_raw.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["Requested_mean"] = group["Requested_CITs"].mean()
        row["Performed_mean"] = group["Performed_CITs"].mean()
        row["Bypassed_mean"] = group["Bypassed_CITs"].mean()
        summary_rows.append(row)
    
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(csv_path, index=False)
    print(f"Finished! Summary saved to {csv_path}")

if __name__ == "__main__":
    main()
