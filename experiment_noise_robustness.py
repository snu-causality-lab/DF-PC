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
import hashlib

from cddd.data import Graph
from cddd.algorithms import PCStable
from cddd.metrics import Evaluator

warnings.filterwarnings('ignore')

class ErrorOracleCITester:
    def __init__(self, true_graph, error_rate=0.0, rep=0):
        self.true_graph = true_graph
        self.error_rate = error_rate
        self.rep = rep
        self.n_actual_calls = 0
        self.total_test_time = 0
        self._flip_cache = {}
        
    def ci_test(self, data, i, j, cond_set):
        self.n_actual_calls += 1
        st = time.time()
        
        X, Y = min(i, j), max(i, j)
        Z_sorted = tuple(sorted(list(cond_set)))
        query_key = (X, Y, Z_sorted)
        
        if query_key not in self._flip_cache:
            is_indep = nx.d_separated(self.true_graph, {i}, {j}, set(cond_set))
            
            if len(cond_set) <= 1 and self.error_rate > 0.0:
                hash_input = str((self.rep, self.error_rate, X, Y, Z_sorted)).encode('utf-8')
                hash_val = hashlib.blake2b(hash_input).hexdigest()
                _flip_seed = int(hash_val, 16)
                rng = random.Random(_flip_seed)
                if rng.random() < self.error_rate:
                    is_indep = not is_indep
            
            self._flip_cache[query_key] = is_indep
        else:
            is_indep = self._flip_cache[query_key]

        self.total_test_time += time.time() - st
        pval = 1.0 if is_indep else 0.0
        return pval, 0.0

    def _set_data(self, data):
        pass

class PCStableRobustnessWrapper(PCStable):
    def __init__(self, alpha: float, ci_tester, algo_mode: str):
        super().__init__(alpha, ci_tester, use_deduction=(algo_mode=='DF-PC (Pure)'), deduction_pure=True)
        self.algo_mode = algo_mode

    def run(self, data):
        self.total_pc_requests = 0
        _, num_of_variables = np.shape(data)
        
        adj_mat = [[1 if i != j else 0 for j in range(num_of_variables)] for i in range(num_of_variables)]
        sepsets = dict()
        
        for k in range(num_of_variables - 1):
            marker = []
            for target in range(num_of_variables):
                adj_target = [i for i in range(num_of_variables) if (adj_mat[i][target] == 1) and (i != target)]
                
                for candidate in adj_target:
                    conditioning_set_pool = list(set(adj_target) - {candidate})
                    
                    if len(conditioning_set_pool) >= k:
                        k_length_conditioning_sets = combinations(conditioning_set_pool, k)
                        
                        for cond_set in k_length_conditioning_sets:
                            self.total_pc_requests += 1 
                            is_independent = False
                            pair_key = tuple(sorted([target, candidate]))
                            
                            if self.algo_mode == 'DF-PC (Pure)':
                                res = self.deductor.deduce(data, target, candidate, set(cond_set))
                                if res == 'ind':
                                    is_independent = True
                            else:
                                pval, _ = self.ci_tester.ci_test(data, target, candidate, cond_set)
                                if pval > self.alpha:
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

def calculate_ci(data):
    if len(data) < 2:
        return 0.0
    return 1.96 * (np.std(data, ddof=1) / np.sqrt(len(data)))

def run_single_rep(n, d, p_err, rep, algo_mode):
    seed = 300 + rep
    random.seed(seed)
    np.random.seed(seed)

    G = Graph.erdos_renyi_DAG(n, int(n * d / 2), max_in_degree=10)
    true_adj = nx.to_numpy_array(G)
    
    dummy_data = pd.DataFrame(np.zeros((10, n)))

    tester = ErrorOracleCITester(G, error_rate=p_err, rep=rep)
    tester.ci_test(dummy_data, 0, 1, [])
    tester.n_actual_calls = 0
    tester.total_test_time = 0.0

    runner = PCStableRobustnessWrapper(alpha=0.01, ci_tester=tester, algo_mode=algo_mode)
    
    adj_est, _ = runner.run(dummy_data)
    
    evaluator = Evaluator(true_adj)
    prec, rec, f1 = evaluator.global_skeleton_metric_evaluation(adj_est)
    shd = evaluator.get_skeleton_SHD(true_adj, adj_est)
    
    req_cits = runner.total_pc_requests
    perf_cits = tester.n_actual_calls
    bypassed_cits = req_cits - perf_cits
    
    return {
        "Nodes": n,
        "Deg": d,
        "ErrorRate": p_err,
        "Rep": rep,
        "Algo": algo_mode,
        "F1": f1,
        "Prec": prec,
        "Rec": rec,
        "SHD": shd,
        "Requests": req_cits,
        "Performed_CITs": perf_cits,
        "Bypassed_CITs": bypassed_cits
    }

def main():
    REPS = 30
    NODES = [20]
    DENSITY = [3]
    ERROR_RATES = [0.0, 0.01, 0.05, 0.10, 0.20]
    ALGO_MODES = ["Standard PC", "DF-PC (Pure)"]

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "experiment_noise_robustness_summary.csv"
    raw_path = output_dir / "experiment_noise_robustness_raw.csv"

    tasks = list(product(NODES, DENSITY, ERROR_RATES, ALGO_MODES, range(REPS)))
    print(f"Total trials to run: {len(tasks)}")
    
    n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    results = Parallel(n_jobs=n_cores, verbose=10)(
        delayed(run_single_rep)(n, d, p_err, rep, algo_mode) 
        for n, d, p_err, algo_mode, rep in tasks
    )
    
    df_raw = pd.DataFrame([r for r in results if r is not None])
    df_raw.to_csv(raw_path, index=False)
    df_raw.to_csv(output_dir / "experiment_5_raw.csv", index=False)

    group_cols = ["Nodes", "Deg", "ErrorRate", "Algo"]
    metrics = ["F1", "Prec", "Rec", "SHD", "Requests", "Performed_CITs", "Bypassed_CITs"]
    
    summary_rows = []
    for keys, group in df_raw.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        for m in metrics:
            row[f"{m}_mean"] = group[m].mean()
            row[f"{m}_ci95"] = calculate_ci(group[m])
        summary_rows.append(row)
    
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(csv_path, index=False)
    df_summary.to_csv(output_dir / "experiment_5_summary.csv", index=False)
    print(f"Finished! Summary saved to {csv_path}")

if __name__ == "__main__":
    main()
