import pandas as pd
import numpy as np
import networkx as nx
import time
import os
import random
import warnings
from scipy import stats
from itertools import product, combinations
from pathlib import Path
from joblib import Parallel, delayed

from cddd.data import Graph, BayesianNetwork, DataGenerator
from cddd.independence import ChiSquareTester, PartialCorrelation
from cddd.algorithms import PCStable
from cddd.metrics import Evaluator

warnings.filterwarnings('ignore')

class PCStableRebuttal(PCStable):
    """
    Custom wrapper to support all 3 baselines cleanly without touching core library.
    """
    def __init__(self, alpha: float, ci_tester, algo_mode: str):
        # Initialize with DF-PC pure settings if it's DF-PC
        super().__init__(alpha, ci_tester, use_deduction=(algo_mode=='DF-PC (Pure)'), deduction_pure=True)
        self.algo_mode = algo_mode
        self.old_deductor = None
        if self.algo_mode == 'PC with deduce-dep':
            from cddd.inference import DeductiveReasoning
            self.old_deductor = DeductiveReasoning(K=1, alpha=alpha)

    def run(self, data):
        self.total_pc_requests = 0
        _, num_of_variables = np.shape(data)
        
        # Initialize complete graph: 1=edge, 0=no edge
        adj_mat = [[1 if i != j else 0 for j in range(num_of_variables)] for i in range(num_of_variables)]
        sepsets = dict()
        consets = dict() 
        
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
                                # DF-PC Logic
                                res = self.deductor.deduce(data, target, candidate, set(cond_set))
                                if res == 'ind':
                                    is_independent = True
                            else:
                                # Standard PC / PC with deduce-dep Logic
                                pval, _ = self.ci_tester.ci_test(data, target, candidate, cond_set)
                                if pval > self.alpha:
                                    is_independent = True
                                
                                # Apply consequent deduction overriding
                                if is_independent and self.algo_mode == 'PC with deduce-dep':
                                    if self.old_deductor.deduce(data, target, candidate, cond_set, sepsets, consets, ci_tester=self.ci_tester):
                                        is_independent = False

                            if is_independent:
                                sepsets[pair_key] = cond_set
                                marker.append([pair_key, cond_set])
                                break 
                            else:
                                consets[pair_key] = cond_set

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

def run_single_rep(n, m, d, g_type, d_type, rep, algo_mode):
    seed = 300 + rep
    random.seed(seed)
    np.random.seed(seed)

    if g_type == 'er':
        G = Graph.erdos_renyi_DAG(n, int(n * d / 2), max_in_degree=10)
    else:
        G = Graph.scale_free_DAG(n, int(max(1, d / 2)), max_in_degree=10)
    
    true_adj = nx.to_numpy_array(G)
    
    bn_model = BayesianNetwork(G, model_type=d_type)
    gen = DataGenerator(bn_model)
    data = gen.sample(m, seed=seed)
    data.columns = range(n)

    if d_type == 'discrete':
        tester = ChiSquareTester()
    else:
        tester = PartialCorrelation()
    
    tester._set_data(data)
    runner = PCStableRebuttal(alpha=0.01, ci_tester=tester, algo_mode=algo_mode)
    
    # Warm-up OS OpenBLAS threads to guarantee completely fair profiling
    tester.ci_test(data, 0, 1, [])
    tester.n_actual_calls = 0
    tester.total_test_time = 0.0
    if hasattr(tester, 'history'):
        tester.history.clear()
    # Recreate CIT backend to flush causal-learn's internal pvalue_cache
    tester._set_data(data)
        
    start_t = time.time()
    adj_est, _ = runner.run(data)
    total_t = time.time() - start_t
    
    evaluator = Evaluator(true_adj)
    prec, rec, f1 = evaluator.global_skeleton_metric_evaluation(adj_est)
    shd = evaluator.get_skeleton_SHD(true_adj, adj_est)
    
    req_cits = runner.total_pc_requests
    perf_cits = tester.n_actual_calls
    bypassed_cits = req_cits - perf_cits
    cit_t = tester.total_test_time
    overhead_t = total_t - cit_t
    
    # Original metrics exactly mapped to their definitions
    uncond_calc = (n * (n - 1)) // 2
    actual_cit = perf_cits
    cond_cit = perf_cits - uncond_calc
    
    return {
        "Nodes": n,
        "Samples": m,
        "Deg": d,
        "Graph": g_type,
        "Data": d_type,
        "Rep": rep,
        "Algo": algo_mode,
        "F1": f1,
        "Prec": prec,
        "Rec": rec,
        "SHD": shd,
        "Requests": req_cits,
        "Actual_CIT": actual_cit,
        "Cond_CIT": cond_cit,
        "Performed_CITs": perf_cits,
        "Bypassed_CITs": bypassed_cits,
        "Time_Total": total_t,
        "Time_CIT": cit_t,
        "Time_Overhead": overhead_t
    }

def main():
    REPS = 30
    NODES = [20]
    SAMPLES = [5000]
    AVG_DEGREES = [4]
    GRAPH_TYPES = ['er', 'sf']
    DATA_TYPES = ['linear_sem', 'discrete']
    ALGO_MODES = ["Standard PC", "PC with deduce-dep", "DF-PC (Pure)"]

    output_dir = Path("results_rebuttal")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "experiment_1_summary.csv"
    raw_path = output_dir / "experiment_1_raw.csv"

    all_tasks = list(product(NODES, SAMPLES, AVG_DEGREES, GRAPH_TYPES, DATA_TYPES, ALGO_MODES, range(REPS)))
    print(f"Total trials to run: {len(all_tasks)}")
    
    n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    
    results = []
    # Isolated Batching Strategy: Separate Linear and Discrete to avoid BLAS/FPU contention
    for dt, algo in product(DATA_TYPES, ALGO_MODES):
        sub_tasks = [t for t in all_tasks if t[4] == dt and t[5] == algo]
        if not sub_tasks:
            continue
        print(f"Running Batch: Data={dt}, Algo={algo} ({len(sub_tasks)} trials)")
        batch_results = Parallel(n_jobs=n_cores, verbose=10)(
            delayed(run_single_rep)(n, m, d, gt, dt_in, rep, algo_in) 
            for n, m, d, gt, dt_in, algo_in, rep in sub_tasks
        )
        results.extend(batch_results)
    
    df_raw = pd.DataFrame([r for r in results if r is not None])
    df_raw.to_csv(raw_path, index=False)

    group_cols = ["Nodes", "Samples", "Deg", "Graph", "Data", "Algo"]
    metrics = ["F1", "Prec", "Rec", "SHD", "Requests", "Actual_CIT", "Cond_CIT", 
               "Performed_CITs", "Bypassed_CITs", "Time_Total", "Time_CIT", "Time_Overhead"]
    
    summary_rows = []
    for keys, group in df_raw.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        for m in metrics:
            row[f"{m}_mean"] = group[m].mean()
            row[f"{m}_ci95"] = calculate_ci(group[m])
        summary_rows.append(row)
    
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(csv_path, index=False)
    print(f"Finished! Summary saved to {csv_path}")

if __name__ == "__main__":
    main()
