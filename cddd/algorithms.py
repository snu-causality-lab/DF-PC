from itertools import combinations
from functools import reduce
from typing import List, Dict, Tuple, Set, Union, Optional

import networkx as nx
import numpy as np

from cddd.inference import Deductor
# DeductiveReasoning (old) unused now within this file, but kept import if needed elsewhere, 
# though we are cleaning up logic. DFPC used Deductor.

from cddd.independence import ci_test_factory

class PCStable:
    """
    PC-Stable algorithm for causal discovery (skeleton only).
    Now supports DF-PC (Deduce-First PC) logic via `use_deduction` flag.
    """
    def __init__(self, alpha: float, ci_tester, use_deduction: bool = False,
                 deduction_priority: str = 'dep', deduction_pure: bool = True,
                 *, early_stopping: bool = True):
        """
        Args:
            alpha: Significance level.
            ci_tester: Conditional Independence Tester object.
            use_deduction: If True, uses DF-PC logic with Deductor. 
            deduction_priority: 'dep' (Dep-First) or 'ind' (Ind-First). Default 'dep'.
            deduction_pure: If True, deduction is pure (top-level CIT only). If False, recursion can trigger CITs.
            early_stopping: If True (the paper's default), stop the current
                ordered pair's conditioning-set search after independence.
                If False, continue all eligible sets at that level; edges are
                still removed only at the level's end. This can increase CI
                tests and change the available deduction history and sepsets.
        """
        self.alpha = alpha
        self.ci_tester = ci_tester
        self.use_deduction = use_deduction
        self.early_stopping = early_stopping
        
        self.deductor = None
        if self.use_deduction:
            from cddd.inference import Deductor
            self.deductor = Deductor(self.ci_tester, self.alpha, priority=deduction_priority, pure=deduction_pure)
            
        # Metric tracking
        self.total_pc_requests = 0

    def run(self, data: 'pd.DataFrame') -> Tuple[np.ndarray, Dict]:
        '''
        Run the PC-stable algorithm.
        
        Args:
            data: Input pandas DataFrame.
            
        Returns:
            adj_mat: Estimated adjacency matrix (numpy array).
            sepsets: Dictionary of separating sets.
        '''
        self.total_pc_requests = 0
        
        _, num_of_variables = np.shape(data)
        
        # Initialize complete graph: 1=edge, 0=no edge
        adj_mat = [[1 if i != j else 0 for j in range(num_of_variables)] for i in range(num_of_variables)]
        sepsets = dict()
        consets = dict() 
        
        # Level-wise iteration (PC Algorithm)
        for k in range(num_of_variables - 1):
            marker = []
            
            # Symmetric Edge Check Loop (Verified Correctness & Standard PC Equivalence)
            for target in range(num_of_variables):
                adj_target = [i for i in range(num_of_variables) if (adj_mat[i][target] == 1) and (i != target)]
                
                for candidate in adj_target:
                    conditioning_set_pool = list(set(adj_target) - {candidate})
                    
                    if len(conditioning_set_pool) >= k:
                        k_length_conditioning_sets = combinations(conditioning_set_pool, k)
                        
                        for cond_set in k_length_conditioning_sets:
                            self.total_pc_requests += 1 # Count every attempt to test/deduce
                            
                            is_independent = False
                            
                            if self.use_deduction:
                                # DF-PC Path
                                res = self.deductor.deduce(data, target, candidate, set(cond_set))
                                if res == 'ind':
                                    is_independent = True
                            else:
                                # Standard PC Path
                                pval, _ = self.ci_tester.ci_test(data, target, candidate, cond_set)
                                if pval > self.alpha:
                                    is_independent = True
                            
                            if is_independent:
                                sepsets[tuple(sorted([target, candidate]))] = cond_set
                                marker.append([tuple(sorted([target, candidate])), cond_set])
                                if self.early_stopping:
                                    break
                            else:
                                consets[tuple(sorted([target, candidate]))] = cond_set

            # Remove edges marked for removal in this level k
            for pair_tuple, cond_set in marker:
                sepsets[pair_tuple] = cond_set
                var1, var2 = pair_tuple
                adj_mat[var1][var2] = 0
                adj_mat[var2][var1] = 0
            
            # Stopping condition implicitly handled by loop range or empty k_length_sets

        return np.array(adj_mat), sepsets



class HitonPC:
    """
    HITON-PC Algorithm for Local Structure Discovery (Parents and Children).
    """
    def __init__(self, alpha: float, ci_tester, is_deduce_dep: bool = False, K: int = 1):
        self.alpha = alpha
        self.ci_tester = ci_tester
        self.K = K
        self.h_ps = 5
        self.deductor = DeductiveReasoning(self.K, self.alpha) if is_deduce_dep else None
        self.max_k = float('inf')

    def _get_num_of_parameters(self, data, target, x, z):
        '''
        Return the total number of parameters of contingency table in CIT.
        '''
        num_of_domain_target = len(data[target].unique())
        num_of_domain_x = len(data[x].unique())
        if z:
            levels_of_domain_z = list(map(lambda x_: len(data[x_].unique()), z))
            num_of_domain_z = reduce(lambda x_, y_: x_ * y_, levels_of_domain_z)
        else:
            num_of_domain_z = 1
        return num_of_domain_x * num_of_domain_target * num_of_domain_z

    def run(self, data, assoc, target):
        '''
        Run HITON-PC.
        
        Args:
            data: dataframe
            assoc: cache of association strengths (modified in place)
            target: target variable index
            
        Returns:
            TPC: list of Parents/Children indices
            sepsets: sepset dict
            ci_number: number of tests performed
        '''
        size_of_dataset, num_of_variables = np.shape(data)
        sepsets = dict()
        consets = dict()
        
        OPEN = []
        TPC = []
        ci_number = 0

        total_variables = [var for var in range(num_of_variables) if var != target]
        
        # 1. Forward Phase (Screening)
        for x in total_variables:
            num_of_parameters_for_cit = self._get_num_of_parameters(data, target, x, [])
            if size_of_dataset >= self.h_ps * num_of_parameters_for_cit:
                # Test Marginal Independence
                pval_gp, dep_gp = self.ci_tester.ci_test(data, target, x, [])
                assoc[target][x] = dep_gp # Store dependency strength
                ci_number += 1

                if pval_gp <= self.alpha:
                    OPEN.append(x)
                else:
                    sepsets[tuple(sorted([target, x]))] = []

        # Sort candidates by association strength (Descending)
        OPEN = sorted(OPEN, key=lambda x_: assoc[target][x_], reverse=True)

        # 2. Backward Phase (Wrapper)
        for x in OPEN:
            TPC.append(x)
            TPC_index = len(TPC)

            while TPC_index > 0:
                TPC_index -= 1
                TPC_var = TPC[TPC_index]
                remaining_TPC = [var for var in TPC if var != TPC_var]
                
                # Check Conditional Independence given subsets of remaining TPC
                max_length_for_cond_set = min(self.max_k, len(remaining_TPC))
                cond_sets = []
                for cond_set_len in range(1, max_length_for_cond_set + 1):
                    cond_sets += list(combinations(remaining_TPC, cond_set_len))

                for cond_set in cond_sets:
                    if (TPC_var != x) and (x not in cond_set):
                        continue

                    num_of_parameters_for_cit = self._get_num_of_parameters(data, target, TPC_var, cond_set)
                    if size_of_dataset >= (self.h_ps * num_of_parameters_for_cit):
                        
                        # Check cache
                        pair_key = tuple(sorted([target, TPC_var]))
                        is_independent = False
                        
                        if pair_key in sepsets and sepsets[pair_key] == cond_set:
                            is_independent = True
                        elif pair_key in consets and consets[pair_key] == cond_set:
                            is_independent = False
                        else:
                            pval_rm, dep_rm = self.ci_tester.ci_test(data, target, TPC_var, cond_set)
                            ci_number += 1
                            is_independent = (pval_rm > self.alpha)

                        if is_independent:
                            if self.deductor:
                                if not self.deductor.deduce(data, target, TPC_var, cond_set, sepsets, consets, ci_tester=self.ci_tester):
                                    sepsets[pair_key] = cond_set
                                    TPC.remove(TPC_var)
                                    break
                                else:
                                    consets[pair_key] = cond_set
                            else:
                                sepsets[pair_key] = cond_set
                                TPC.remove(TPC_var)
                                break
                        else:
                            consets[pair_key] = cond_set

        return list(set(TPC)), sepsets, ci_number

