
import numpy as np
from typing import Optional
from itertools import combinations

class DeductiveReasoning:
    """
    Implements deductive reasoning rules to infer conditional dependence 
    from a set of conditional independence tests.
    
    References:
    - Estimating The Skeleton of DAGs via Decomposable Conditional Independence Tests
    - Section 4. Deductive Reasoning on Conditional Independence
    """
    def __init__(self, K: int, alpha: float, true_adj: Optional[np.ndarray] = None):
        """
        Args:
            K: Reliability Threshold.
            alpha: Significance level for independence tests.
            true_adj: Optional True adjacency matrix for debugging/stats.
        """
        self.K = K
        self.alpha = alpha
        self.true_adj = true_adj
        
        # Stats
        self.total_calls = 0
        self.total_corrections = 0
        self.total_correct_corrections = 0

    def deduce(self, data, X, Y, Z, sepsets, consets=None, ci_tester=None, depth=0):
        """
        Recursively deduce a dependence statement from CIT results via logical rules.
        """
        self.total_calls += 1
        if consets is None:
            consets = dict()
            
        if len(Z) > self.K:
            for z in Z:
                # Z \ {z}
                remaining_Z = tuple(sorted(list(set(Z) - {z})))

                # Collect status for (X, Y | Z\z), (X, z | Z\z), (Y, z | Z\z)
                triplet_results = []
                for A, B, C in [(X, Y, remaining_Z), (X, z, remaining_Z), (Y, z, remaining_Z)]:
                    pair_key = tuple(sorted([A, B]))
                    
                    is_ind = False
                    if pair_key in sepsets and sepsets[pair_key] == C:
                        is_ind = True
                    elif pair_key in consets and consets[pair_key] == C:
                        is_ind = False
                    else:
                        pval, _ = ci_tester.ci_test(data, A, B, C)
                        if pval > self.alpha:
                            if not self.deduce(data, A, B, C, sepsets, consets, ci_tester=ci_tester, depth=depth + 1):
                                sepsets[pair_key] = C
                                is_ind = True
                            else:
                                consets[pair_key] = C
                                is_ind = False
                        else:
                            consets[pair_key] = C
                            is_ind = False
                    
                    triplet_results.append("ind" if is_ind else "dep")

                results_tuple = tuple(triplet_results)
                if results_tuple[0] == "dep" and (results_tuple[1] == "ind" or results_tuple[2] == "ind"):
                    self.total_corrections += 1
                    return True
                if results_tuple[0] == "ind" and results_tuple[1] == "dep" and results_tuple[2] == "dep":
                    self.total_corrections += 1
                    return True
        return False

class Deductor:
    """
    New DEDUCE module for DF-PC.
    Deduces dependence or independence based on logical rules over triplets.
    """
    def __init__(self, ci_tester, alpha: float, priority: str = 'dep', pure: bool = True):
        self.ci_tester = ci_tester
        self.alpha = alpha
        self.priority = priority.lower()
        self.pure = pure
        self.cache = {} 
        self.stats = {
            'deduced_dep': 0,
            'deduced_ind': 0,
            'cit_calls': 0,
            'total_queries': 0
        }
        self._current_root_query = None
        self.cit_provenance = {}

    def reset(self):
        """Discard run-specific state while preserving deduction settings."""
        self.cache.clear()
        for name in self.stats:
            self.stats[name] = 0
        self.cit_provenance.clear()
        self._current_root_query = None

    def _get_cache_key(self, X, Y, Z):
        return (tuple(sorted((X, Y))), tuple(sorted(Z)))

    def _query_cit(self, data, X, Y, Z):
        key = self._get_cache_key(X, Y, Z)
        
        # Record provenance if it's a new call
        if key not in self.cache and self._current_root_query is not None:
             self.cit_provenance[key] = self._current_root_query

        # Skip cache if result is 'unknown' (must be resolved by actual CIT)
        if key in self.cache and self.cache[key] != 'unknown':
            return self.cache[key]
        
        self.stats['cit_calls'] += 1
        pval, _ = self.ci_tester.ci_test(data, X, Y, Z)
        result = 'ind' if pval > self.alpha else 'dep'
        self.cache[key] = result
        return result

    def deduce(self, data, X, Y, Z):
        is_root = (self._current_root_query is None)
        if is_root:
            self._current_root_query = self._get_cache_key(X, Y, Z)
            
        try:
            res = self._pure_deduce(data, X, Y, Z)
            if res == 'unknown':
                return self._query_cit(data, X, Y, Z)
            return res
        finally:
            if is_root:
                self._current_root_query = None

    def _pure_deduce(self, data, X, Y, Z):
        self.stats['total_queries'] += 1
        key = self._get_cache_key(X, Y, Z)
        if key in self.cache:
            return self.cache[key]

        if not Z:
            return 'unknown'
        
        seen_ind = False
        Z_list = list(Z)
        recurse = self._pure_deduce if self.pure else self.deduce
        
        if self.priority == 'dep':
            for z in Z_list:
                Z_prime = set(Z_list) - {z}
                r1 = recurse(data, X, Y, Z_prime)
                r2 = recurse(data, X, z, Z_prime)
                if (r1, r2) == ('dep', 'ind'):
                    self.stats['deduced_dep'] += 1
                    self.cache[key] = 'dep'
                    return 'dep'
                if (r1, r2) == ('ind', 'ind'):
                    seen_ind = True
                r3 = recurse(data, Y, z, Z_prime)
                if (r1, r3) == ('dep', 'ind'):
                    self.stats['deduced_dep'] += 1
                    self.cache[key] = 'dep'
                    return 'dep'
                if (r1, r3) == ('ind', 'ind'):
                    seen_ind = True
                if (r1, r2, r3) == ('ind', 'dep', 'dep'):
                    self.stats['deduced_dep'] += 1
                    self.cache[key] = 'dep'
                    return 'dep'
            if seen_ind:
                self.stats['deduced_ind'] += 1
                self.cache[key] = 'ind'
                return 'ind'

        elif self.priority == 'ind':
            seen_dep = False
            for z in Z_list:
                Z_prime = set(Z_list) - {z}
                r1 = recurse(data, X, Y, Z_prime)
                r2 = recurse(data, X, z, Z_prime)
                if (r1, r2) == ('ind', 'ind'):
                     self.stats['deduced_ind'] += 1
                     self.cache[key] = 'ind'
                     return 'ind'
                if (r1, r2) == ('dep', 'ind'):
                    seen_dep = True
                r3 = recurse(data, Y, z, Z_prime)
                if (r1, r3) == ('ind', 'ind'):
                     self.stats['deduced_ind'] += 1
                     self.cache[key] = 'ind'
                     return 'ind'
                if (r1, r3) == ('dep', 'ind'):
                    seen_dep = True
                if (r1, r2, r3) == ('ind', 'dep', 'dep'):
                    seen_dep = True
            if seen_dep:
                self.stats['deduced_dep'] += 1
                self.cache[key] = 'dep'
                return 'dep'
        
        self.cache[key] = 'unknown'
        return 'unknown'
