#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A conditional independency test function for discrete data.

The code included in this package is logically copied and pasted from
the pcalg package for R developed by Markus Kalisch, Alain Hauser,
Martin Maechler, Diego Colombo, Doris Entner, Patrik Hoyer, Antti
Hyttinen, and Jonas Peters.

License: GPLv2
"""


import logging
import warnings
import time
import numpy as np
import pandas as pd
from causallearn.utils.cit import CIT

warnings.filterwarnings('ignore')

_logger = logging.getLogger(__name__)

class CITester:
    def __init__(self, **kwargs):
        self.n_actual_calls = 0
        self.total_test_time = 0.0
        self.history = set()

    def reset_stats(self):
        self.n_actual_calls = 0
        self.total_test_time = 0.0
        self.history = set()

    def ci_test(self, data, X, Y, cond_set=frozenset()):
        """
        Public interface for CI testing. 
        Handles counting of unique queries (simulating actual computation cost).
        """
        # Normalize key
        # X, Y are symmetric. cond_set is set.
        call_key = (frozenset({X, Y}), frozenset(cond_set) if cond_set else frozenset())
        
        if call_key not in self.history:
            self.history.add(call_key)
            self.n_actual_calls += 1
            
        
        start_t = time.time()
        res = self._compute_p_value(data, X, Y, list(cond_set))
        self.total_test_time += (time.time() - start_t)
        return res

    def _compute_p_value(self, data, X, Y, cond_set):
        """
        Implementation specific computation.
        Returns:
            pval: p-value
            dep: dependence score (optional, can be -pval or statistic)
        """
        return 0.0, 0.0

class G2Tester(CITester):
    """
    Wrapper around causal-learn's G-Square test ('gsq').
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = None
        self.cl_cit = None
        if 'data' in kwargs:
            self._set_data(kwargs['data'])
            
    def _set_data(self, data):
        """Initialize causal-learn CIT instance"""
        self.data = data
        # causal-learn CIT expects numpy array
        np_data = data.to_numpy() if isinstance(data, pd.DataFrame) else data
        self.cl_cit = CIT(np_data, "gsq")

    def _compute_p_value(self, data, X, Y, cond_set):
        # Check if data has changed
        if data is not None and data is not self.data:
            # Create a one-off CIT for this new data
            np_data = data.to_numpy() if isinstance(data, pd.DataFrame) else data
            temp_cit = CIT(np_data, "gsq")
            pval = temp_cit(X, Y, cond_set)
        else:
            # Use cached CIT instance
            if self.cl_cit is None:
                raise ValueError("G2Tester: data not initialized and no data passed to ci_test")
            pval = self.cl_cit(X, Y, cond_set)
            
        # Return pval, dep. 
        # For G2, dep could be simply 1-pval or just 0 as placeholder since algorithms mostly use pval.
        # Original code returned abs(G2_statistic) as dep. 
        # causal-learn's generic CIT wrapper returns p-value only.
        # We'll return -pval as a proxy for dependence strength if needed, or 0.
        return pval, -pval

class PartialCorrelation(CITester):
    """
    Wrapper around causal-learn's Fisher-Z test ('fisherz').
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = None
        self.cl_cit = None
        if 'data' in kwargs:
            self._set_data(kwargs['data'])

    def _set_data(self, data):
        self.data = data
        np_data = data.to_numpy() if isinstance(data, pd.DataFrame) else data
        self.cl_cit = CIT(np_data, "fisherz")

    def _compute_p_value(self, data, X, Y, cond_set):
        if data is not None and data is not self.data:
            # THIS IS NOT ALLOWED in rebuttal experiments to ensure fairness
            raise ValueError(f"FAIRNESS VIOLATION: Heavy branch hit! Data identity mismatch. data={id(data)}, self.data={id(self.data)}")
            # np_data = data.to_numpy() if isinstance(data, pd.DataFrame) else data
        else:
            if self.cl_cit is None:
                raise ValueError("PartialCorrelation: data not initialized and no data passed to ci_test")
            pval = self.cl_cit(X, Y, cond_set)
            
        return pval, -pval

class KernelCITest(CITester):
    """
    Wrapper around causal-learn's KCI test ('kci').
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = None
        self.cl_cit = None
        if 'data' in kwargs:
            self._set_data(kwargs['data'])

    def _set_data(self, data):
        self.data = data
        np_data = data.to_numpy() if isinstance(data, pd.DataFrame) else data
        # KCI initialization can be expensive
        self.cl_cit = CIT(np_data, "kci")

    def _compute_p_value(self, data, X, Y, cond_set):
        if data is not None and data is not self.data:
            np_data = data.to_numpy() if isinstance(data, pd.DataFrame) else data
            temp_cit = CIT(np_data, "kci")
            pval = temp_cit(X, Y, cond_set)
        else:
            if self.cl_cit is None:
                raise ValueError("KernelCITest: data not initialized")
            pval = self.cl_cit(X, Y, cond_set)
        return pval, -pval

import networkx as nx

class ChiSquareTester(CITester):
    """
    Wrapper around causal-learn's Chi-Square test ('chisq').
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = None
        self.cl_cit = None
        if 'data' in kwargs:
            self._set_data(kwargs['data'])
            
    def _set_data(self, data):
        """Initialize causal-learn CIT instance"""
        self.data = data
        # causal-learn CIT expects numpy array
        np_data = data.to_numpy() if isinstance(data, pd.DataFrame) else data
        self.cl_cit = CIT(np_data, "chisq")

    def _compute_p_value(self, data, X, Y, cond_set):
        # Check if data has changed
        if data is not None and data is not self.data:
            # Create a one-off CIT for this new data
            np_data = data.to_numpy() if isinstance(data, pd.DataFrame) else data
            temp_cit = CIT(np_data, "chisq")
            pval = temp_cit(X, Y, cond_set)
        else:
            # Use cached CIT instance
            if self.cl_cit is None:
                raise ValueError("ChiSquareTester: data not initialized and no data passed to ci_test")
            pval = self.cl_cit(X, Y, cond_set)
            
        return pval, -pval

class OracleCITester(CITester):
    """
    Oracle CI Tester using d-separation on the True Graph.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'true_graph' not in kwargs:
            raise ValueError("OracleCITester requires 'true_graph' in kwargs.")
        self.true_graph = kwargs['true_graph']
        
    def _compute_p_value(self, data, X, Y, cond_set):
        is_separated = nx.d_separated(self.true_graph, {X}, {Y}, set(cond_set))
        if is_separated:
            return 1.0, 0.0
        else:
            return 0.0, 1.0

def ci_test_factory(name, **kwargs):
    if name == 'G2':
        return G2Tester(**kwargs)
    elif name == 'chisq':
        return ChiSquareTester(**kwargs)
    elif name == 'ParCorr':
        return PartialCorrelation(**kwargs)
    elif name == 'KCI':
        return KernelCITest(**kwargs)
    elif name == 'oracle':
        return OracleCITester(**kwargs)
    else:
        raise AssertionError(f'unknown CI tester: {name}')
