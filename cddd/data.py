"""
Graph and data generation for Bayesian network experiments.

- Graph construction: random_DAG, erdos_renyi_DAG, scalefree_DAG.
- BayesianNetwork: one abstraction over (graph + model). The model can be
  discrete (CPTs), linear SEM, nonlinear SEM, or an external sampler.
  Implements .sample(n) and .sample_batch(n_per_run, num_runs) where applicable.
- DataGenerator: wraps a BayesianNetwork and exposes .generate() / .generate_batch();
  factory methods build the BN and return a DataGenerator.
- I/O: get_adj_mat, save_adj_mat.
"""
from __future__ import annotations

import random
from collections import deque
from functools import lru_cache
from itertools import product, combinations
from typing import AbstractSet, Callable, Dict, Literal, Optional, Sequence, Tuple, TypeVar, Set, Collection, List, Hashable, Generic, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from numpy.random import randint, choice, rand
from .utils import get_adj_mat, save_adj_mat

CPTType = Dict[Tuple[int, ...], Sequence[float]]
T = TypeVar("T")
H = TypeVar("H", bound=Hashable)
BNModelKind = Literal["discrete", "linear_sem", "nonlinear_sem", "external"]


class Graph:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self._order = list(nx.topological_sort(graph))

    @property
    def n_variables(self):
        return self.graph.number_of_nodes()

    @staticmethod
    def random_DAG(num_vars: int, num_edges: int) -> nx.DiGraph:
        order = list(range(num_vars))
        random.shuffle(order)
        edges = list(combinations(order, 2))
        random.shuffle(edges)
        edges = edges[:num_edges]
        G = nx.DiGraph(edges)
        G.add_nodes_from(range(num_vars))
        return G

    @staticmethod
    def erdos_renyi_DAG(num_vars: int, num_edges: int, max_in_degree: Optional[int] = None) -> nx.DiGraph:
        G = nx.gnm_random_graph(num_vars, num_edges)
        adj = np.tril(nx.to_numpy_array(G), -1)
        if max_in_degree is not None:
             for i in range(num_vars):
                 parents = np.where(adj[:, i] == 1)[0]
                 if len(parents) > max_in_degree:
                     # Prune random parents
                     to_remove = np.random.choice(parents, len(parents) - max_in_degree, replace=False)
                     adj[to_remove, i] = 0
        return nx.DiGraph(adj)

    @staticmethod
    def scale_free_DAG(num_vars: int, m: int, max_in_degree: Optional[int] = None) -> nx.DiGraph:
        G = nx.barabasi_albert_graph(num_vars, m)
        adj = np.tril(nx.to_numpy_array(G), -1)
        if max_in_degree is not None:
            for i in range(num_vars):
                parents = np.where(adj[:, i] == 1)[0]
                if len(parents) > max_in_degree:
                    to_remove = np.random.choice(parents, len(parents) - max_in_degree, replace=False)
                    adj[to_remove, i] = 0
        return nx.DiGraph(adj)


class BayesianNetwork(Graph):
    def __init__(
        self,
        graph: nx.DiGraph,
        *,
        model_type: BNModelKind = "discrete",
        cpts: Optional[Dict[int, CPTType]] = None,
        sample_fn: Optional[Callable[[int], pd.DataFrame]] = None,
        linear_w_ranges: Tuple[Tuple[float, float], ...] = ((-2.0, -0.5), (0.5, 2.0)),
        nonlinear_hidden: int = 100,
        cardinality: int = 3,
        eps: float = 0.4,
    ):
        super().__init__(graph)
        self._model_type = model_type
        self.cardinality = cardinality
        self.eps = eps
        self.CPTs = {}
        self.W = None
        self.nonlinear_params = {}
        self._sample_fn = None

        if model_type == "discrete":
            self.CPTs = cpts if cpts is not None else self._init_discrete_params(graph, cardinality, eps)
        elif model_type == "linear_sem":
            self.W = self._init_linear_params(graph, linear_w_ranges)
        elif model_type == "nonlinear_sem":
            self.nonlinear_hidden = nonlinear_hidden
            self.nonlinear_params = self._init_nonlinear_params(graph, nonlinear_hidden)
        elif model_type == "external":
            self._sample_fn = sample_fn

    @staticmethod
    def _get_prob_with_floor(card: int, eps: float) -> np.ndarray:
        u = np.ones(card) / card
        r = rand(card)
        r = r / r.sum()
        p = (1 - eps) * u + eps * r
        return p / p.sum()

    @staticmethod
    def _init_discrete_params(G: nx.DiGraph, card: int, eps: float) -> Dict[int, CPTType]:
        cpts = {}
        for v in G.nodes:
            parents = tuple(sorted(G.predecessors(v)))
            cpt = {}
            for pa_vals in product(range(card), repeat=len(parents)):
                cpt[pa_vals] = BayesianNetwork._get_prob_with_floor(card, eps)
            cpts[v] = cpt
        return cpts

    @staticmethod
    def _init_linear_params(G: nx.DiGraph, ranges: Tuple[Tuple[float, float], ...]) -> np.ndarray:
        n = G.number_of_nodes()
        B = nx.to_numpy_array(G, nodelist=range(n))
        W = np.zeros(B.shape)
        S = np.random.randint(len(ranges), size=B.shape)
        for i, (lo, hi) in enumerate(ranges):
            U = np.random.uniform(lo, hi, size=B.shape)
            W += B * (S == i) * U
        return W

    @staticmethod
    def _init_nonlinear_params(G: nx.DiGraph, hidden: int) -> Dict[int, Optional[Tuple[np.ndarray, np.ndarray]]]:
        params = {}
        for v in G.nodes:
            pa_size = len(list(G.predecessors(v)))
            if pa_size > 0:
                W1 = np.random.uniform(0.5, 1.5, (pa_size, hidden))
                W1[np.random.rand(*W1.shape) < 0.5] *= -1
                W2 = np.random.uniform(0.5, 1.5, hidden)
                W2[np.random.rand(hidden) < 0.5] *= -1
                params[v] = (W1, W2)
            else:
                params[v] = None
        return params


class DataGenerator:
    def __init__(self, bn: BayesianNetwork):
        self.bn = bn

    def sample(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if self.bn._model_type == "discrete":
            return self._sample_discrete(n)
        if self.bn._model_type == "linear_sem":
            return self._sample_linear_sem(n)
        if self.bn._model_type == "nonlinear_sem":
            return self._sample_nonlinear_sem(n)
        if self.bn._model_type == "external" and self.bn._sample_fn:
            return self.bn._sample_fn(n)
        raise ValueError(f"Unknown model type: {self.bn._model_type}")

    def _sample_discrete(self, n: int) -> pd.DataFrame:
        out = np.zeros((n, self.bn.n_variables))
        for i in range(n):
            row = out[i]
            for v in self.bn._order:
                pa_vals = tuple(int(row[_]) for _ in sorted(self.bn.graph.predecessors(v)))
                probs = self.bn.CPTs[v][pa_vals]
                row[v] = choice(self.bn.cardinality, 1, p=probs)[0]
        return pd.DataFrame(out, columns=range(self.bn.n_variables))

    def _sample_linear_sem(self, n: int) -> pd.DataFrame:
        W = self.bn.W
        X = np.zeros((n, self.bn.n_variables))
        scale = np.ones(self.bn.n_variables)
        for j in self.bn._order:
            pa = tuple(self.bn.graph.predecessors(j))
            X[:, j] = X[:, pa] @ W[pa, j] + np.random.normal(scale=scale[j], size=n)
        return pd.DataFrame(X.astype(np.float32), columns=range(self.bn.n_variables))

    def _sample_nonlinear_sem(self, n: int) -> pd.DataFrame:
        from scipy.special import expit as sigmoid
        scale = np.ones(self.bn.n_variables)
        X = np.zeros((n, self.bn.n_variables))
        for j in self.bn._order:
            pa = tuple(sorted(self.bn.graph.predecessors(j)))
            z = np.random.normal(scale=scale[j], size=n)
            params = self.bn.nonlinear_params.get(j)
            if params is None:
                X[:, j] = z
            else:
                W1, W2 = params
                X[:, j] = sigmoid(X[:, pa] @ W1) @ W2 + z
        return pd.DataFrame(X.astype(np.float32), columns=range(self.bn.n_variables))
