import os
from typing import List, Optional
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from filelock import FileLock



def safe_save_to_csv(result: List[List], columns: List[str], file_path: str):
    df = pd.DataFrame(result, columns=columns)
    lock = FileLock(file_path + '.lock')
    with lock:
        if not os.path.exists(file_path):
            df.to_csv(file_path, mode='w', index=False)
        else:
            df.to_csv(file_path, mode='a', header=False, index=False)





def get_adj_mat(num_vars: int, path: str) -> pd.DataFrame:
    """Read adjacency matrix from file (space-separated 0/1, no header)."""
    with open(path) as f:
        lines = f.readlines()
    mat = [[int(x) for x in line.strip().split()] for line in lines if line.strip()]
    return pd.DataFrame(mat, index=range(num_vars), columns=range(num_vars))


def save_adj_mat(G: nx.DiGraph, path: str, num_vars: Optional[int] = None) -> None:
    """Write adjacency matrix to file (space-separated, no header)."""
    n = num_vars if num_vars is not None else G.number_of_nodes()
    adj = nx.to_numpy_array(G, nodelist=range(n)) if n else nx.to_numpy_array(G)
    pd.DataFrame(adj.astype(np.int32), columns=range(adj.shape[1])).to_csv(
        path, sep=" ", header=False, index=False
    )
