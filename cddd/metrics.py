import numpy as np
import pandas as pd


class Evaluator:
    """
    Evaluator for graphical structure learning algorithms.
    """
    def __init__(self, true_adj_mat):
        self.true_adj_mat = true_adj_mat
        self.num_vars = len(true_adj_mat.columns) if isinstance(true_adj_mat, pd.DataFrame) else len(true_adj_mat[0])
        # Ensure numpy array access
        if isinstance(true_adj_mat, pd.DataFrame):
            self.true_adj_mat = true_adj_mat.values

    def global_skeleton_metric_evaluation(self, estim_adj_mat):
        """
        Evaluate global skeleton metrics (precision, recall, f1).
        """
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        # Iterate upper triangle
        for var1 in range(self.num_vars):
            for var2 in range(var1 + 1, self.num_vars):
                # Check adjacency (undirected)
                truth = (self.true_adj_mat[var1][var2] != 0 or self.true_adj_mat[var2][var1] != 0)
                estim = (estim_adj_mat[var1][var2] != 0 or estim_adj_mat[var2][var1] != 0)
                
                if truth and estim:
                    TP += 1
                elif truth and not estim:
                    FN += 1
                elif not truth and estim:
                    FP += 1
                elif not truth and not estim:
                    TN += 1
        

        precision = (TP / (TP + FP)) if TP + FP > 0 else 0
        recall = (TP / (TP + FN)) if TP + FN > 0 else 0
        f1 = ((2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0)

        # Accuracy is not returned as requested
        return precision, recall, f1

    def local_skeleton_metric_evaluation(self, target_index, estimated_pc_set):
        """
        Evaluate local skeleton metrics (precision, recall, f1) for a specific target node.
        estimated_pc_set: list or set of indices adjacent to target in estimated graph.
        """
        # True PC set (neighbors in skeleton)
        # Assuming true_adj_mat is symmetric or we check both directions
        true_neighbors = set()
        for i in range(self.num_vars):
            if i == target_index:
                continue
            if self.true_adj_mat[target_index][i] != 0 or self.true_adj_mat[i][target_index] != 0:
                true_neighbors.add(i)
        
        estimated_pc_set = set(estimated_pc_set)
        
        tp = len(true_neighbors.intersection(estimated_pc_set))
        fp = len(estimated_pc_set - true_neighbors)
        fn = len(true_neighbors - estimated_pc_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Distance metric from original code? 
        # distance = ((1 - precision) ** 2 + (1 - recall) ** 2) ** 0.5
        # We can return it if needed, but F1/Prec/Rec is standard.
        return precision, recall, f1

    def get_SHD(self, oracle_adj_mat, estim_adj_mat):
        """
        Calculate Structural Hamming Distance (SHD).
        """
        # Ensure inputs are comparable types
        oracle = oracle_adj_mat
        estim = estim_adj_mat
        
        diff = np.abs(oracle - estim)
        
        diff = diff + diff.transpose()
        diff[diff > 1] = 1
        return np.sum(diff) / 2

    def get_skeleton_SHD(self, oracle_adj_mat, estim_adj_mat):
        """
        Calculate Skeleton SHD (ignoring edge orientation).
        """
        # Symmetrize both to get skeleton
        # Treat any non-zero as edge existence
        oracle_skel = (oracle_adj_mat + oracle_adj_mat.T > 0).astype(int)
        estim_skel = (estim_adj_mat + estim_adj_mat.T > 0).astype(int)
        
        # Calculate diff on upper triangle only to avoid double counting?
        # Or calculate full diff and divide by 2.
        # Since they are symmetric, full diff is symmetric.
        
        diff = np.abs(oracle_skel - estim_skel)
        # diff matrix will be symmetric. 
        # Sum of diff counts each mismatch twice.
        return np.sum(diff) / 2


