from scipy.spatial import cKDTree
import numpy as np


EPS = 1e-12


def compute_metrics(
        pred: np.ndarray, true: np.ndarray,
        cutoff: float = 3.0,
        ):
    """ Compute metrics for predicted and true coordinates.

    Args:
        pred: predicted coordinates, 2d array (n, 2)
        true: true coordinates, 2d array (n, 2)
        cutoff: cutoff distance for matching
    """
    if pred.shape[0] > 0:
        if true.shape[0] > 0:
            tree = cKDTree(pred)
            dist, pred_idx = tree.query(true, k=1)
            tp_idx = (dist <= cutoff)
            pred_true_idx = np.unique(pred_idx[tp_idx])
            tp = len(pred_true_idx)
            fn = len(true) - tp
            fp = len(pred) - tp
            mean_dist = dist[tp_idx].mean()
        else:
            tp = 0
            fn = 0
            fp = len(pred)
            mean_dist = np.nan
    else:
        tp = 0
        fn = len(true)
        fp = 0
        mean_dist = np.nan
    recall = tp / (tp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    f1 = 2 * (precision * recall) / (precision + recall + EPS)
    res = {
        'true_positive': tp,
        'false_negative': fn,
        'false_positive': fp,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'mean_dist': mean_dist,
    }
    return res
