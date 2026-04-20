"""
Evaluation metrics for the Lo-Hi benchmark.

Hi task:
- PR AUC
- ROC AUC
- BEDROC
- F1 at threshold 0.5
- positive rate

Lo task:
- Mean intra-cluster Spearman correlation
- Mean intra-cluster R2
- Mean intra-cluster MAE
- Number of evaluated clusters

Metrics are aggregated across outer folds as mean ± std.
"""

import numpy as np
from typing import Dict, List
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    r2_score,
    mean_absolute_error,
)
from deepchem.metrics import bedroc_score


# ---------------------------------------------------------------------------
# Hi metrics (Hit Identification — binary classification)
# ---------------------------------------------------------------------------

def get_hi_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Dict[str, float]:
    """
    Compute metrics for the Hi (Hit Identification) task.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels (0/1).
    y_pred_proba : array-like of shape (n_samples,)
        Predicted probability (or score) for the positive class.

    Returns
    -------
    dict
        Dictionary containing Hi evaluation metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)

    pr_auc = average_precision_score(y_true, y_pred_proba)

    # try/except to manage cases with no active molecules
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        roc_auc = float("nan")

    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0.0)

    two_class_prob = np.column_stack([1.0 - y_pred_proba, y_pred_proba])
    try:
        bedroc = bedroc_score(y_true, two_class_prob, alpha=70.0)
    except Exception:
        bedroc = float("nan")

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "bedroc": round(bedroc, 4),
        "f1_at_05": round(f1, 4),
        "positive_rate": round(y_true.mean(), 4),
    }


# ---------------------------------------------------------------------------
# Lo metrics (Lead Optimization — ranking within clusters)
# ---------------------------------------------------------------------------

def get_lo_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cluster_ids: np.ndarray,
) -> Dict[str, float]:
    """
    Compute metrics for the Lo (Lead Optimization) task.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth continuous activity values.
    y_pred : array-like of shape (n_samples,)
        Predicted values or scores.
    cluster_ids : array-like of shape (n_samples,)
        Cluster identifier for each sample.

    Returns
    -------
    dict
        Dictionary containing Lo evaluation metrics aggregated across clusters.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    cluster_ids = np.asarray(cluster_ids)

    unique_clusters = np.unique(cluster_ids)

    spearman_scores = []
    r2_scores = []
    mae_scores = []

    for cluster in unique_clusters:
        mask = cluster_ids == cluster

        if mask.sum() < 2:
            continue

        y_cluster = y_true[mask]
        pred_cluster = y_pred[mask]

        rho, _ = spearmanr(y_cluster, pred_cluster)
        if np.isnan(rho):
            rho = 0.0
        spearman_scores.append(rho)

        try:
            r2 = r2_score(y_cluster, pred_cluster)
        except ValueError:
            r2 = float("nan")
        if not np.isnan(r2):
            r2_scores.append(r2)

        mae = mean_absolute_error(y_cluster, pred_cluster)
        mae_scores.append(mae)

    if len(spearman_scores) == 0:
        return {
            "mean_spearman": 0.0,
            "std_spearman": 0.0,
            "mean_r2": 0.0,
            "mean_mae": 0.0,
            "n_clusters": 0,
        }

    return {
        "mean_spearman": round(np.mean(spearman_scores), 4),
        "std_spearman": round(np.std(spearman_scores), 4),
        "mean_r2": round(np.mean(r2_scores), 4) if len(r2_scores) > 0 else 0.0,
        "mean_mae": round(np.mean(mae_scores), 4),
        "n_clusters": len(spearman_scores),
    }


# ---------------------------------------------------------------------------
# Aggregation across outer folds
# ---------------------------------------------------------------------------

def aggregate_fold_metrics(
    fold_metrics: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregate metrics across outer folds.

    Parameters
    ----------
    fold_metrics : list of dict
        One metrics dictionary per outer fold.

    Returns
    -------
    dict
        Mean and standard deviation for each numeric metric.
    """
    all_keys = fold_metrics[0].keys()
    aggregated = {}

    for key in all_keys:
        values = [
            metrics[key]
            for metrics in fold_metrics
            if isinstance(metrics[key], (int, float, np.floating))
        ]

        if len(values) > 0:
            aggregated[f"{key}_mean"] = round(np.mean(values), 4)
            aggregated[f"{key}_std"] = round(np.std(values), 4)

    return aggregated