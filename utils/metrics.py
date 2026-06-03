"""
Evaluation metrics for the Lo-Hi benchmark.

Hi task:
- PR AUC
- ROC AUC
- BEDROC (alpha=70.0, corresponding to strong early-enrichment focus on ~top 8% of ranked list)
- F1 at threshold 0.5 (only if scores are probabilities in [0, 1])
- positive rate

Lo task:
- Mean intra-cluster Spearman correlation
- Mean intra-cluster R2
- Mean intra-cluster MAE
- Number of evaluated clusters (clusters with < 3 members are skipped)

Metrics are aggregated across outer folds as mean ± std.
NaN values from individual folds are excluded from aggregation with a warning.
"""

import logging
import warnings

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
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from scipy.stats import ConstantInputWarning

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hi metrics (Hit Identification — binary classification)
# ---------------------------------------------------------------------------

def get_hi_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    """
    Compute metrics for the Hi (Hit Identification) task.

    Parameters
    ----------
    y_true : Ground truth binary labels (0/1).
    y_score : Predicted score for the positive class.
              If all values are in [0, 1], F1 at threshold 0.5 is also computed.

    Notes
    -----
    BEDROC alpha=70.0 corresponds to a strong early-enrichment focus on
    approximately the top 8% of the ranked list. This is a standard choice
    for virtual screening benchmarks but may be unstable on small test sets
    (< 200 molecules). Ties in y_score are broken with a tiny uniform noise
    (magnitude 1e-9) seeded at 42; this is sufficient for probability scores
    in [0, 1] but may be effectively zero for large-magnitude decision function
    scores (e.g., SVM). In that case ties are not broken.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    pr_auc = average_precision_score(y_true, y_score)

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = float("nan")

    # BEDROC requires a sorted 2D array [score, label]
    try:
        scores = np.column_stack([y_score, y_true])

        # Break ties with tiny uniform noise (see Notes above).
        rng = np.random.default_rng(seed=42)
        tiebreak = rng.uniform(0, 1e-9, size=len(y_score))

        order = np.argsort(-(y_score + tiebreak))
        scores_sorted = scores[order]

        bedroc = CalcBEDROC(scores_sorted, col=1, alpha=70.0)
    except Exception:
        bedroc = float("nan")

    if np.all((y_score >= 0.0) & (y_score <= 1.0)):
        y_pred_binary = (y_score >= 0.5).astype(int)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0.0)
        f1_rounded: float = round(float(f1), 4)
    else:
        f1_rounded = float("nan")

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4) if not np.isnan(roc_auc) else float("nan"),
        "bedroc": round(bedroc, 4) if not np.isnan(bedroc) else float("nan"),
        "f1_at_05": f1_rounded,
        "positive_rate": round(float(y_true.mean()), 4),
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
    y_true : Ground truth continuous activity values.
    y_pred : Predicted values or scores.
    cluster_ids : Cluster identifier for each sample.

    Notes
    -----
    Clusters with fewer than 3 members are skipped. If no cluster has >= 3
    members, all metrics are returned as NaN (not 0.0) so that
    aggregate_fold_metrics can correctly exclude this fold from the mean.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    cluster_ids = np.asarray(cluster_ids)

    unique_clusters = np.unique(cluster_ids)

    spearman_scores = []
    r2_scores = []
    mae_scores = []
    n_skipped = 0

    for cluster in unique_clusters:
        mask = cluster_ids == cluster

        if mask.sum() < 3:
            n_skipped += 1
            continue

        y_cluster = y_true[mask]
        pred_cluster = y_pred[mask]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
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

    if n_skipped > 0:
        logger.warning(
            f"get_lo_metrics: skipped {n_skipped} cluster(s) with < 3 members "
            f"(out of {len(unique_clusters)} total clusters)."
        )

    if len(spearman_scores) == 0:
        logger.warning(
            "get_lo_metrics: no cluster had >= 3 members; returning NaN for all metrics. "
            "This fold will be excluded from aggregation."
        )
        return {
            "mean_spearman": float("nan"),
            "std_spearman": float("nan"),
            "mean_r2": float("nan"),
            "mean_mae": float("nan"),
            "n_clusters": 0,
        }

    return {
        "mean_spearman": round(float(np.mean(spearman_scores)), 4),
        "std_spearman": round(float(np.std(spearman_scores)), 4),
        "mean_r2": round(float(np.mean(r2_scores)), 4) if len(r2_scores) > 0 else float("nan"),
        "mean_mae": round(float(np.mean(mae_scores)), 4),
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
    fold_metrics : One metrics dictionary per outer fold.

    Notes
    -----
    NaN values are excluded from aggregation. A warning is logged whenever
    a fold is dropped for a given metric, including how many folds contributed.
    This is important for paper reporting: if a metric is averaged over fewer
    than all folds, this must be disclosed.
    """
    all_keys = fold_metrics[0].keys()
    aggregated = {}
    n_total_folds = len(fold_metrics)

    for key in all_keys:
        values = []
        for metrics in fold_metrics:
            v = metrics[key]
            if isinstance(v, (int, float, np.floating)) and not np.isnan(v):
                values.append(float(v))

        n_valid = len(values)

        if n_valid < n_total_folds:
            logger.warning(
                f"aggregate_fold_metrics: metric '{key}' has only {n_valid}/{n_total_folds} "
                f"valid (non-NaN) folds. Mean and std are computed over {n_valid} fold(s) only."
            )

        if n_valid > 0:
            aggregated[f"{key}_mean"] = round(np.mean(values), 4)
            aggregated[f"{key}_std"] = round(np.std(values), 4)
            aggregated[f"{key}_n_valid_folds"] = n_valid

    return aggregated
