"""
I/O utilities for the Lo-Hi project.

Handles:
- Loading train/test CSV folds from data/
- Managing feature cache paths in features/
- Saving predictions to results/
- Saving best hyperparameters as JSON
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """
    Find the project root.
    """
    current = Path(__file__).resolve().parent  # utils/
    for parent in [current] + list(current.parents):
        if (parent / "README.md").exists() or (parent / "requirements.txt").exists():
            return parent
    return current.parent


PROJECT_ROOT = get_project_root()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_fold(
    task: str,
    dataset: str,
    fold_idx: int,
    data_dir: Optional[str] = None,
) -> tuple:
    """
    Load a train/test fold.

    Parameters
    ----------
    task : str
        "hi" or "lo"
    dataset : str
        es. "drd2", "hiv", "kdr", "sol", "kcnh2"
    fold_idx : int
        Fold number (1, 2, or 3)
    data_dir : str, optional
        Override default data/ directory.

    Returns
    -------
    (train_df, test_df) : tuple of DataFrames
        Each has columns: smiles, value, [cluster for Lo task]
    """
    if data_dir is None:
        data_dir = PROJECT_ROOT / "data"

    base = Path(data_dir) / task / dataset
    train_path = base / f"train_{fold_idx}.csv"
    test_path = base / f"test_{fold_idx}.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train = pd.read_csv(train_path, index_col=0)
    test = pd.read_csv(test_path, index_col=0)

    logger.info(
        f"Loaded {task}/{dataset} fold {fold_idx}: "
        f"train={len(train)}, test={len(test)}"
    )
    return train, test


# ---------------------------------------------------------------------------
# Feature cache paths
# ---------------------------------------------------------------------------

def get_feature_cache_path(
    task: str, dataset: str, fp_type: str, split: str, fold_idx: int,
) -> str:
    """
    Return the cache path for precomputed fingerprints.
    
    Example: features/hi/drd2/ecfp4_train_1.npz
    """
    features_dir = PROJECT_ROOT / "features" / task / dataset
    return str(features_dir / f"{fp_type}_{split}_{fold_idx}.npz")


# ---------------------------------------------------------------------------
# Results directory
# ---------------------------------------------------------------------------

def get_results_dir(
    task: str, dataset: str, model_name: str, fp_type: str,
) -> Path:
    """
    Return (and create) the results directory for a specific experiment.
    
    Example: results/hi/drd2/knn_ecfp4/
    """
    results_dir = PROJECT_ROOT / "results" / task / dataset / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


# ---------------------------------------------------------------------------
# Save predictions
# ---------------------------------------------------------------------------

def save_predictions(
    df: pd.DataFrame,
    preds: np.ndarray,
    task: str,
    dataset: str,
    model_name: str,
    fp_type: str,
    split: str,
    fold_idx: int,
) -> str:
    """
    Save predictions alongside original data.

    Creates a CSV with all original columns + 'preds' column.
    
    Returns the path of the saved file.
    """
    results_dir = get_results_dir(task, dataset, model_name, fp_type)
    out_df = df.copy()
    out_df["preds"] = preds
    
    out_path = results_dir / f"{split}_{fold_idx}.csv"
    out_df.to_csv(out_path)
    logger.info(f"Saved predictions to {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# Save hyperparameters
# ---------------------------------------------------------------------------

def save_params(
    params: dict,
    task: str,
    dataset: str,
    model_name: str,
    fp_type: str,
    fold_idx: int,
    extra_info: Optional[dict] = None,
) -> str:
    """
    Save best hyperparameters for a fold as JSON.

    Parameters
    ----------
    params : dict
        Best hyperparameters from inner CV.
    extra_info : dict, optional
        Additional info (inner_cv_score, training_time, etc.)

    Returns the path of the saved file.
    """
    results_dir = get_results_dir(task, dataset, model_name, fp_type)

    record = {
        "fold": fold_idx,
        "best_params": params,
    }
    if extra_info:
        record.update(extra_info)

    out_path = results_dir / f"params_fold_{fold_idx}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    logger.info(f"Saved params to {out_path}")
    return str(out_path)


def load_all_params(
    task: str, dataset: str, model_name: str, fp_type: str,
) -> list:
    """Load all fold params JSONs for a given experiment."""
    results_dir = get_results_dir(task, dataset, model_name, fp_type)
    params = []
    for fold_idx in [1, 2, 3]:
        path = results_dir / f"params_fold_{fold_idx}.json"
        if path.exists():
            with open(path) as f:
                params.append(json.load(f))
    return params