"""
Nested cross-validation pipeline for the Lo-Hi benchmark.

Implements the proper methodology:
- Outer loop: 3 pre-defined folds (from Steshin's splitting)
- Inner loop: k-fold stratified CV on train_i for hyperparameter selection
- Retrain on full train_i with best params
- Single evaluation on test_i

For neural networks with early stopping:
- Inner CV determines both best hyperparams AND avg best epoch
- Retrain on full train_i for avg_best_epoch epochs (no val leakage)
"""

import time
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.base import BaseEstimator

from utils.fingerprints import compute_fingerprints
from utils.metrics import get_hi_metrics, get_lo_metrics, aggregate_fold_metrics
from utils.io_utils import (
    load_fold,
    save_predictions,
    save_params,
    get_feature_cache_path,
)

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inner CV: hyperparameter search on a single outer fold
# ---------------------------------------------------------------------------

def _inner_cv_sklearn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    estimator: BaseEstimator,
    param_grid: dict,
    inner_k: int = 3,
    scoring: str = "average_precision",
    search_strategy: str = "grid",
    n_iter: int = 50,
    random_state: int = 42,
) -> Tuple[BaseEstimator, dict, float]:
    """
    Run inner cross-validation to select hyperparameters.

    Returns
    -------
    (best_estimator, best_params, best_inner_score)
        best_estimator is already refit on the full X_train.
    """
    inner_cv = StratifiedKFold(
        n_splits=inner_k, shuffle=True, random_state=random_state
    )

    if search_strategy == "random":
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=inner_cv,
            scoring=scoring,
            refit=True,
            n_jobs=-1,
            random_state=random_state,
            error_score="raise",
        )
    else:
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scoring,
            refit=True,
            n_jobs=-1,
            error_score="raise",
        )

    search.fit(X_train, y_train)

    logger.info(f"  Inner CV best score: {search.best_score_:.4f}")
    logger.info(f"  Inner CV best params: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.best_score_


# ---------------------------------------------------------------------------
# Single outer fold execution
# ---------------------------------------------------------------------------

def run_single_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_idx: int,
    task: str,
    dataset: str,
    fp_type: str,
    model_name: str,
    estimator_factory: Callable[[], BaseEstimator],
    param_grid: dict,
    inner_k: int = 3,
    scoring: str = "average_precision",
    search_strategy: str = "grid",
    n_iter: int = 50,
    random_state: int = 42,
    save_results: bool = True,
) -> Dict[str, Any]:
    """
    Execute one outer fold: featurize → inner CV → retrain → evaluate → save.

    Parameters
    ----------
    train_df, test_df : DataFrame
        Must have 'smiles' and 'value' columns.
    fold_idx : int
        Which outer fold (1, 2, or 3).
    task : str
        "hi" or "lo".
    fp_type : str
        Fingerprint type key (e.g. "ecfp4").
    model_name : str
        Short model name (e.g. "knn", "svm", "gb").
    estimator_factory : callable
        Returns a fresh (unfitted) sklearn estimator.
    param_grid : dict
        Hyperparameter search space.
    save_results : bool
        Whether to save predictions and params to disk.

    Returns
    -------
    dict with: best_params, inner_cv_score, test_metrics, train_metrics, time_seconds
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"FOLD {fold_idx} | {model_name} + {fp_type} | {task}/{dataset}")
    logger.info(f"{'='*60}")

    t0 = time.time()

    # --- 1. Featurize ---
    train_cache = get_feature_cache_path(task, dataset, fp_type, "train", fold_idx)
    test_cache = get_feature_cache_path(task, dataset, fp_type, "test", fold_idx)

    X_train = compute_fingerprints(train_df["smiles"].tolist(), fp_type, train_cache)
    X_test = compute_fingerprints(test_df["smiles"].tolist(), fp_type, test_cache)
    y_train = train_df["value"].values
    y_test = test_df["value"].values

    # --- 2. Inner CV (hyperparameter search on train only) ---
    estimator = estimator_factory()
    best_model, best_params, inner_score = _inner_cv_sklearn(
        X_train, y_train, estimator, param_grid,
        inner_k=inner_k, scoring=scoring,
        search_strategy=search_strategy, n_iter=n_iter,
        random_state=random_state,
    )

    # --- 3. Predict (best_model is already refit on full train) ---
    if hasattr(best_model, "predict_proba"):
        train_preds = best_model.predict_proba(X_train)[:, 1]
        test_preds = best_model.predict_proba(X_test)[:, 1]
    elif hasattr(best_model, "decision_function"):
        train_preds = best_model.decision_function(X_train)
        test_preds = best_model.decision_function(X_test)
    else:
        train_preds = best_model.predict(X_train)
        test_preds = best_model.predict(X_test)

    # --- 4. Evaluate ---
    if task == "hi":
        train_metrics = get_hi_metrics(y_train, train_preds)
        test_metrics = get_hi_metrics(y_test, test_preds)
    else:
        cluster_train = train_df.get("cluster", np.zeros(len(train_df))).values
        cluster_test = test_df["cluster"].values
        train_metrics = get_lo_metrics(y_train, train_preds, cluster_train)
        test_metrics = get_lo_metrics(y_test, test_preds, cluster_test)

    elapsed = time.time() - t0

    logger.info(f"  Train metrics: {train_metrics}")
    logger.info(f"  Test metrics:  {test_metrics}")
    logger.info(f"  Time: {elapsed:.1f}s")

    # --- 5. Save ---
    if save_results:
        save_predictions(train_df, train_preds, task, dataset, model_name, fp_type, "train", fold_idx)
        save_predictions(test_df, test_preds, task, dataset, model_name, fp_type, "test", fold_idx)
        save_params(best_params, task, dataset, model_name, fp_type, fold_idx,
                    extra_info={
                        "inner_cv_score": inner_score,
                        "train_metrics": train_metrics,
                        "test_metrics": test_metrics,
                        "time_seconds": round(elapsed, 1),
                    })

    return {
        "fold": fold_idx,
        "best_params": best_params,
        "inner_cv_score": inner_score,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "time_seconds": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Full nested CV (all 3 outer folds)
# ---------------------------------------------------------------------------

def run_nested_cv(
    task: str,
    dataset: str,
    fp_type: str,
    model_name: str,
    estimator_factory: Callable[[], BaseEstimator],
    param_grid: dict,
    inner_k: int = 3,
    scoring: str = "average_precision",
    search_strategy: str = "grid",
    n_iter: int = 50,
    random_state: int = 42,
    folds: List[int] = [1, 2, 3],
    save_results: bool = True,
) -> Dict[str, Any]:
    """
    Run the full nested cross-validation across all outer folds.

    This is the main entry point for running an experiment.

    Returns
    -------
    dict with:
        fold_results    - list of per-fold result dicts
        aggregated      - mean ± std across folds
        experiment_id   - string identifier
    """
    logger.info(f"\n{'#'*60}")
    logger.info(f"EXPERIMENT: {model_name} + {fp_type} on {task}/{dataset}")
    logger.info(f"{'#'*60}")

    fold_results = []

    for fold_idx in folds:
        train_df, test_df = load_fold(task, dataset, fold_idx)

        result = run_single_fold(
            train_df, test_df, fold_idx,
            task=task, dataset=dataset,
            fp_type=fp_type, model_name=model_name,
            estimator_factory=estimator_factory,
            param_grid=param_grid,
            inner_k=inner_k, scoring=scoring,
            search_strategy=search_strategy, n_iter=n_iter,
            random_state=random_state,
            save_results=save_results,
        )
        fold_results.append(result)

    # Aggregate test metrics across folds
    test_metrics_list = [r["test_metrics"] for r in fold_results]
    aggregated = aggregate_fold_metrics(test_metrics_list)

    logger.info(f"\n{'='*60}")
    logger.info(f"AGGREGATED TEST METRICS:")
    for k, v in aggregated.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"{'='*60}")

    # Check hyperparameter stability
    all_params = [r["best_params"] for r in fold_results]
    if len(set(str(p) for p in all_params)) > 1:
        logger.info("NOTE: Best hyperparameters differ across folds (expected in proper nested CV)")
        for r in fold_results:
            logger.info(f"  Fold {r['fold']}: {r['best_params']}")
    else:
        logger.info(f"Best hyperparameters consistent across folds: {all_params[0]}")

    experiment_id = f"{model_name}_{fp_type}_{task}_{dataset}"

    return {
        "experiment_id": experiment_id,
        "fold_results": fold_results,
        "aggregated": aggregated,
    }