"""
Nested cross-validation pipeline for the Lo-Hi benchmark.

Implements the proper methodology:
- Outer loop: 3 pre-defined folds (from Steshin's splitting)
- Inner model selection strategy:
  1. kfold: k-fold CV on train_i for hyperparameter selection
     (StratifiedKFold for Hi, KFold for Lo)
  2. holdout: OOD-aware fixed inner holdout for Hi only
  3. random_shuffle: random fixed inner holdout
- Refit selected model on the available inner-selection data via GridSearchCV/RandomizedSearchCV
- Single evaluation on test_i

"""

import time
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    KFold,
    train_test_split,
    PredefinedSplit,
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
    task: str,
    inner_k: int = 2,
    scoring: str = "average_precision",
    search_strategy: str = "grid",
    n_iter: int = 50,
    random_state: int = 42,
) -> Tuple[BaseEstimator, dict, float, Optional[float]]:

    if task == "hi":
        inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=random_state)
    else:
        inner_cv = KFold(n_splits=inner_k, shuffle=True, random_state=random_state)
        scoring = "neg_mean_absolute_error"

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
            return_train_score=True,
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
            return_train_score=True,
        )

    search.fit(X_train, y_train)

    best_train_score = None
    if "mean_train_score" in search.cv_results_:
        best_train_score = float(search.cv_results_["mean_train_score"][search.best_index_])

    logger.info(f"  Inner CV best score: {search.best_score_:.4f}")
    logger.info(f"  Inner CV best params: {search.best_params_}")
    if best_train_score is not None:
        logger.info(f"  Inner CV best train score: {best_train_score:.4f}")

    return search.best_estimator_, search.best_params_, search.best_score_, best_train_score


# ---------------------------------------------------------------------------
# Inner holdout
# ---------------------------------------------------------------------------

def _inner_holdout_sklearn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    estimator: BaseEstimator,
    param_grid: dict,
    task: str,
    scoring: str = "average_precision",
    search_strategy: str = "grid",
    n_iter: int = 50,
    random_state: int = 42,
) -> Tuple[BaseEstimator, dict, float, Optional[float]]:
    """
    Hyperparameter search with a fixed train/validation holdout split
    (no k-fold). Used for both 'holdout' (OOD inner split) and
    'random_shuffle' (in-distribution inner split).

    The validation part is used only for model selection inside the
    PredefinedSplit. With refit=True, GridSearchCV/RandomizedSearchCV refits
    the selected estimator on X_train + X_val, which corresponds to the full
    outer training set for the current fold in the OOD holdout protocol.
    """

    # Stack train + val together
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])

    # PredefinedSplit: -1 = always in train, 0 = validation fold for split 0
    test_fold = np.concatenate([
        -np.ones(len(X_train), dtype=int),
        np.zeros(len(X_val), dtype=int),
    ])
    inner_cv = PredefinedSplit(test_fold)

    if task == "hi":
        pass
    else:
        scoring = "neg_mean_absolute_error"

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
            return_train_score=True,
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
            return_train_score=True,
        )

    search.fit(X_all, y_all)

    best_train_score = None
    if "mean_train_score" in search.cv_results_:
        best_train_score = float(search.cv_results_["mean_train_score"][search.best_index_])

    logger.info(f"  Inner holdout best score: {search.best_score_:.4f}")
    logger.info(f"  Inner holdout best params: {search.best_params_}")
    if best_train_score is not None:
        logger.info(f"  Inner holdout best train score: {best_train_score:.4f}")

    return search.best_estimator_, search.best_params_, search.best_score_, best_train_score


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
    inner_k: int = 2,
    scoring: str = "average_precision",
    search_strategy: str = "grid",
    n_iter: int = 50,
    random_state: int = 42,
    save_results: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute one outer fold: featurize → inner model selection → evaluate → save.

    Returns: dict with:
        best_params,
        inner_selection_score,
        inner_train_score,
        test_metrics,
        train_metrics,
        time_seconds
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"FOLD {fold_idx} | {model_name} + {fp_type} | {task}/{dataset}")
    logger.info(f"{'='*60}")

    t0 = time.time()

    # Featurize
    train_cache = get_feature_cache_path(task, dataset, fp_type, "train", fold_idx)
    test_cache = get_feature_cache_path(task, dataset, fp_type, "test", fold_idx)

    X_train = compute_fingerprints(train_df["smiles"].tolist(), fp_type, train_cache)
    X_test = compute_fingerprints(test_df["smiles"].tolist(), fp_type, test_cache)

    # Cast to bool only for KNN/Jaccard distance
    if model_name.startswith("knn") and fp_type in ["ecfp4", "maccs", "rdkit_topo"]:
        X_train = X_train.astype(bool)
        X_test = X_test.astype(bool)

    y_train = train_df["value"].values
    y_test = test_df["value"].values

    # Inner hyperparameter search
    estimator = estimator_factory()
    inner_split_strategy = kwargs.get("inner_split_strategy", "kfold")

    if inner_split_strategy == "kfold":
        # Original: k-fold CV on train
        best_model, best_params, inner_score, inner_train_score = _inner_cv_sklearn(
            X_train, y_train, estimator, param_grid,
            task=task,
            inner_k=inner_k, scoring=scoring,
            search_strategy=search_strategy, n_iter=n_iter,
            random_state=random_state,
        )

    elif inner_split_strategy == "holdout":
        # OOD holdout: F_a as inner train, F_b as inner val (chemically distinct subsets)
        # X_inner_train, y_inner_train, X_inner_val, y_inner_val are passed via kwargs from run_nested_cv.
        X_inner_train = kwargs["inner_train_X"]
        y_inner_train = kwargs["inner_train_y"]
        X_inner_val   = kwargs["inner_val_X"]
        y_inner_val   = kwargs["inner_val_y"]

        best_model, best_params, inner_score, inner_train_score = _inner_holdout_sklearn(
            X_inner_train, y_inner_train, X_inner_val, y_inner_val,
            estimator, param_grid,
            task=task,
            scoring=scoring, search_strategy=search_strategy,
            n_iter=n_iter, random_state=random_state,
        )

    elif inner_split_strategy == "random_shuffle":
        # Random shuffle: mix train, split randomly
        val_frac = kwargs.get("holdout_val_fraction", 0.2)

        if task == "hi":
            X_tr, X_vl, y_tr, y_vl = train_test_split(
                X_train, y_train, test_size=val_frac,
                random_state=random_state, stratify=y_train,
            )
        else:
            X_tr, X_vl, y_tr, y_vl = train_test_split(
                X_train, y_train, test_size=val_frac,
                random_state=random_state,
            )

        best_model, best_params, inner_score, inner_train_score = _inner_holdout_sklearn(
            X_tr, y_tr, X_vl, y_vl,
            estimator, param_grid,
            task=task,
            scoring=scoring, search_strategy=search_strategy,
            n_iter=n_iter, random_state=random_state,
        )

    else:
        raise ValueError(f"Unknown inner_split_strategy: {inner_split_strategy}")

    # Predict
    if hasattr(best_model, "predict_proba"):
        train_preds = best_model.predict_proba(X_train)[:, 1]
        test_preds = best_model.predict_proba(X_test)[:, 1]
    elif hasattr(best_model, "decision_function"):
        train_preds = best_model.decision_function(X_train)
        test_preds = best_model.decision_function(X_test)
    else:
        train_preds = best_model.predict(X_train)
        test_preds = best_model.predict(X_test)

    # Evaluate
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

    # Save
    if save_results:
        save_predictions(train_df, train_preds, task, dataset, model_name, fp_type, "train", fold_idx)
        save_predictions(test_df, test_preds, task, dataset, model_name, fp_type, "test", fold_idx)
        save_params(best_params, task, dataset, model_name, fp_type, fold_idx,
                    extra_info={
                        "inner_split_strategy": inner_split_strategy,
                        "inner_selection_score": inner_score,
                        "inner_train_score": inner_train_score,
                        "train_metrics": train_metrics,
                        "test_metrics": test_metrics,
                        "time_seconds": round(elapsed, 1),
                    })

    return {
        "fold": fold_idx,
        "best_params": best_params,
        "inner_split_strategy": inner_split_strategy,
        "inner_selection_score": inner_score,
        "inner_train_score": inner_train_score,
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
    inner_k: int = 2,
    scoring: str = "average_precision",
    search_strategy: str = "grid",
    n_iter: int = 50,
    random_state: int = 42,
    folds: List[int] = [1, 2, 3],
    save_results: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Run the full nested cross-validation across all outer folds.

    Returns dict with:
        fold_results    - list of per-fold result dicts
        aggregated      - mean ± std across folds
        experiment_id   - string identifier
    """
    logger.info(f"\n{'#'*60}")
    logger.info(f"EXPERIMENT: {model_name} + {fp_type} on {task}/{dataset}")
    logger.info(f"{'#'*60}")

    inner_split_strategy = kwargs.get("inner_split_strategy", "kfold")

    if inner_split_strategy == "holdout" and task != "hi":
        raise ValueError(
            "The OOD holdout strategy based on test_1/test_2/test_3 reconstruction "
            "is currently valid only for Hi tasks. For Lo, use 'kfold' or 'random_shuffle', "
            "or implement a dedicated cluster-aware holdout."
        )

    fold_results = []

    for fold_idx in folds:
        train_df, test_df = load_fold(task, dataset, fold_idx)

        inner_split_strategy = kwargs.get("inner_split_strategy", "kfold")
        extra_kwargs = dict(kwargs)

        if inner_split_strategy == "holdout":
            # Reconstruct F1, F2, F3 from the test sets of the 3 outer folds:
            #   test_1.csv = F3, test_2.csv = F2, test_3.csv = F1
            # For each outer fold, train = union of 2 subsets, test = remaining subset.
            # We use one subset as inner train and the other as inner validation,
            # so inner val is chemically OOD 
            #
            # outer fold 1: train = F1∪F2, test = F3 → inner train = F1 (test_3), inner val = F2 (test_2)
            # outer fold 2: train = F1∪F3, test = F2 → inner train = F1 (test_3), inner val = F3 (test_1)
            # outer fold 3: train = F2∪F3, test = F1 → inner train = F2 (test_2), inner val = F3 (test_1)
            inner_fold_map = {
                1: (3, 2),   # (fold_idx whose test_i is inner train, fold_idx whose test_i is inner val)
                2: (3, 1),
                3: (2, 1),
            }
            train_inner_idx, val_inner_idx = inner_fold_map[fold_idx]

            # load_fold returns (train_df, test_df); we take the TEST portion as F_i
            _, inner_train_df = load_fold(task, dataset, train_inner_idx)
            _, inner_val_df   = load_fold(task, dataset, val_inner_idx)

            inner_train_cache = get_feature_cache_path(task, dataset, fp_type, "test", train_inner_idx)
            inner_val_cache   = get_feature_cache_path(task, dataset, fp_type, "test", val_inner_idx)

            X_inner_train = compute_fingerprints(
                inner_train_df["smiles"].tolist(), fp_type, inner_train_cache
            )
            X_inner_val = compute_fingerprints(
                inner_val_df["smiles"].tolist(), fp_type, inner_val_cache
            )
            y_inner_train = inner_train_df["value"].values
            y_inner_val   = inner_val_df["value"].values

            # Cast to bool for KNN/Jaccard distance (same logic as run_single_fold)
            if model_name.startswith("knn") and fp_type in ["ecfp4", "maccs", "rdkit_topo"]:
                X_inner_train = X_inner_train.astype(bool)
                X_inner_val = X_inner_val.astype(bool)

            extra_kwargs["inner_train_X"] = X_inner_train
            extra_kwargs["inner_train_y"] = y_inner_train
            extra_kwargs["inner_val_X"]   = X_inner_val
            extra_kwargs["inner_val_y"]   = y_inner_val

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
            **extra_kwargs,
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