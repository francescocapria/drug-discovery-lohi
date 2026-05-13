# CORE CODE CONTEXT
# Generated on: Wed May 13 12:58:17 AM CEST 2026


================================================================================
FILE: training/train_model.py
================================================================================

"""
Main training script 

Usage example:
    python training/train_model.py --config configs/hi/drd2/knn/knn_ecfp4_drd2_hi.yaml

This script:
1. Loads the YAML config of the project
2. Construct the right model --> Maps model name --> sklearn estimator factory
3. Runs the full nested CV pipeline
4. Saves predictions + params to the folder results/
"""

import sys
import argparse
import logging
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_config
from utils.cv_pipeline import run_nested_cv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tanimoto kernel (for SVM on binary fingerprints)
# ---------------------------------------------------------------------------

def tanimoto_kernel(X, Y):
    XY = X @ Y.T
    X_sq = (X ** 2).sum(axis=1).reshape(-1, 1)
    Y_sq = (Y ** 2).sum(axis=1).reshape(-1, 1)
    return XY / (X_sq + Y_sq.T - XY)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_estimator_factory(model_selected: dict, task: str, fp_type: str = "ecfp4"):
    """
    Map the model section config to an sklearn estimator factory 
    
    Returns a callable that produces a fresh, unfitted estimator with
    the fixed params already set before.
    """
    name = model_selected["name"]
    fixed = model_selected.get("fixed", {})

    if name == "knn":
        if task == "lo":
            from sklearn.neighbors import KNeighborsRegressor
            def factory():
                return KNeighborsRegressor(**fixed)
            return factory
        else:
            from sklearn.neighbors import KNeighborsClassifier
            def factory():
                return KNeighborsClassifier(**fixed)
            return factory

    elif name == "svm":
        from sklearn.svm import SVC, SVR

        # tanimoto kernel
        kernel_type = fixed.get("kernel")

        # Scaling: always for rdkit_desc, for Lo binary fingerprints
        # except Tanimoto which must operate on unscaled binary fingerprints
        use_scaling = fp_type == "rdkit_desc" or (task == "lo" and kernel_type != "tanimoto")

        if kernel_type == "tanimoto":
            fixed = {k: v for k, v in fixed.items() if k != "kernel"}
            kernel_arg = tanimoto_kernel
        else:
            kernel_arg = kernel_type

        SvmClass = SVR if task == "lo" else SVC

        def factory():
            if kernel_type == "tanimoto":
                model = SvmClass(kernel=tanimoto_kernel, **fixed)
            else:
                model = SvmClass(**fixed)

            if use_scaling:
                return Pipeline([("scaler", StandardScaler()), ("model", model)])
            return model

        return factory

    elif name == "gb":
        if task == "lo":
            from sklearn.ensemble import GradientBoostingRegressor
            def factory():
                return GradientBoostingRegressor(**fixed)
            return factory
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            def factory():
                return GradientBoostingClassifier(**fixed)
            return factory

    elif name == "rf":
        if task == "lo":
            from sklearn.ensemble import RandomForestRegressor
            def factory():
                return RandomForestRegressor(**fixed)
            return factory
        else:
            from sklearn.ensemble import RandomForestClassifier
            def factory():
                return RandomForestClassifier(**fixed)
            return factory

    elif name == "lr":
        from sklearn.linear_model import LogisticRegression
        def factory():
            model = LogisticRegression(**fixed)
            if fp_type == "rdkit_desc":
                return Pipeline([("scaler", StandardScaler()), ("model", model)])
            return model
        return factory
    
    elif name == "linreg":
        from sklearn.linear_model import LinearRegression
        def factory():
            model = LinearRegression(**fixed)
            if fp_type == "rdkit_desc":
                return Pipeline([("scaler", StandardScaler()), ("model", model)])
            return model
        return factory
    
    elif name == "dt":
        if task == "lo":
            from sklearn.tree import DecisionTreeRegressor
            def factory():
                return DecisionTreeRegressor(**fixed)
            return factory
        else:
            from sklearn.tree import DecisionTreeClassifier
            def factory():
                return DecisionTreeClassifier(**fixed)
            return factory

    elif name == "dummy":
        if task == "lo":
            from sklearn.dummy import DummyRegressor
            def factory():
                return DummyRegressor(strategy="mean")
            return factory
        else:
            from sklearn.dummy import DummyClassifier
            def factory():
                return DummyClassifier(**fixed)
            return factory

    elif name == "xgb":
        if task == "lo":
            from xgboost import XGBRegressor
            def factory():
                return XGBRegressor(**fixed)
            return factory
        else:
            from xgboost import XGBClassifier
            def factory():
                return XGBClassifier(**fixed)
            return factory

    else:
        raise ValueError(
            f"Unknown model name: '{name}'. "
            f"Available: knn, svm, gb, rf, lr, dt, dummy, xgb"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a model using nested CV on Lo-Hi benchmark"
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Which outer folds to run (default: 1 2 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without training",
    )

    args = parser.parse_args()

    # Load config (transform yaml file)
    cfg = load_config(args.config)
    config_name = Path(args.config).stem
    
    task = cfg["experiment"]["task"] 

    if config_name.endswith(f"_{task}"):
        model_name = config_name[:-(len(task) + 1)]
    else:
        model_name = config_name  

    fp_config = cfg["fingerprint"]
    if "types" in fp_config:
        fp_list = fp_config["types"]
    elif "type" in fp_config:
        fp_list = [fp_config["type"]]
    else:
        raise ValueError("Config must have fingerprint.type or fingerprint.types")
    
    # if --dry-run is used, it doesn't train nothing, only check the config. Useful for debug
    if args.dry_run:
        import json
        print(json.dumps(cfg, indent=2))
        return

    # Run nested CV for each fingerprint
    for fp_type in fp_list:
        factory = get_estimator_factory(cfg["model"], cfg["experiment"]["task"], fp_type)

        param_grid = cfg["model"]["search"].copy()
        if isinstance(factory(), Pipeline):
            param_grid = {f"model__{k}": v for k, v in param_grid.items()}

        # model_name include sia il nome del file config che il fingerprint
        model_name = f"{config_name}_{fp_type}"

        results = run_nested_cv(
            task=cfg["experiment"]["task"],
            dataset=cfg["experiment"]["dataset"],
            fp_type=fp_type,
            model_name=model_name,
            estimator_factory=factory,
            param_grid=param_grid,
            inner_k=cfg["cv"]["inner_k"],
            scoring=cfg["cv"]["scoring"],
            search_strategy=cfg["cv"]["search_strategy"],
            n_iter=cfg["cv"]["n_iter"],
            random_state=cfg["cv"]["random_state"],
            folds=args.folds,
            inner_split_strategy=cfg["cv"]["inner_split_strategy"],
            holdout_val_fraction=cfg["cv"]["holdout_val_fraction"],
        )

        # Summary
        print("\n" + "=" * 60)
        print(f"EXPERIMENT COMPLETE: {results['experiment_id']}")
        print("=" * 60)
        print(f"\nAggregated test metrics:")
        for k, v in results["aggregated"].items():
            print(f"  {k}: {v}")
        print(f"\nPer-fold best params:")
        for r in results["fold_results"]:
            print(f"  Fold {r['fold']}: {r['best_params']}")
        print()


if __name__ == "__main__":
    main()

================================================================================
FILE: utils/config_loader.py
================================================================================

"""
Config loader for experiment YAML files.

Each YAML file defines ONE experiment: model + fingerprint + dataset + search space.

"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default config values
# ---------------------------------------------------------------------------

DEFAULTS = {
    "cv": {
        "inner_k": 3,
        "scoring": "average_precision",
        "search_strategy": "grid",
        "n_iter": 50,
        "random_state": 42,
        "inner_split_strategy": "kfold",       # "kfold" | "holdout" | "random_shuffle"
        "holdout_val_fraction": 0.2,            # only with random_shuffle
    }
}


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate a YAML experiment config.

    config_path : Path to the YAML config file.

    Return: dict with sections: experiment, fingerprint, model, cv
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Validate 
    for section in ["experiment", "fingerprint", "model"]:
        if section not in cfg:
            raise ValueError(f"Config missing required section: '{section}'")

    # Fill defaults for cv section
    if "cv" not in cfg:
        cfg["cv"] = {}
    for key, default_val in DEFAULTS["cv"].items():
        if key not in cfg["cv"]:
            cfg["cv"][key] = default_val

    # Validate experiment section
    exp = cfg["experiment"]
    for key in ["task", "dataset"]:
        if key not in exp:
            raise ValueError(f"experiment.{key} is required")
    if exp["task"] not in ("hi", "lo"):
        raise ValueError(f"experiment.task must be 'hi' or 'lo', got '{exp['task']}'")
    
    # Validate inner split strategy
    valid_strategies = ("kfold", "holdout", "random_shuffle")
    strategy = cfg["cv"].get("inner_split_strategy", "kfold")
    if strategy not in valid_strategies:
        raise ValueError(
            f"cv.inner_split_strategy must be one of {valid_strategies}, got '{strategy}'"
        )

    # Validate fingerprint section
    fp = cfg["fingerprint"]
    if "type" not in fp and "types" not in fp:
        raise ValueError("fingerprint.type or fingerprint.types is required")

    # Validate model section
    model = cfg["model"]
    if "name" not in model:
        raise ValueError("model.name is required")
    if "search" not in model:
        model["search"] = {}
    if "fixed" not in model:
        model["fixed"] = {}

    logger.info(
        f"Loaded config: {exp.get('name', config_path.stem)} | "
        f"model={model['name']} fp={fp.get('type', fp.get('types'))} "
        f"task={exp['task']} dataset={exp['dataset']}"
    )

    return cfg


def config_to_experiment_id(cfg: Dict[str, Any]) -> str:
    """Generate a unique experiment ID from config."""
    exp = cfg["experiment"]
    return exp.get("name", f"{cfg['model']['name']}_{cfg['fingerprint']['type']}_{exp['dataset']}_{exp['task']}")

================================================================================
FILE: utils/cv_pipeline.py
================================================================================

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

================================================================================
FILE: utils/io_utils.py
================================================================================

"""
I/O utilities for the Lo-Hi project.

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
    params : Best hyperparameters from inner CV.
    extra_info : Additional info (inner_cv_score, training_time, etc.)

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

================================================================================
FILE: utils/fingerprints.py
================================================================================

"""
Fingerprint computation

All functions return NumPy arrays of shape (n_molecules, n_features).

Computed fingerprints can be optionally cached on disk in the features/ folder 

Supported fingerprint types:
- ECFP4 (Morgan fingerprints, radius=2)
- MACCS keys (167-bit structural keys)
- RDKit topological fingerprints (path-based)
- RDKit 2D descriptors

"""

import os
import logging
from typing import List, Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

logger = logging.getLogger(__name__)


def smiles_to_mols(smiles_list: List[str]):
    mols = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning(f"Skipping invalid SMILES at index {i}: {smi[:50]}")
            mol = Chem.MolFromSmiles("C")  
        mols.append(mol)
    return mols


def compute_ecfp4(smiles_list: List[str], n_bits: int = 1024) -> np.ndarray:
    """Compute ECFP4 (Morgan radius=2) fingerprints."""
    mols = smiles_to_mols(smiles_list)
    X = np.zeros((len(mols), n_bits), dtype=np.uint8)
    gen = GetMorganGenerator(radius=2, fpSize=n_bits)

    for i, mol in enumerate(mols):
        fp = gen.GetFingerprintAsNumPy(mol)
        X[i] = fp

    return X


def compute_maccs(smiles_list: List[str]) -> np.ndarray:
    """Compute MACCS keys fingerprints."""
    mols = smiles_to_mols(smiles_list)
    X = np.zeros((len(mols), 167), dtype=np.uint8)

    for i, mol in enumerate(mols):
        fp = MACCSkeys.GenMACCSKeys(mol)
        DataStructs.ConvertToNumpyArray(fp, X[i])

    return X


def compute_rdkit_descriptors(smiles_list: List[str]) -> np.ndarray:
    mols = smiles_to_mols(smiles_list)
    X = np.array([list(Descriptors.CalcMolDescriptors(mol).values()) for mol in mols], dtype=np.float64)

    for j in range(X.shape[1]):
        col = X[:, j]
        mask = ~np.isfinite(col)
        if mask.any():
            median = np.nanmedian(col)
            col[mask] = median if not np.isnan(median) else 0.0

    X = np.clip(X, -1e15, 1e15)

    return X


def compute_rdkit_topo(
    smiles_list: List[str],
    min_path: int = 1,
    max_path: int = 7,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute RDKit topological (path-based) fingerprints."""
    mols = smiles_to_mols(smiles_list)
    X = np.zeros((len(mols), n_bits), dtype=np.uint8)

    for i, mol in enumerate(mols):
        fp = Chem.RDKFingerprint(
            mol,
            minPath=min_path,
            maxPath=max_path,
            fpSize=n_bits,
        )
        DataStructs.ConvertToNumpyArray(fp, X[i])

    return X


def compute_fingerprints(
    smiles_list: List[str],
    fp_type: str,
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute molecular fingerprints with optional caching.
    
    """
    if cache_path is not None and os.path.exists(cache_path):
        logger.info(f"Loading fingerprints from cache: {cache_path}")
        data = np.load(cache_path)
        return data["X"]

    if fp_type == "ecfp4":
        X = compute_ecfp4(smiles_list)
    elif fp_type == "maccs":
        X = compute_maccs(smiles_list)
    elif fp_type == "rdkit_topo":
        X = compute_rdkit_topo(smiles_list)
    elif fp_type == "rdkit_desc":
        X = compute_rdkit_descriptors(smiles_list)
    else:
        raise ValueError(
            "fp_type must be one of: 'ecfp4', 'maccs', 'rdkit_topo', 'rdkit_desc'"
        )

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, X=X)
        logger.info(f"Saved fingerprint cache to: {cache_path}")

    return X

================================================================================
FILE: utils/metrics.py
================================================================================

"""
Evaluation metrics for the Lo-Hi benchmark.

Hi task:
- PR AUC
- ROC AUC
- BEDROC
- F1 at threshold 0.5 (only if scores are probabilities)
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
from rdkit.ML.Scoring.Scoring import CalcBEDROC
import warnings
from scipy.stats import ConstantInputWarning

# ---------------------------------------------------------------------------
# Hi metrics (Hit Identification — binary classification)
# ---------------------------------------------------------------------------

def get_hi_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    """
    Compute metrics for the Hi (Hit Identification) task.

    y_true : Ground truth binary labels (0/1).
    y_score : Predicted score for the positive class.
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

        # for identical scores
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
    else:
        f1 = float("nan")

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "bedroc": round(bedroc, 4),
        "f1_at_05": round(f1, 4) if not np.isnan(f1) else float("nan"),
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

    y_true : Ground truth continuous activity values.
    y_pred : Predicted values or scores.
    cluster_ids :Cluster identifier for each sample.
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

        if mask.sum() < 3:
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

    fold_metrics : One metrics dictionary per outer fold.
    """
    all_keys = fold_metrics[0].keys()
    aggregated = {}

    for key in all_keys:
        values = [
            metrics[key]
            for metrics in fold_metrics
            if isinstance(metrics[key], (int, float, np.floating))
            and not np.isnan(metrics[key])
        ]

        if len(values) > 0:
            aggregated[f"{key}_mean"] = round(np.mean(values), 4)
            aggregated[f"{key}_std"] = round(np.std(values), 4)

    return aggregated
