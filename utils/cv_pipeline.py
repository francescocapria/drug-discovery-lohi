"""
Nested cross-validation pipeline for the Lo-Hi benchmark.

The pipeline uses the 3 predefined Lo-Hi outer folds. For each fold, models are
selected only on train_i and evaluated once on the held-out test_i.

Supported inner-selection strategies:
- kfold: standard inner CV on train_i;
- holdout: Hi-only OOD holdout, reconstructed from the Lo-Hi fold subsets;
- random_shuffle: random inner holdout matched to the same train/validation
  proportion as the corresponding OOD holdout.

Feature importance:
- Decision Trees use permutation importance as the main feature ranking, computed
  post-hoc on a held-out evaluation set, usually the outer test fold. The native
  sklearn impurity importance is still saved only as a diagnostic.
- Logistic Regression and Linear SVM keep the standard coefficient-based ranking
  using absolute weights. No permutation importance is computed for them.

Optional artifacts include fitted models, parameters, predictions, complexity
metrics, feature-importance tables and CV/search results.
"""

import time
import json
import joblib
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
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from utils.fingerprints import compute_fingerprints
from utils.metrics import get_hi_metrics, get_lo_metrics, aggregate_fold_metrics
from utils.io_utils import (
    load_fold,
    save_predictions,
    save_params,
    get_feature_cache_path,
    get_results_dir,
)

import logging
from collections import Counter

logger = logging.getLogger(__name__)


# Default permutation-importance settings (can be overridden via kwargs)
DEFAULT_PERM_N_REPEATS = 10 # Number of times each feature is shuffled
DEFAULT_PERM_N_JOBS = -1 


# Artifact utilities


def _json_safe(value: Any) -> Any:
    """
    Convert NumPy / NaN objects into JSON-safe Python objects.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, float):
        if np.isnan(value):
            return None
        return value

    return value


def _extract_model_from_pipeline(model: BaseEstimator) -> BaseEstimator:
    """
    If the fitted estimator is a sklearn Pipeline, return the final model step.
    Otherwise return the model itself.
    This is useful for RDKit descriptors
    """
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def _get_feature_names(fp_type: str, n_features: int) -> List[str]:
    """
    Generate generic feature names for fingerprints/descriptors.

    For ECFP4/MACCS this gives bit indices
    For RDKit descriptors this gives descriptor indices
    """
    if fp_type in ["ecfp4", "maccs", "rdkit_topo"]:
        return [f"{fp_type}_bit_{i}" for i in range(n_features)]

    if fp_type == "rdkit_desc":
        return [f"rdkit_desc_{i}" for i in range(n_features)]

    return [f"{fp_type}_feature_{i}" for i in range(n_features)]


def _tree_minimum_depths(tree_model: BaseEstimator, n_features: int) -> Dict[int, int]:
    """
    Compute the minimum depth at which each feature appears in a fitted Decision Tree.

    """
    if not hasattr(tree_model, "tree_"):
        return {}

    tree = tree_model.tree_
    feature = tree.feature
    children_left = tree.children_left
    children_right = tree.children_right

    min_depth = {}

    stack = [(0, 0)]  # (node_id, depth)
    while stack: # for every node in the tree
        node_id, depth = stack.pop()
        feat_idx = int(feature[node_id])

        if feat_idx >= 0: # it's not a leaf, but a split
            if feat_idx not in min_depth:
                min_depth[feat_idx] = depth
            else:
                min_depth[feat_idx] = min(min_depth[feat_idx], depth) # I keep only the min distance from the root

            left = children_left[node_id]
            right = children_right[node_id]

            if left != -1:
                stack.append((left, depth + 1))
            if right != -1:
                stack.append((right, depth + 1))

    return min_depth


def _extract_complexity_metrics(
    fitted_model: BaseEstimator,
    model_name: str,
    fp_type: str,
    n_features: int,
) -> Dict[str, Any]:
    """
    Extract model complexity metrics from the fitted estimator.

    For Logistic Regression / linear models:
        - number of coefficients
        - number of non-zero coefficients
        - sparsity
        - L1 norm
        - L2 norm
        - intercept
        - C when available
        - l1_ratio when available

    For linear SVM:
        - C
        - L2 norm of w
        - approximate margin = 1 / ||w||_2

    For Decision Tree:
        - effective depth
        - number of nodes
        - number of leaves
        - number of used features
        - selected tree regularization parameters
    """
    base_model = _extract_model_from_pipeline(fitted_model)

    complexity = {
        "model_class": base_model.__class__.__name__,
        "model_name": model_name,
        "fp_type": fp_type,
        "n_features": int(n_features),
    }

    # Linear models: Logistic Regression, Linear SVM, LinearRegression, etc.
    # 
    if hasattr(base_model, "coef_"):
        coef = np.asarray(base_model.coef_)

        if coef.ndim > 1:
            coef_flat = coef.ravel() # If coef_ is multidimensional, flatten it, we want (n_features,) not (1, n_features))
        else:
            coef_flat = coef

        abs_coef = np.abs(coef_flat)
        nonzero = int(np.sum(abs_coef > 0.0))
        l1_norm = float(np.sum(abs_coef))
        l2_norm = float(np.linalg.norm(coef_flat, ord=2))

        complexity.update({
            "coefficient_shape": list(coef.shape),
            "n_coefficients": int(coef_flat.shape[0]),
            "n_nonzero_coefficients": nonzero,
            "sparsity": float(1.0 - nonzero / coef_flat.shape[0]) if coef_flat.shape[0] > 0 else None, # If I have 1024 features and 200 coef -->  sparsity = 1 - 200/1024 = 0.805
            "l1_norm": l1_norm,
            "l2_norm": l2_norm,
            "approx_margin": float(1.0 / l2_norm) if l2_norm > 0 else None, # Principally for svm 
        })

        if hasattr(base_model, "intercept_"):
            intercept = np.asarray(base_model.intercept_).ravel()
            complexity["intercept"] = intercept.tolist()

        if hasattr(base_model, "C"):
            complexity["C"] = base_model.C

        if hasattr(base_model, "class_weight"):
            complexity["class_weight"] = base_model.class_weight

        if hasattr(base_model, "penalty"):
            complexity["penalty"] = base_model.penalty

        if hasattr(base_model, "l1_ratio"):
            complexity["l1_ratio"] = base_model.l1_ratio

        if hasattr(base_model, "dual"):
            complexity["dual"] = base_model.dual

        if hasattr(base_model, "loss"):
            complexity["loss"] = base_model.loss

    # SVM support-vector information, if available
    if hasattr(base_model, "n_support_"):
        complexity["n_support_per_class"] = np.asarray(base_model.n_support_).tolist()
        complexity["n_support_total"] = int(np.sum(base_model.n_support_))

    if hasattr(base_model, "support_"):
        complexity["n_support_total"] = int(len(base_model.support_))

    # Decision Tree metrics
    if hasattr(base_model, "tree_"):
        tree = base_model.tree_
        used_features = tree.feature[tree.feature >= 0] # only split nodes
        unique_used_features = np.unique(used_features)

        complexity.update({
            "effective_depth": int(base_model.get_depth()),
            "n_nodes": int(tree.node_count),
            "n_leaves": int(base_model.get_n_leaves()),
            "n_features_used": int(len(unique_used_features)),
            "used_feature_fraction": float(len(unique_used_features) / n_features) if n_features > 0 else None,
        })

        for attr in [
            "criterion",
            "max_depth",
            "max_features",
            "min_samples_leaf",
            "min_samples_split",
            "ccp_alpha",
            "class_weight",
        ]:
            if hasattr(base_model, attr):
                complexity[attr] = getattr(base_model, attr)

        min_depths = _tree_minimum_depths(base_model, n_features)
        if min_depths:
            depth_values = list(min_depths.values())
            complexity["feature_min_depth_mean"] = float(np.mean(depth_values))
            complexity["feature_min_depth_std"] = float(np.std(depth_values))
            complexity["feature_min_depth_min"] = int(np.min(depth_values))
            complexity["feature_min_depth_max"] = int(np.max(depth_values))

    return _json_safe(complexity)


def _extract_feature_importance(
    fitted_model: BaseEstimator,
    model_name: str,
    fp_type: str,
    n_features: int,
    X_eval: Optional[np.ndarray] = None,
    y_eval: Optional[np.ndarray] = None,
    task: str = "hi",
    perm_n_repeats: int = DEFAULT_PERM_N_REPEATS,
    perm_scoring: Optional[str] = None,
    perm_n_jobs: int = DEFAULT_PERM_N_JOBS,
    random_state: int = 42,
    eval_set_name: str = "test",
) -> Optional[pd.DataFrame]:
    """
    Extract feature-importance information for fitted models.

    For linear models such as Logistic Regression and Linear SVM, features are ranked
    by the absolute value of their learned coefficients. The signed coefficient,
    absolute weight, normalized weight, direction and rank are saved.

    For Decision Trees, features are ranked by permutation importance computed on a
    held-out evaluation set, usually the outer test fold. The native sklearn
    impurity importance is also saved as a diagnostic, together with minimum depth
    and whether each feature is used in the tree.

    If permutation importance cannot be computed for a tree, the function falls back
    to impurity-based ordering.
    """
    base_model = _extract_model_from_pipeline(fitted_model)
    feature_names = _get_feature_names(fp_type, n_features)

    # Linear model coefficients  
    if hasattr(base_model, "coef_"):
        coef = np.asarray(base_model.coef_)

        if coef.ndim > 1:
            coef_flat = coef.ravel()
        else:
            coef_flat = coef

        if coef_flat.shape[0] != n_features:
            # Multi-class or unexpected shape: still save what we can.
            feature_names = [f"{fp_type}_coef_{i}" for i in range(coef_flat.shape[0])]

        abs_weight = np.abs(coef_flat)
        denom = abs_weight.sum()

        if denom > 0:
            normalized = abs_weight / denom
        else:
            normalized = np.zeros_like(abs_weight)

        df = pd.DataFrame({
            "feature_idx": np.arange(coef_flat.shape[0], dtype=int),
            "feature_name": feature_names,
            "raw_weight": coef_flat.astype(float),
            "abs_weight": abs_weight.astype(float),
            "normalized_abs_importance": normalized.astype(float),
        })

        df["direction"] = np.where(
            df["raw_weight"] > 0,
            "positive",
            np.where(df["raw_weight"] < 0, "negative", "zero")
        )

        df = df.sort_values("abs_weight", ascending=False).reset_index(drop=True)
        df["rank_abs_weight"] = np.arange(1, len(df) + 1)

        return df

    # Decision Tree  ->  permutation importance is the PRIMARY ranking
    # impurity-based importance is kept as a diagnostic column
    if hasattr(base_model, "feature_importances_"):
        importance = np.asarray(base_model.feature_importances_)
        min_depths = _tree_minimum_depths(base_model, n_features)

        df = pd.DataFrame({
            "feature_idx": np.arange(n_features, dtype=int),
            "feature_name": feature_names,
            "tree_importance": importance.astype(float),
            "minimum_depth": [
                min_depths.get(i, np.nan) for i in range(n_features)
            ],
            "used_in_tree": [
                i in min_depths for i in range(n_features)
            ],
        })

        denom = df["tree_importance"].sum()
        if denom > 0:
            df["normalized_tree_importance"] = df["tree_importance"] / denom
        else:
            df["normalized_tree_importance"] = 0.0

        # Diagnostic impurity-based rank (kept for backward compatibility)
        df["rank_tree_importance"] = (
            df["tree_importance"].rank(ascending=False, method="first").astype(int)
        )

        # --- Permutation importance (PRIMARY) ---
        perm_df = None
        if X_eval is not None and y_eval is not None:
            if perm_scoring is None:
                perm_scoring = "average_precision" if task == "hi" else "neg_mean_absolute_error"
            try:
                perm = permutation_importance(
                    fitted_model,            # full pipeline: preprocessing applied
                    X_eval,
                    y_eval,
                    scoring=perm_scoring,
                    n_repeats=perm_n_repeats,
                    random_state=random_state,
                    n_jobs=perm_n_jobs,
                )
                perm_df = pd.DataFrame({
                    "feature_idx": np.arange(len(perm.importances_mean), dtype=int),
                    "permutation_importance_mean": perm.importances_mean.astype(float),
                    "permutation_importance_std": perm.importances_std.astype(float),
                })
                perm_df["permutation_scoring"] = perm_scoring
                perm_df["permutation_eval_set"] = eval_set_name
                perm_df["permutation_n_repeats"] = int(perm_n_repeats)

                logger.info(
                    f"  DT permutation importance computed on '{eval_set_name}' set "
                    f"(scoring={perm_scoring}, n_repeats={perm_n_repeats})"
                )
            except Exception as exc:  # robust fallback: impurity ordering
                logger.warning(
                    f"  DT permutation importance failed ({exc}); "
                    f"falling back to impurity-based ordering."
                )
                perm_df = None

        if perm_df is not None:
            df = df.merge(perm_df, on="feature_idx", how="left")
            df["permutation_importance_rank"] = (
                df["permutation_importance_mean"]
                .rank(ascending=False, method="first")
                .astype(int)
            )
            df = df.sort_values(
                ["permutation_importance_mean", "feature_idx"],
                ascending=[False, True],
            ).reset_index(drop=True)
        else:
            # Permutation unavailable: original impurity-based ordering
            df = df.sort_values(
                ["tree_importance", "minimum_depth"],
                ascending=[False, True],
                na_position="last",
            ).reset_index(drop=True)

        return df

    return None


def _save_model_artifacts(
    fitted_model: BaseEstimator,
    search_object: Optional[Any],
    task: str,
    dataset: str,
    model_name: str,
    fp_type: str,
    fold_idx: int,
    complexity: Optional[Dict[str, Any]] = None,
    feature_importance: Optional[pd.DataFrame] = None,
    artifacts: Optional[Dict[str, bool]] = None,
) -> None:
    """
    Save optional artifacts required for protocol explainability notebooks.

    Saved files:
        model_fold_{i}.joblib
        complexity_fold_{i}.json
        feature_importance_fold_{i}.csv
        cv_results_fold_{i}.csv
    """
    if artifacts is None:
        artifacts = {}

    result_dir = get_results_dir(task, dataset, model_name, fp_type)

    if artifacts.get("save_model", False):
        model_path = result_dir / f"model_fold_{fold_idx}.joblib"
        joblib.dump(fitted_model, model_path)
        logger.info(f"Saved fitted model to {model_path}")

    if artifacts.get("save_complexity", False) and complexity is not None:
        complexity_path = result_dir / f"complexity_fold_{fold_idx}.json"
        with open(complexity_path, "w") as f:
            json.dump(_json_safe(complexity), f, indent=2)
        logger.info(f"Saved complexity metrics to {complexity_path}")

    if artifacts.get("save_feature_importance", False) and feature_importance is not None:
        fi_path = result_dir / f"feature_importance_fold_{fold_idx}.csv"
        feature_importance.to_csv(fi_path, index=False)
        logger.info(f"Saved feature importance to {fi_path}")

    if artifacts.get("save_cv_results", False) and search_object is not None:
        cv_results_path = result_dir / f"cv_results_fold_{fold_idx}.csv"
        cv_results_df = pd.DataFrame(search_object.cv_results_)
        cv_results_df.to_csv(cv_results_path, index=False)
        logger.info(f"Saved CV/search results to {cv_results_path}")


# Inner CV: hyperparameter search on a single outer fold

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
) -> Tuple[BaseEstimator, dict, float, Optional[float], Any]:

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

    return search.best_estimator_, search.best_params_, search.best_score_, best_train_score, search


# Inner holdout

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
) -> Tuple[BaseEstimator, dict, float, Optional[float], Any]:
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

    return search.best_estimator_, search.best_params_, search.best_score_, best_train_score, search



# Random shuffle helpers: same train/validation proportion as OOD holdout

def _infer_fold_origin_labels(
    train_df: pd.DataFrame,
    inner_train_df: pd.DataFrame,
    inner_val_df: pd.DataFrame,
) -> np.ndarray:
    """
    Infer, for each molecule in the outer training set, whether it belongs to
    the OOD inner-train subset, the OOD inner-validation subset, both, or neither.
    """
    inner_train_smiles = set(inner_train_df["smiles"].astype(str))
    inner_val_smiles = set(inner_val_df["smiles"].astype(str))

    origins = []

    for smi in train_df["smiles"].astype(str):
        in_train_subset = smi in inner_train_smiles
        in_val_subset = smi in inner_val_smiles

        if in_train_subset and in_val_subset:
            origins.append("shared")
        elif in_train_subset:
            origins.append("inner_train_origin")
        elif in_val_subset:
            origins.append("inner_val_origin")
        else:
            origins.append("unknown")

    return np.asarray(origins)


def _make_joint_stratify_labels(
    y: np.ndarray,
    fold_origin: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Build joint stratification labels.

    For Hi:
        target + fold origin

    Example:
        y=1 | origin=inner_train_origin
        y=0 | origin=inner_val_origin

    This allows random_shuffle to preserve both:
        - class balance;
        - original fold/subset composition.
    """
    if fold_origin is None:
        return None

    y = np.asarray(y)
    fold_origin = np.asarray(fold_origin)

    if len(y) != len(fold_origin):
        return None

    labels = np.asarray([
        f"y={target}|origin={origin}"
        for target, origin in zip(y, fold_origin)
    ])

    return labels

def _assert_inner_reconstructs_outer_train(
    train_df: pd.DataFrame,
    inner_train_df: pd.DataFrame,
    inner_val_df: pd.DataFrame,
    fold_idx: int,
) -> None:
    """
    Check that the OOD inner train/validation subsets exactly reconstruct
    the outer training set.

    """
    expected_n = len(train_df)
    got_n = len(inner_train_df) + len(inner_val_df)

    if got_n != expected_n:
        raise ValueError(
            f"Inner holdout reconstruction mismatch for outer fold {fold_idx}: "
            f"|inner_train| + |inner_val| = {got_n}, but |outer_train| = {expected_n}."
        )

    outer_smiles = Counter(train_df["smiles"].astype(str))
    inner_smiles = Counter(
        pd.concat(
            [
                inner_train_df["smiles"].astype(str),
                inner_val_df["smiles"].astype(str),
            ],
            ignore_index=True,
        )
    )

    if outer_smiles != inner_smiles:
        missing = list((outer_smiles - inner_smiles).elements())[:5]
        extra = list((inner_smiles - outer_smiles).elements())[:5]

        raise ValueError(
            f"Inner holdout subsets do not reconstruct the outer training SMILES "
            f"for fold {fold_idx}.\n"
            f"Missing examples: {missing}\n"
            f"Extra examples: {extra}"
        )


# Single outer fold execution

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
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"FOLD {fold_idx} | {model_name} + {fp_type} | {task}/{dataset}")
    logger.info(f"{'='*60}")

    t0 = time.time()

    # Featurize
    train_cache = get_feature_cache_path(task, dataset, fp_type, "train", fold_idx)
    test_cache = get_feature_cache_path(task, dataset, fp_type, "test", fold_idx)

    X_train, valid_train = compute_fingerprints(train_df["smiles"].tolist(), fp_type, train_cache)
    X_test, valid_test = compute_fingerprints(test_df["smiles"].tolist(), fp_type, test_cache)

    # DataFrames are already cleaned in load_fold, so no invalid SMILES should remain. This assert guards against a stale cache built before cleaning.
    assert valid_train.all(), "Unexpected invalid SMILES in train after load_fold cleaning"
    assert valid_test.all(), "Unexpected invalid SMILES in test after load_fold cleaning"

    # Cast to bool only for KNN/Jaccard distance
    if model_name.startswith("knn") and fp_type in ["ecfp4", "maccs", "rdkit_topo"]:
        X_train = X_train.astype(bool)
        X_test = X_test.astype(bool)

    y_train = train_df["value"].values
    y_test = test_df["value"].values

    # Inner hyperparameter search
    estimator = estimator_factory()
    inner_split_strategy = kwargs.get("inner_split_strategy", "kfold")
    artifacts = kwargs.get("artifacts", {})

    if inner_split_strategy == "kfold":
        # Original: k-fold CV on train
        best_model, best_params, inner_score, inner_train_score, search_object = _inner_cv_sklearn(
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

        best_model, best_params, inner_score, inner_train_score, search_object = _inner_holdout_sklearn(
            X_inner_train, y_inner_train, X_inner_val, y_inner_val,
            estimator, param_grid,
            task=task,
            scoring=scoring, search_strategy=search_strategy,
            n_iter=n_iter, random_state=random_state,
        )

    elif inner_split_strategy == "random_shuffle":
        # Random shuffle:
        # mix the same outer-training molecules, but keep the same validation proportion as the corresponding OOD holdout split.
       
        # If joint stratification is impossible because some strat are too small, we go back to target-only stratification, then to no stratification.

        val_frac = kwargs.get("random_val_fraction", None)

        if val_frac is None:
            # Backward-compatible fallback for old configs.
            val_frac = kwargs.get("holdout_val_fraction", 0.2)

        random_stratify_labels = kwargs.get("random_stratify_labels", None)

        logger.info(
            f"  Random shuffle holdout: validation fraction = {val_frac:.4f}"
        )

        stratify_candidates = []

        if task == "hi":
            if random_stratify_labels is not None:
                stratify_candidates.append(random_stratify_labels)

        # fallback: stratify by target only
            stratify_candidates.append(y_train)

        # final fallback: no stratification
        stratify_candidates.append(None)

        split_done = False
        last_error = None

        for stratify_labels in stratify_candidates:
            try:
                X_tr, X_vl, y_tr, y_vl = train_test_split(
                    X_train,
                    y_train,
                    test_size=val_frac,
                    random_state=random_state,
                    stratify=stratify_labels,
                )
                split_done = True

                if stratify_labels is random_stratify_labels:
                    logger.info("  Random shuffle stratification: target + fold origin")
                elif stratify_labels is y_train:
                    logger.info("  Random shuffle stratification: target only")
                else:
                    logger.info("  Random shuffle stratification: none")

                break

            except ValueError as e:
                last_error = e
                continue

        if not split_done:
            raise ValueError(
                "Could not create random shuffle holdout split. "
                f"Last error: {last_error}"
            )

        best_model, best_params, inner_score, inner_train_score, search_object = _inner_holdout_sklearn(
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

    # Extract optional protocol explainability artifacts
    complexity = None
    feature_importance = None

    want_complexity = artifacts.get("save_complexity", False)
    want_feature_importance = artifacts.get("save_feature_importance", False)

    # Complexity is cheap (no data needed); compute it whenever either artifact
    # is requested, preserving the original behaviour of the returned dict.
    if want_complexity or want_feature_importance:
        complexity = _extract_complexity_metrics(
            fitted_model=best_model,
            model_name=model_name,
            fp_type=fp_type,
            n_features=X_train.shape[1],
        )

    # Feature importance is computed ONLY when it will be saved, because the permutation-importance step is the expensive part.
    if want_feature_importance:
        # Permutation-importance settings (overridable via config kwargs).
        perm_n_repeats = kwargs.get("perm_n_repeats", DEFAULT_PERM_N_REPEATS)
        perm_scoring = kwargs.get("perm_scoring", None)
        perm_n_jobs = kwargs.get("perm_n_jobs", DEFAULT_PERM_N_JOBS)
        perm_eval_set = kwargs.get("perm_eval_set", "test")  # "test" or "train"

        if perm_eval_set == "train":
            X_perm, y_perm = X_train, y_train
        else:
            perm_eval_set = "test"
            X_perm, y_perm = X_test, y_test

        feature_importance = _extract_feature_importance(
            fitted_model=best_model,
            model_name=model_name,
            fp_type=fp_type,
            n_features=X_train.shape[1],
            X_eval=X_perm,
            y_eval=y_perm,
            task=task,
            perm_n_repeats=perm_n_repeats,
            perm_scoring=perm_scoring,
            perm_n_jobs=perm_n_jobs,
            random_state=random_state,
            eval_set_name=perm_eval_set,
        )

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

        _save_model_artifacts(
            fitted_model=best_model,
            search_object=search_object,
            task=task,
            dataset=dataset,
            model_name=model_name,
            fp_type=fp_type,
            fold_idx=fold_idx,
            complexity=complexity,
            feature_importance=feature_importance,
            artifacts=artifacts,
        )

    return {
        "fold": fold_idx,
        "best_params": best_params,
        "inner_split_strategy": inner_split_strategy,
        "inner_selection_score": inner_score,
        "inner_train_score": inner_train_score,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "complexity": complexity,
        "time_seconds": round(elapsed, 1),
    }


# Full nested CV (all 3 outer folds)

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
    if inner_split_strategy == "holdout" and task == "hi" and dataset == "kdr":
        raise ValueError(
            "OOD holdout reconstruction is not valid for KDR-Hi because the outer "
            "training folds are restricted to 500 molecules. Exclude KDR from the "
            "OOD-vs-random protocol comparison."
        )

    # Inner subset reconstruction map (only used for holdout / random_shuffle):
    #   test_1.csv = F3, test_2.csv = F2, test_3.csv = F1
    #
    #   outer fold 1: train = F1 u F2, test = F3
    #       inner train = F1 = test_3 ; inner val = F2 = test_2
    #   outer fold 2: train = F1 u F3, test = F2
    #       inner train = F1 = test_3 ; inner val = F3 = test_1
    #   outer fold 3: train = F2 u F3, test = F1
    #       inner train = F2 = test_2 ; inner val = F3 = test_1
    inner_fold_map = {
        1: (3, 2),
        2: (3, 1),
        3: (2, 1),
    }

    fold_results = []

    for fold_idx in folds:
        train_df, test_df = load_fold(task, dataset, fold_idx)

        # Per-fold copy of kwargs, inner-split material here.
        extra_kwargs = dict(kwargs)

        if inner_split_strategy in ["holdout", "random_shuffle"]:
            train_inner_idx, val_inner_idx = inner_fold_map[fold_idx]

            # load_fold returns (train_df, test_df); the TEST portion is F_i
            _, inner_train_df = load_fold(task, dataset, train_inner_idx)
            _, inner_val_df = load_fold(task, dataset, val_inner_idx)
            
            _assert_inner_reconstructs_outer_train(
                train_df=train_df,
                inner_train_df=inner_train_df,
                inner_val_df=inner_val_df,
                fold_idx=fold_idx,
            )

            if inner_split_strategy == "holdout":
                # OOD holdout: use the two chemically distinct subsets directly
                inner_train_cache = get_feature_cache_path(
                    task, dataset, fp_type, "test", train_inner_idx
                )
                inner_val_cache = get_feature_cache_path(
                    task, dataset, fp_type, "test", val_inner_idx
                )

                X_inner_train, valid_itr = compute_fingerprints(
                    inner_train_df["smiles"].tolist(), fp_type, inner_train_cache
                )
                X_inner_val, valid_ivl = compute_fingerprints(
                    inner_val_df["smiles"].tolist(), fp_type, inner_val_cache
                )
                assert valid_itr.all() and valid_ivl.all(), \
                    "Unexpected invalid SMILES in inner holdout subsets"

                y_inner_train = inner_train_df["value"].values
                y_inner_val = inner_val_df["value"].values

                # Cast to bool for KNN/Jaccard distance.
                if model_name.startswith("knn") and fp_type in ["ecfp4", "maccs", "rdkit_topo"]:
                    X_inner_train = X_inner_train.astype(bool)
                    X_inner_val = X_inner_val.astype(bool)

                extra_kwargs["inner_train_X"] = X_inner_train
                extra_kwargs["inner_train_y"] = y_inner_train
                extra_kwargs["inner_val_X"] = X_inner_val
                extra_kwargs["inner_val_y"] = y_inner_val

            elif inner_split_strategy == "random_shuffle":
                # Random shuffle, but matched to the OOD holdout:
                # same validation fraction as the OOD inner split;
                #   stratify by target AND original fold origin.
                n_ood_train = len(inner_train_df)
                n_ood_val = len(inner_val_df)
                random_val_fraction = n_ood_val / (n_ood_train + n_ood_val)

                fold_origin = _infer_fold_origin_labels(
                    train_df=train_df,
                    inner_train_df=inner_train_df,
                    inner_val_df=inner_val_df,
                )

                random_stratify_labels = _make_joint_stratify_labels(
                    y=train_df["value"].values,
                    fold_origin=fold_origin,
                )

                extra_kwargs["random_val_fraction"] = random_val_fraction
                extra_kwargs["random_stratify_labels"] = random_stratify_labels

                logger.info(
                    f"  Random shuffle matched to OOD split: "
                    f"n_ood_train={n_ood_train}, n_ood_val={n_ood_val}, "
                    f"val_fraction={random_val_fraction:.4f}"
                )

        # ---- Execute this outer fold ----
        result = run_single_fold(
            train_df=train_df,
            test_df=test_df,
            fold_idx=fold_idx,
            task=task,
            dataset=dataset,
            fp_type=fp_type,
            model_name=model_name,
            estimator_factory=estimator_factory,
            param_grid=param_grid,
            inner_k=inner_k,
            scoring=scoring,
            search_strategy=search_strategy,
            n_iter=n_iter,
            random_state=random_state,
            save_results=save_results,
            **extra_kwargs,
        )
        fold_results.append(result)

    # ---- Aggregate test metrics across folds ----
    test_metrics_list = [r["test_metrics"] for r in fold_results]
    aggregated = aggregate_fold_metrics(test_metrics_list)

    logger.info(f"\n{'='*60}")
    logger.info("AGGREGATED TEST METRICS:")
    for k, v in aggregated.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"{'='*60}")

    # ---- Check hyperparameter stability across folds ----
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