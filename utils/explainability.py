"""
Explainability utilities for the Lo-Hi protocol comparison.

Provides helpers for:
- Extracting base models from sklearn Pipelines
- Computing local feature contributions for linear models
- Building feature overlap and stability tables
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Set
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


def extract_base_model(model: BaseEstimator) -> BaseEstimator:
    """If the model is a sklearn Pipeline, return the final estimator."""
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def transform_features_if_pipeline(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """
    If model is a Pipeline, apply all preprocessing steps (e.g. StandardScaler)
    before the final estimator. Needed for RDKit descriptors.
    """
    if not isinstance(model, Pipeline):
        return X
    Xt = X.copy()
    for step_name, step in model.steps[:-1]:
        Xt = step.transform(Xt)
    return Xt


def compute_linear_contributions(
    model: BaseEstimator,
    X_row: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute local contributions x_j * w_j for one molecule.

    Returns (transformed_features, coefficients, contributions) or (None, None, None).
    """
    base = extract_base_model(model)
    X_transformed = transform_features_if_pipeline(model, X_row)

    if not hasattr(base, "coef_"):
        return None, None, None

    coef = np.asarray(base.coef_).ravel()
    x = np.asarray(X_transformed).ravel()
    return x, coef, x * coef


def compute_topk_overlap(set_a: Set[int], set_b: Set[int], k: int) -> float:
    """Compute overlap@k = |A ∩ B| / k."""
    return len(set_a.intersection(set_b)) / k if k > 0 else 0.0