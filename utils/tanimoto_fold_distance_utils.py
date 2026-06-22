"""
Utilities for model-wise fold-to-fold Tanimoto distance analysis.

Put this file under:

    utils/tanimoto_fold_distance_utils.py

Main design choices:
- Complete pairwise Tanimoto distance is the main structural distance.
- Restricted-space distances are conditional on non-zero restricted fingerprints.
- Valid molecule/pair fractions are saved and must be used when interpreting
  restricted-space results.
- List A uses tree_importance for Decision Trees and coefficient-based
  importance for LR/SVM.
- List B uses the fold-discriminator importance table.
- Random bits are kept as the dimensionality-control baseline, and the summary
  reports coverage diagnostics plus z-scores against the random-bit repeats.
- Wasserstein is optional and should be treated as a secondary same-k diagnostic.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
import zlib
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.stats import wasserstein_distance_nd

    HAS_WASSERSTEIN = True
except Exception:
    wasserstein_distance_nd = None
    HAS_WASSERSTEIN = False


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass
class TanimotoDistanceConfig:
    task: str = "hi"

    datasets_main: list[str] = field(default_factory=lambda: ["drd2", "hiv", "sol"])
    dataset_labels: dict[str, str] = field(
        default_factory=lambda: {"drd2": "DRD2", "hiv": "HIV", "sol": "Sol"}
    )

    subset_files: dict[str, str] = field(
        default_factory=lambda: {
            "F1": "test_3.csv",
            "F2": "test_2.csv",
            "F3": "test_1.csv",
        }
    )

    pair_to_outer_fold: dict[str, int] = field(
        default_factory=lambda: {
            "F1_vs_F2": 1,
            "F1_vs_F3": 2,
            "F2_vs_F3": 3,
        }
    )

    fp_type: str = "ecfp4"
    expected_ecfp4_bits: int = 2048

    k_values: list[int] = field(
        default_factory=lambda: [10, 20, 50, 100, 150, 200, 250, 500]
    )

    # Distances are computed separately for each model family used in OOD-vs-random.
    models: list[str] = field(default_factory=lambda: ["DT", "LR", "SVM"])

    pairwise_chunk_size: int = 512
    n_random_bit_repeats: int = 30

    verbose_pairwise: bool = False
    print_every_random_repeat: int = 5

    # Wasserstein is much more expensive than pairwise Tanimoto.
    # It is computed only for a few representative k values and only for a few
    # random-bit repeats. Pairwise Tanimoto is still computed for all k and all
    # random-bit repeats.
    run_wasserstein: bool = True
    wasserstein_subsample: int = 200
    wasserstein_k_values: list[int] = field(default_factory=lambda: [50, 100, 200])
    wasserstein_valid_only: bool = True
    compute_wasserstein_random_bits: bool = True
    wasserstein_random_bit_repeats: int = 5

    # Coverage threshold used to mark rows as interpretable for main plots.
    coverage_threshold: float = 0.90

    random_state: int = 42

    project_root: Path | None = None

    @property
    def datasets(self) -> list[str]:
        return self.datasets_main

    @property
    def pairs(self) -> list[tuple[str, str]]:
        return list(combinations(["F1", "F2", "F3"], 2))

    @property
    def data_root(self) -> Path:
        return self.project_root / "data" / self.task

    @property
    def ood_cross_dir(self) -> Path:
        return (
            self.project_root
            / "results"
            / "results_ood_vs_random_shuffle"
            / self.task
            / "cross_dataset"
        )

    @property
    def out_root(self) -> Path:
        return (
            self.project_root / "results" / "results_fold_distance_tanimoto" / self.task
        )

    @property
    def fig_root(self) -> Path:
        return self.out_root / "figures"

    @property
    def cache_root(self) -> Path:
        return self.project_root / "features" / "fold_distance_tanimoto" / self.task

    def ensure_paths(self) -> None:
        if self.project_root is None:
            self.project_root = find_project_root()

        for d in (self.out_root, self.fig_root, self.cache_root):
            d.mkdir(parents=True, exist_ok=True)


MODEL_NAME_MAP = {
    "Decision Tree": "DT",
    "Logistic Regression": "LR",
    "Linear SVM": "SVM",
    "dt": "DT",
    "decision_tree": "DT",
    "lr": "LR",
    "logistic_regression": "LR",
    "svm": "SVM",
    "svm_linear": "SVM",
    "DT": "DT",
    "LR": "LR",
    "SVM": "SVM",
}

FP_MAP = {
    "ECFP4": "ecfp4",
    "MACCS": "maccs",
    "RDKit desc": "rdkit_desc",
    "ecfp4": "ecfp4",
    "maccs": "maccs",
    "rdkit_desc": "rdkit_desc",
}

IMPORTANCE_COL_CANDIDATES = [
    "importance_value",
    "permutation_importance_mean",
    "abs_weight",
    "normalized_abs_importance",
    "tree_importance",
]

DIST_ROW_KEYS = [
    "dataset",
    "dataset_label",
    "pair",
    "space",
    "k",
    "model",
    "bit_source",
    "activity_protocol",
    "activity_fold",
    "bit_repeat",
    "bits_used",
    "nn_sym_distance",
    "nn_A_to_B_mean",
    "nn_B_to_A_mean",
    "wasserstein_nd",
    "wasserstein_nd_normalized",
    "valid_molecule_fraction",
    "n_valid_molecules",
    "n_total_molecules",
    "pairwise_distance",
    "valid_pair_fraction",
    "n_valid_pairs",
    "n_total_pairs",
    "pairwise_mode",
    "outer_fold",
]


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------


def find_project_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path.cwd().resolve()

    current = start
    while current != current.parent:
        if all((current / d).exists() for d in ["data", "utils", "results"]):
            return current
        current = current.parent

    raise RuntimeError(
        "Could not find project root containing data/, utils/ and results/."
    )


def ensure_project_on_path(project_root: Path) -> None:
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))


def stable_seed(*parts: Any, base: int = 42) -> int:
    """Deterministic, cross-platform seed from arbitrary tuple of parts."""
    s = "|".join(str(p) for p in parts) + f"|{base}"
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def local_rng(*parts: Any, base: int = 42) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*parts, base=base))


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def get_smiles_col(df: pd.DataFrame) -> str:
    for col in ["smiles", "SMILES", "canonical_smiles"]:
        if col in df.columns:
            return col

    raise ValueError(f"No SMILES column found. Columns: {df.columns.tolist()}")


def normalize_model_name(x: Any) -> str:
    s = str(x).strip()
    return MODEL_NAME_MAP.get(s, s)


def normalize_fingerprint_name(x: Any) -> str:
    s = str(x).strip()
    return FP_MAP.get(s, s)


def protocol_match(x: Any) -> str:
    s = str(x).strip().lower()

    if s in {"ood holdout", "ood_holdout", "holdout_ood", "ood"}:
        return "ood"

    if s in {"random shuffle", "random_shuffle", "random"}:
        return "random"

    if "ood" in s and "random" not in s:
        return "ood"

    if "random" in s:
        return "random"

    return s


def find_importance_col(df: pd.DataFrame) -> str:
    if "importance_value" in df.columns:
        return "importance_value"

    for c in IMPORTANCE_COL_CANDIDATES:
        if c in df.columns:
            return c

    raise ValueError(
        f"No importance column found among {IMPORTANCE_COL_CANDIDATES}. "
        f"Columns: {df.columns.tolist()}"
    )


def _safe_numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)

    return pd.to_numeric(df[col], errors="coerce").fillna(default)


# ---------------------------------------------------------------------
# Loading fingerprints and feature lists
# ---------------------------------------------------------------------


def load_subset_fps(
    dataset: str,
    cfg: TanimotoDistanceConfig,
) -> dict[str, tuple[np.ndarray, pd.DataFrame]]:
    """Read F1/F2/F3 SMILES, compute ECFP4 with cache, drop invalid SMILES."""
    cfg.ensure_paths()
    ensure_project_on_path(cfg.project_root)

    from utils.fingerprints import compute_fingerprints

    print_section(f"Loading subsets: {dataset.upper()}-{cfg.task}")

    data_dir = cfg.data_root / dataset
    cache_dir = cfg.cache_root / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}

    for subset, fname in cfg.subset_files.items():
        df = pd.read_csv(data_dir / fname).copy()
        smi_col = get_smiles_col(df)
        smiles = df[smi_col].astype(str).tolist()
        raw_n = len(smiles)

        t0 = time.time()
        X, valid_mask = compute_fingerprints(
            smiles_list=smiles,
            fp_type=cfg.fp_type,
            cache_path=str(cache_dir / f"{subset}_{cfg.fp_type}.npz"),
        )
        elapsed = time.time() - t0

        valid_mask = np.asarray(valid_mask, dtype=bool)

        if len(valid_mask) != raw_n:
            raise ValueError(
                f"valid_mask length mismatch: {len(valid_mask)} vs {raw_n}"
            )

        df_valid = df.loc[valid_mask].reset_index(drop=True)

        X = np.asarray(X, dtype=np.uint8)
        n_valid = X.shape[0]

        if n_valid != len(df_valid):
            raise ValueError(
                f"X/df mismatch after valid_mask: {n_valid} vs {len(df_valid)}"
            )

        if X.shape[1] != cfg.expected_ecfp4_bits:
            raise ValueError(
                f"Expected {cfg.expected_ecfp4_bits} bits, got {X.shape[1]} for {subset}"
            )

        invalid_n = raw_n - n_valid

        print(
            f"  {subset} ({fname}): raw={raw_n}, valid={n_valid}, "
            f"invalid={invalid_n}, n_bits={X.shape[1]}, time={elapsed:.1f}s"
        )

        out[subset] = (X, df_valid)

    return out


def load_list_a(cfg: TanimotoDistanceConfig) -> pd.DataFrame:
    """Load List A activity feature importance, normalised and ECFP4-only."""
    cfg.ensure_paths()

    print_section("Loading List A")

    # Main diagnostic version:
    #   DT     -> tree_importance
    #   LR/SVM -> coefficient-based importance
    #
    # Fallback is kept for backward compatibility, but we still force DT to use
    # tree_importance if the column exists.
    preferred_path = cfg.ood_cross_dir / "cross_dataset_feature_importance_all_tree.csv"
    fallback_path = cfg.ood_cross_dir / "cross_dataset_feature_importance_all.csv"

    if preferred_path.exists():
        path = preferred_path
    elif fallback_path.exists():
        path = fallback_path
        print(
            f"WARNING: {preferred_path.name} not found. "
            f"Falling back to {fallback_path.name}."
        )
    else:
        raise FileNotFoundError(
            f"Missing List A file. Tried:\n"
            f"  {preferred_path}\n"
            f"  {fallback_path}"
        )

    print(f"Using List A file: {path.name}")

    df = pd.read_csv(path, low_memory=False).copy()
    print(f"Raw List A: {df.shape}")

    df["dataset"] = df["dataset"].astype(str).str.lower()
    df = df[df["dataset"].isin(cfg.datasets_main)].copy()

    df["model"] = df["model"].map(normalize_model_name).astype(str)
    df["fingerprint"] = df["fingerprint"].map(normalize_fingerprint_name).astype(str)
    df = df[df["fingerprint"] == "ecfp4"].copy()

    if "feature_idx" not in df.columns:
        raise ValueError("List A is missing feature_idx.")

    if "fold" not in df.columns:
        raise ValueError("List A is missing fold.")

    if "protocol" not in df.columns:
        raise ValueError("List A is missing protocol.")

    df["feature_idx"] = df["feature_idx"].astype(int)
    df["fold"] = df["fold"].astype(int)
    df["protocol_norm"] = df["protocol"].map(protocol_match)

    # Force the correct main importance source.
    df["importance_value_numeric"] = 0.0
    df["importance_source_selected"] = "unknown"

    dt_mask = df["model"].eq("DT")
    linear_mask = df["model"].isin(["LR", "SVM"])

    if dt_mask.any():
        if "tree_importance" not in df.columns:
            raise ValueError(
                "List A contains DT rows but no tree_importance column. "
                "Regenerate cross_dataset_feature_importance_all_tree.csv first."
            )

        df.loc[dt_mask, "importance_value_numeric"] = (
            _safe_numeric_series(df.loc[dt_mask], "tree_importance", default=0.0)
            .clip(lower=0.0)
            .to_numpy()
        )
        df.loc[dt_mask, "importance_source_selected"] = "tree_importance"

    if linear_mask.any():
        if "normalized_abs_importance" in df.columns:
            df.loc[linear_mask, "importance_value_numeric"] = (
                _safe_numeric_series(
                    df.loc[linear_mask], "normalized_abs_importance", default=0.0
                )
                .abs()
                .to_numpy()
            )
            df.loc[linear_mask, "importance_source_selected"] = (
                "normalized_abs_importance"
            )

        elif "abs_weight" in df.columns:
            df.loc[linear_mask, "importance_value_numeric"] = (
                _safe_numeric_series(df.loc[linear_mask], "abs_weight", default=0.0)
                .abs()
                .to_numpy()
            )
            df.loc[linear_mask, "importance_source_selected"] = "abs_weight"

        elif "importance_value" in df.columns:
            df.loc[linear_mask, "importance_value_numeric"] = (
                _safe_numeric_series(
                    df.loc[linear_mask], "importance_value", default=0.0
                )
                .abs()
                .to_numpy()
            )
            df.loc[linear_mask, "importance_source_selected"] = "importance_value"

        else:
            raise ValueError(
                "List A contains LR/SVM rows but no coefficient-based importance column."
            )

    df["abs_importance"] = df["importance_value_numeric"].abs()

    print(f"Filtered List A: {df.shape}")
    print(f"Datasets : {sorted(df['dataset'].unique())}")
    print(f"Models   : {sorted(df['model'].unique())}")
    print(f"Protocols: {sorted(df['protocol_norm'].unique())}")
    print(f"Folds    : {sorted(df['fold'].unique())}")

    print("\nSelected importance sources:")
    print(
        df.groupby(["model", "importance_source_selected"])
        .size()
        .rename("n_rows")
        .reset_index()
        .to_string(index=False)
    )

    if "importance_source" in df.columns:
        print("\nOriginal importance_source column:")
        print(
            df.groupby(["model", "importance_source"])
            .size()
            .rename("n_rows")
            .reset_index()
            .to_string(index=False)
        )

    print("\nRows by dataset/model:")
    print(
        df.groupby(["dataset", "model"])
        .size()
        .rename("n_rows")
        .reset_index()
        .to_string(index=False)
    )

    return df


def load_list_b(cfg: TanimotoDistanceConfig) -> pd.DataFrame:
    """
    Load List B: fold-discriminator feature importance.

    List B contains the ECFP4 features used by the dataset/fold detection task,
    i.e. the classifiers trained to distinguish Lo-Hi fold pairs such as
    F1_vs_F2, F1_vs_F3 and F2_vs_F3.
    """
    cfg.ensure_paths()

    print_section("Loading List B")

    preferred_path = (
        cfg.project_root
        / "results"
        / "results_classifier_shift_test"
        / cfg.task
        / "cross_dataset_listB_same_search_cv_feature_importance.csv"
    )

    if preferred_path.exists():
        path = preferred_path
    else:
        candidates = list(
            cfg.project_root.glob(
                "results/**/cross_dataset_listB_same_search_cv_feature_importance.csv"
            )
        )

        if not candidates:
            raise FileNotFoundError(
                "Could not find cross_dataset_listB_same_search_cv_feature_importance.csv "
                "under results/**. Run the distribution-shift notebook first and check the saved filename."
            )

        path = sorted(candidates)[0]

    print(f"Using List B file: {path}")

    assert (
        path.name == "cross_dataset_listB_same_search_cv_feature_importance.csv"
    ), f"Wrong List B file selected: {path}"

    df = pd.read_csv(path, low_memory=False).copy()
    print(f"Raw List B: {df.shape}")

    df["dataset"] = df["dataset"].astype(str).str.lower()
    df = df[df["dataset"].isin(cfg.datasets_main)].copy()

    df["model"] = df["model"].map(normalize_model_name).astype(str)

    if "fingerprint" in df.columns:
        df["fingerprint"] = (
            df["fingerprint"].map(normalize_fingerprint_name).astype(str)
        )
    elif "fp_type" in df.columns:
        df["fingerprint"] = df["fp_type"].map(normalize_fingerprint_name).astype(str)
    else:
        raise ValueError(
            f"No fingerprint/fp_type column found in List B. "
            f"Columns: {df.columns.tolist()}"
        )

    df = df[df["fingerprint"] == "ecfp4"].copy()

    if "pair" not in df.columns:
        if {"fold_a", "fold_b"}.issubset(df.columns):
            df["pair"] = df["fold_a"].astype(str) + "_vs_" + df["fold_b"].astype(str)
        elif {"subset_a", "subset_b"}.issubset(df.columns):
            df["pair"] = (
                df["subset_a"].astype(str) + "_vs_" + df["subset_b"].astype(str)
            )
        else:
            raise ValueError(
                "List B has no pair column and no fold_a/fold_b columns. "
                f"Columns: {df.columns.tolist()}"
            )

    df["pair"] = df["pair"].astype(str)

    # List B is a fold-discriminator ranking.
    # List B is a fold-discriminator ranking.
    # For DT, the main ranking uses tree_importance, consistently with List A.
    # Permutation importance, if present in the CSV, is kept only as a diagnostic.
    # For LR/SVM, use coefficient-based non-negative importance.
    df["importance_value_numeric"] = 0.0
    df["importance_source_selected"] = "unknown"

    dt_mask = df["model"].eq("DT")
    linear_mask = df["model"].isin(["LR", "SVM"])

    if dt_mask.any():
        if "tree_importance" not in df.columns:
            raise ValueError(
                "List B contains DT rows but no tree_importance column. "
                "Regenerate cross_dataset_listB_same_search_cv_feature_importance.csv "
                "with DT tree_importance as the main ranking signal."
            )

        df.loc[dt_mask, "importance_value_numeric"] = (
            _safe_numeric_series(df.loc[dt_mask], "tree_importance", default=0.0)
            .clip(lower=0.0)
            .to_numpy()
        )
        df.loc[dt_mask, "importance_source_selected"] = "tree_importance"

    if linear_mask.any():
        if "normalized_importance" in df.columns:
            df.loc[linear_mask, "importance_value_numeric"] = (
                _safe_numeric_series(
                    df.loc[linear_mask],
                    "normalized_importance",
                    default=0.0,
                )
                .clip(lower=0.0)
                .to_numpy()
            )
            df.loc[linear_mask, "importance_source_selected"] = "normalized_importance"
        elif "abs_importance" in df.columns:
            df.loc[linear_mask, "importance_value_numeric"] = (
                _safe_numeric_series(
                    df.loc[linear_mask],
                    "abs_importance",
                    default=0.0,
                )
                .clip(lower=0.0)
                .to_numpy()
            )
            df.loc[linear_mask, "importance_source_selected"] = "abs_importance"
        elif "coefficient" in df.columns:
            df.loc[linear_mask, "importance_value_numeric"] = (
                _safe_numeric_series(df.loc[linear_mask], "coefficient", default=0.0)
                .abs()
                .to_numpy()
            )
            df.loc[linear_mask, "importance_source_selected"] = "abs_coefficient"
        elif "importance_value" in df.columns:
            df.loc[linear_mask, "importance_value_numeric"] = (
                _safe_numeric_series(
                    df.loc[linear_mask],
                    "importance_value",
                    default=0.0,
                )
                .abs()
                .to_numpy()
            )
            df.loc[linear_mask, "importance_source_selected"] = "abs_importance_value"
        elif "importance" in df.columns:
            df.loc[linear_mask, "importance_value_numeric"] = (
                _safe_numeric_series(df.loc[linear_mask], "importance", default=0.0)
                .abs()
                .to_numpy()
            )
            df.loc[linear_mask, "importance_source_selected"] = "abs_importance"
        else:
            raise ValueError(
                "List B contains LR/SVM rows but no coefficient-based importance column."
            )

    df["abs_importance"] = df["importance_value_numeric"].clip(lower=0.0)
    df["feature_idx"] = df["feature_idx"].astype(int)

    print(f"Filtered List B: {df.shape}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Models:   {sorted(df['model'].unique())}")
    print(f"Pairs:    {sorted(df['pair'].unique())}")
    print("Rows by dataset/model/pair:")
    print(
        df.groupby(["dataset", "model", "pair"])
        .size()
        .rename("n_rows")
        .reset_index()
        .to_string(index=False)
    )

    print("\nSelected List B importance sources:")
    print(
        df.groupby(["model", "importance_source_selected"])
        .size()
        .rename("n_rows")
        .reset_index()
        .to_string(index=False)
    )

    return df


# ---------------------------------------------------------------------
# Tanimoto distances
# ---------------------------------------------------------------------


def nn_max_sim(X: np.ndarray, Y: np.ndarray, chunk: int = 512) -> np.ndarray:
    """For each row in X, return max Tanimoto similarity to any row in Y."""
    X = X.astype(np.float32, copy=False)
    Y = Y.astype(np.float32, copy=False)

    sx = X.sum(axis=1)
    sy = Y.sum(axis=1)

    n = X.shape[0]
    out = np.zeros(n, dtype=np.float32)

    for i in range(0, n, chunk):
        block = X[i : i + chunk]
        sb = sx[i : i + chunk]

        dots = block @ Y.T
        denom = sb[:, None] + sy[None, :] - dots

        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(denom > 0, dots / denom, 0.0)

        out[i : i + chunk] = sim.max(axis=1)

    return out


def nn_distances(XA: np.ndarray, XB: np.ndarray) -> dict[str, Any]:
    """Symmetric nearest-neighbour Tanimoto distance, restricted-space aware."""
    sa = XA.sum(axis=1)
    sb = XB.sum(axis=1)

    valid_a = sa > 0
    valid_b = sb > 0

    n_total_a = int(XA.shape[0])
    n_total_b = int(XB.shape[0])

    XA_v = XA[valid_a]
    XB_v = XB[valid_b]

    if XA_v.shape[0] == 0 or XB_v.shape[0] == 0:
        return {
            "nn_sym_distance": np.nan,
            "nn_A_to_B_mean": np.nan,
            "nn_B_to_A_mean": np.nan,
            "nn_AB_array": np.array([], dtype=np.float32),
            "nn_BA_array": np.array([], dtype=np.float32),
            "n_valid_a": int(valid_a.sum()),
            "n_valid_b": int(valid_b.sum()),
            "n_total_a": n_total_a,
            "n_total_b": n_total_b,
        }

    sim_ab = nn_max_sim(XA_v, XB_v)
    sim_ba = nn_max_sim(XB_v, XA_v)

    d_ab = 1.0 - sim_ab
    d_ba = 1.0 - sim_ba

    return {
        "nn_sym_distance": float(0.5 * (d_ab.mean() + d_ba.mean())),
        "nn_A_to_B_mean": float(d_ab.mean()),
        "nn_B_to_A_mean": float(d_ba.mean()),
        "nn_AB_array": d_ab,
        "nn_BA_array": d_ba,
        "n_valid_a": int(valid_a.sum()),
        "n_valid_b": int(valid_b.sum()),
        "n_total_a": n_total_a,
        "n_total_b": n_total_b,
    }


def complete_pairwise_distance(
    XA: np.ndarray,
    XB: np.ndarray,
    dataset: str,
    pair: str,
    space: str,
    chunk: int = 512,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Complete cross-fold pairwise Tanimoto distance.

    Computes the mean Tanimoto distance over all valid pairs.
    Pairs with union=0 are excluded, and valid_pair_fraction is reported.
    """
    XA = XA.astype(np.float32, copy=False)
    XB = XB.astype(np.float32, copy=False)

    sum_dist = 0.0
    n_valid = 0
    n_total = int(XA.shape[0] * XB.shape[0])

    if verbose:
        print(
            f"      pairwise {dataset} | {pair} | {space}: "
            f"{XA.shape[0]} x {XB.shape[0]} = {n_total:,} pairs"
        )

    y_sum = XB.sum(axis=1)

    for start in range(0, XA.shape[0], chunk):
        xb = XA[start : start + chunk]
        x_sum = xb.sum(axis=1)

        inter = xb @ XB.T
        union = x_sum[:, None] + y_sum[None, :] - inter

        valid = union > 0

        sim = np.zeros_like(inter, dtype=np.float32)
        np.divide(inter, union, out=sim, where=valid)

        dist = 1.0 - sim

        sum_dist += float(dist[valid].sum())
        n_valid += int(valid.sum())

    mean_dist = sum_dist / n_valid if n_valid > 0 else np.nan

    if verbose:
        print(
            f"        done: mean={mean_dist:.4f}, "
            f"valid_pairs={n_valid:,}/{n_total:,} "
            f"({n_valid / n_total:.3f})"
        )

    return {
        "pairwise_distance": mean_dist,
        "n_valid_pairs": n_valid,
        "n_total_pairs": n_total,
        "valid_pair_fraction": n_valid / n_total if n_total > 0 else np.nan,
        "pairwise_mode": "complete",
    }


# ---------------------------------------------------------------------
# Bit selection
# ---------------------------------------------------------------------


def _positive_importance(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only positive-importance features to avoid arbitrary zero-importance top-k bits."""
    out = df.copy()
    out["abs_importance"] = pd.to_numeric(out["abs_importance"], errors="coerce")
    out = out[out["abs_importance"] > 0].copy()
    return out


def topk_bits_global(
    list_a: pd.DataFrame,
    dataset: str,
    model: str,
    k: int,
) -> np.ndarray:
    """Top-k ECFP4 bits pooled across protocols and folds for one (dataset, model)."""
    sub = list_a[(list_a["dataset"] == dataset) & (list_a["model"] == model)]
    sub = _positive_importance(sub)

    if sub.empty:
        return np.array([], dtype=int)

    agg = (
        sub.groupby("feature_idx", as_index=False)["abs_importance"]
        .mean()
        .sort_values(["abs_importance", "feature_idx"], ascending=[False, True])
    )

    return agg.head(k)["feature_idx"].to_numpy(dtype=int)


def topk_bits_fold_protocol(
    list_a: pd.DataFrame,
    dataset: str,
    model: str,
    fold: int,
    protocol: str,
    k: int,
) -> np.ndarray:
    """Top-k ECFP4 bits for (dataset, model, fold, protocol) — fold/protocol-aware."""
    protocol = protocol_match(protocol)

    sub = list_a[
        (list_a["dataset"] == dataset)
        & (list_a["model"] == model)
        & (list_a["fold"].astype(int) == int(fold))
        & (list_a["protocol_norm"] == protocol)
    ]

    sub = _positive_importance(sub)

    if sub.empty:
        return np.array([], dtype=int)

    agg = (
        sub.groupby("feature_idx", as_index=False)["abs_importance"]
        .mean()
        .sort_values(["abs_importance", "feature_idx"], ascending=[False, True])
    )

    return agg.head(k)["feature_idx"].to_numpy(dtype=int)


def random_topk_bits(
    dataset: str,
    pair: str,
    k: int,
    repeat: int,
    cfg: TanimotoDistanceConfig,
) -> np.ndarray:
    """Random subset of k ECFP4 bits, deterministic by (dataset, pair, k, repeat)."""
    rng = local_rng("random_bits", dataset, pair, k, repeat, base=cfg.random_state)

    k_eff = min(int(k), cfg.expected_ecfp4_bits)

    return rng.choice(cfg.expected_ecfp4_bits, size=k_eff, replace=False)


def topk_bits_list_b(
    list_b: pd.DataFrame,
    dataset: str,
    model: str,
    pair: str,
    k: int,
) -> np.ndarray:
    """
    Top-k ECFP4 bits from List B for one dataset, model and fold pair.

    These bits come from the dataset/fold detection task, not from the
    activity-prediction task.
    """
    sub = list_b[
        (list_b["dataset"] == dataset)
        & (list_b["model"] == model)
        & (list_b["pair"] == pair)
    ]

    sub = _positive_importance(sub)

    if sub.empty:
        warnings.warn(
            f"No positive List B bits found for dataset={dataset}, model={model}, pair={pair}."
        )
        return np.array([], dtype=int)

    agg = (
        sub.groupby("feature_idx", as_index=False)["abs_importance"]
        .mean()
        .sort_values(["abs_importance", "feature_idx"], ascending=[False, True])
    )

    return agg.head(k)["feature_idx"].to_numpy(dtype=int)


def build_selected_bits_table(
    list_a: pd.DataFrame,
    list_b: pd.DataFrame,
    cfg: TanimotoDistanceConfig,
) -> pd.DataFrame:
    """
    Save the selected top-k ECFP4 bit sets used in the restricted-space analysis.
    """
    cfg.ensure_paths()

    rows = []

    for dataset in cfg.datasets:
        for model in cfg.models:
            for fold_a, fold_b in cfg.pairs:
                pair = f"{fold_a}_vs_{fold_b}"
                outer_fold = cfg.pair_to_outer_fold[pair]

                for k in cfg.k_values:
                    bit_sets = {
                        "activity_global": (
                            topk_bits_global(list_a, dataset, model, k),
                            "pooled",
                            np.nan,
                        ),
                        "activity_ood": (
                            topk_bits_fold_protocol(
                                list_a, dataset, model, outer_fold, "ood", k
                            ),
                            "ood",
                            outer_fold,
                        ),
                        "activity_random_shuffle": (
                            topk_bits_fold_protocol(
                                list_a, dataset, model, outer_fold, "random", k
                            ),
                            "random",
                            outer_fold,
                        ),
                        "dataset_detection": (
                            topk_bits_list_b(list_b, dataset, model, pair, k),
                            "same_search_cv",
                            outer_fold,
                        ),
                    }

                    for bit_source, (bits, protocol, activity_fold) in bit_sets.items():
                        bits = [int(b) for b in bits]

                        rows.append(
                            {
                                "dataset": dataset,
                                "dataset_label": cfg.dataset_labels.get(
                                    dataset, dataset
                                ),
                                "model": model,
                                "pair": pair,
                                "outer_fold": outer_fold,
                                "k": k,
                                "bit_source": bit_source,
                                "activity_protocol": protocol,
                                "activity_fold": activity_fold,
                                "n_bits": len(bits),
                                "bits_json": json.dumps(bits),
                            }
                        )

    out = pd.DataFrame(rows)

    out_path = cfg.out_root / "fold_distance_selected_bits_modelwise.csv"
    out.to_csv(out_path, index=False)

    print(f"Saved selected bits table: {out_path} ({len(out)} rows)")

    fewer_than_k = out[out["n_bits"] < out["k"]].copy()

    if len(fewer_than_k) > 0:
        print(
            "\nWARNING: some selected bit sets contain fewer than k positive-importance bits."
        )
        print(
            fewer_than_k[["dataset", "model", "pair", "k", "bit_source", "n_bits"]]
            .head(20)
            .to_string(index=False)
        )
    else:
        print("OK: all selected bit sets contain k positive-importance bits.")

    return out


# ---------------------------------------------------------------------
# Wasserstein
# ---------------------------------------------------------------------


def restrict_to_bits(X: np.ndarray, bits: np.ndarray | list[int]) -> np.ndarray:
    if len(bits) == 0:
        return np.zeros((X.shape[0], 0), dtype=X.dtype)

    return X[:, bits]


def should_compute_wasserstein(
    bit_source: str,
    k: int,
    bit_repeat: Any,
    cfg: TanimotoDistanceConfig,
) -> bool:
    """Small guard to avoid computing Wasserstein for every single random repeat."""
    if not cfg.run_wasserstein or not HAS_WASSERSTEIN:
        return False

    bit_source = str(bit_source)

    if bit_source == "full_ecfp4":
        return True

    if int(k) not in set(cfg.wasserstein_k_values):
        return False

    if bit_source == "random_bits":
        if not cfg.compute_wasserstein_random_bits:
            return False

        if pd.isna(bit_repeat):
            return False

        return int(bit_repeat) < int(cfg.wasserstein_random_bit_repeats)

    return bit_source in {
        "activity_global",
        "activity_ood",
        "activity_random_shuffle",
        "dataset_detection",
    }


def wasserstein_nd_optional(
    XA: np.ndarray,
    XB: np.ndarray,
    n: int,
    dataset: str,
    pair: str,
    space: str,
    k: int,
    model: str,
    bit_source: str,
    bit_repeat: Any,
    bits_used: int,
    cfg: TanimotoDistanceConfig,
) -> tuple[float, float]:
    """
    Optional ND Wasserstein-1 distance with Euclidean ground metric.

    It is computed only for a subset of rows because it is much more expensive
    than complete pairwise Tanimoto. For restricted spaces, all-zero restricted
    fingerprints are removed before computing Wasserstein, consistently with
    the valid-molecule diagnostics used for Tanimoto.
    """
    if not should_compute_wasserstein(bit_source, k, bit_repeat, cfg):
        return np.nan, np.nan

    if wasserstein_distance_nd is None:
        return np.nan, np.nan

    if XA.shape[0] == 0 or XB.shape[0] == 0:
        return np.nan, np.nan

    XA_w = XA
    XB_w = XB

    if cfg.wasserstein_valid_only:
        valid_a = XA_w.sum(axis=1) > 0
        valid_b = XB_w.sum(axis=1) > 0

        XA_w = XA_w[valid_a]
        XB_w = XB_w[valid_b]

    if XA_w.shape[0] == 0 or XB_w.shape[0] == 0:
        return np.nan, np.nan

    rng = local_rng(
        "wasserstein",
        dataset,
        pair,
        model,
        space,
        k,
        bit_source,
        bit_repeat,
        base=cfg.random_state,
    )

    if XA_w.shape[0] > n:
        XA_w = XA_w[rng.choice(XA_w.shape[0], size=n, replace=False)]

    if XB_w.shape[0] > n:
        XB_w = XB_w[rng.choice(XB_w.shape[0], size=n, replace=False)]

    try:
        wd = float(
            wasserstein_distance_nd(
                XA_w.astype(np.float32),
                XB_w.astype(np.float32),
            )
        )

        wd_norm = wd / np.sqrt(max(int(bits_used), 1))

        return wd, wd_norm

    except Exception as exc:
        warnings.warn(f"Wasserstein failed for {dataset}|{pair}|{model}|{space}: {exc}")
        return np.nan, np.nan


def pre_run_sanity_checks(
    list_a: pd.DataFrame,
    list_b: pd.DataFrame,
    cfg: TanimotoDistanceConfig,
) -> None:
    print_section("Pre-run sanity checks")

    # 1. Check List A: DT must be tree-based in the main run.
    dt_a = list_a[list_a["model"] == "DT"].copy()

    if len(dt_a) == 0:
        raise RuntimeError("No DT rows found in List A.")

    print("DT List A selected importance source:")
    print(dt_a["importance_source_selected"].value_counts(dropna=False))

    assert set(dt_a["importance_source_selected"].dropna().unique()) == {
        "tree_importance"
    }, (
        "DT List A is not using tree_importance. "
        "Check that cross_dataset_feature_importance_all_tree.csv exists and was loaded."
    )

    assert (
        "tree_importance" in dt_a.columns
    ), "Missing tree_importance column for DT List A."

    assert np.allclose(
        dt_a["importance_value_numeric"].to_numpy(),
        pd.to_numeric(dt_a["tree_importance"], errors="coerce").fillna(0.0).to_numpy(),
        equal_nan=True,
    ), (
        "DT importance_value_numeric does not match tree_importance. "
        "You are probably still using the permutation-based List A."
    )

    print("OK: DT List A is tree_importance-based.")
    
        # 2. Check List B: DT must also be tree-based in the main run.
    dt_b = list_b[list_b["model"] == "DT"].copy()

    if len(dt_b) == 0:
        raise RuntimeError("No DT rows found in List B.")

    print("\nDT List B selected importance source:")
    print(dt_b["importance_source_selected"].value_counts(dropna=False))

    assert set(dt_b["importance_source_selected"].dropna().unique()) == {
        "tree_importance"
    }, (
        "DT List B is not using tree_importance. "
        "Check that cross_dataset_listB_same_search_cv_feature_importance.csv "
        "was regenerated after updating the List B extraction cell."
    )

    assert (
        "tree_importance" in dt_b.columns
    ), "Missing tree_importance column for DT List B."

    assert np.allclose(
        dt_b["importance_value_numeric"].to_numpy(),
        pd.to_numeric(dt_b["tree_importance"], errors="coerce").fillna(0.0).to_numpy(),
        equal_nan=True,
    ), (
        "DT List B importance_value_numeric does not match tree_importance. "
        "You are probably still using the permutation-based List B."
    )

    print("OK: DT List B is tree_importance-based.")

    # 2. Check Wasserstein logic before expensive computation.
    print("RUN_WASSERSTEIN:", cfg.run_wasserstein)
    print("HAS_WASSERSTEIN:", HAS_WASSERSTEIN)

    if cfg.run_wasserstein:
        assert HAS_WASSERSTEIN is True

        assert should_compute_wasserstein(
            bit_source="full_ecfp4",
            k=cfg.expected_ecfp4_bits,
            bit_repeat=np.nan,
            cfg=cfg,
        )

        XA_test = np.random.default_rng(0).integers(0, 2, size=(30, 32), dtype=np.uint8)
        XB_test = np.random.default_rng(1).integers(0, 2, size=(30, 32), dtype=np.uint8)

        wd_test, wd_norm_test = wasserstein_nd_optional(
            XA_test,
            XB_test,
            n=30,
            dataset="test",
            pair="F1_vs_F2",
            space="full_ecfp4",
            k=cfg.expected_ecfp4_bits,
            model="DT",
            bit_source="full_ecfp4",
            bit_repeat=np.nan,
            bits_used=32,
            cfg=cfg,
        )

        print("Smoke-test Wasserstein:", wd_test, wd_norm_test)

        assert np.isfinite(wd_test), "Wasserstein smoke test failed."
        assert np.isfinite(wd_norm_test), "Normalized Wasserstein smoke test failed."

        print("OK: Wasserstein smoke test passed.")


# ---------------------------------------------------------------------
# Row builders and dataset computation
# ---------------------------------------------------------------------


def _base_row(
    dataset: str,
    pair: str,
    space: str,
    k: int,
    model: str,
    bit_source: str,
    activity_protocol: Any,
    activity_fold: Any,
    bit_repeat: Any,
    bits_used: int,
    cfg: TanimotoDistanceConfig,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "dataset_label": cfg.dataset_labels.get(dataset, dataset),
        "pair": pair,
        "outer_fold": cfg.pair_to_outer_fold.get(pair, np.nan),
        "space": space,
        "k": int(k),
        "model": model,
        "bit_source": bit_source,
        "activity_protocol": activity_protocol,
        "activity_fold": activity_fold,
        "bit_repeat": bit_repeat,
        "bits_used": int(bits_used),
    }


def add_distance_row(
    dist_rows: list[dict[str, Any]],
    hist_rows: list[dict[str, Any]],
    dataset: str,
    pair: str,
    XA: np.ndarray,
    XB: np.ndarray,
    model: str,
    cfg: TanimotoDistanceConfig,
) -> None:
    """Full ECFP4 space row."""
    space = "full_ecfp4"
    bit_source = "full_ecfp4"

    nn = nn_distances(XA, XB)

    pw = complete_pairwise_distance(
        XA,
        XB,
        dataset=dataset,
        pair=pair,
        space=space,
        chunk=cfg.pairwise_chunk_size,
        verbose=cfg.verbose_pairwise,
    )

    wd, wd_norm = wasserstein_nd_optional(
        XA,
        XB,
        cfg.wasserstein_subsample,
        dataset,
        pair,
        space,
        cfg.expected_ecfp4_bits,
        model,
        bit_source,
        np.nan,
        cfg.expected_ecfp4_bits,
        cfg,
    )

    n_valid_mol = nn["n_valid_a"] + nn["n_valid_b"]
    n_total_mol = nn["n_total_a"] + nn["n_total_b"]

    row = _base_row(
        dataset,
        pair,
        space,
        cfg.expected_ecfp4_bits,
        model,
        bit_source,
        np.nan,
        np.nan,
        np.nan,
        cfg.expected_ecfp4_bits,
        cfg,
    )

    row.update(
        {
            "nn_sym_distance": nn["nn_sym_distance"],
            "nn_A_to_B_mean": nn["nn_A_to_B_mean"],
            "nn_B_to_A_mean": nn["nn_B_to_A_mean"],
            "wasserstein_nd": wd,
            "wasserstein_nd_normalized": wd_norm,
            "valid_molecule_fraction": n_valid_mol / max(n_total_mol, 1),
            "n_valid_molecules": n_valid_mol,
            "n_total_molecules": n_total_mol,
            "pairwise_distance": pw["pairwise_distance"],
            "valid_pair_fraction": pw["valid_pair_fraction"],
            "n_valid_pairs": pw["n_valid_pairs"],
            "n_total_pairs": pw["n_total_pairs"],
            "pairwise_mode": pw["pairwise_mode"],
        }
    )

    dist_rows.append(row)

    for d in nn["nn_AB_array"]:
        hist_rows.append(
            {
                "dataset": dataset,
                "dataset_label": cfg.dataset_labels.get(dataset, dataset),
                "model": model,
                "pair": pair,
                "outer_fold": cfg.pair_to_outer_fold.get(pair, np.nan),
                "space": space,
                "k": cfg.expected_ecfp4_bits,
                "bit_source": bit_source,
                "activity_protocol": np.nan,
                "activity_fold": np.nan,
                "bit_repeat": np.nan,
                "bits_used": cfg.expected_ecfp4_bits,
                "direction": "A_to_B",
                "nn_distance": float(d),
            }
        )

    for d in nn["nn_BA_array"]:
        hist_rows.append(
            {
                "dataset": dataset,
                "dataset_label": cfg.dataset_labels.get(dataset, dataset),
                "model": model,
                "pair": pair,
                "outer_fold": cfg.pair_to_outer_fold.get(pair, np.nan),
                "space": space,
                "k": cfg.expected_ecfp4_bits,
                "bit_source": bit_source,
                "activity_protocol": np.nan,
                "activity_fold": np.nan,
                "bit_repeat": np.nan,
                "bits_used": cfg.expected_ecfp4_bits,
                "direction": "B_to_A",
                "nn_distance": float(d),
            }
        )


def add_restricted_distance_row(
    dist_rows: list[dict[str, Any]],
    hist_rows: list[dict[str, Any]],
    dataset: str,
    pair: str,
    XA: np.ndarray,
    XB: np.ndarray,
    bits: np.ndarray,
    k: int,
    model: str,
    space: str,
    bit_source: str,
    activity_protocol: Any,
    activity_fold: Any,
    bit_repeat: Any,
    store_hist: bool,
    cfg: TanimotoDistanceConfig,
) -> None:
    """Top-k restricted space row."""
    if len(bits) == 0 and bit_source in {
        "activity_global",
        "activity_ood",
        "activity_random_shuffle",
        "dataset_detection",
    }:
        warnings.warn(
            f"No restricted-space bits found for {dataset} | {pair} | {space} | "
            f"model={model} | bit_source={bit_source}. "
            "Distances will be NaN or based on empty restricted fingerprints."
        )

    XA_r = restrict_to_bits(XA, bits)
    XB_r = restrict_to_bits(XB, bits)

    nn = nn_distances(XA_r, XB_r)

    pw = complete_pairwise_distance(
        XA_r,
        XB_r,
        dataset=dataset,
        pair=pair,
        space=space,
        chunk=cfg.pairwise_chunk_size,
        verbose=cfg.verbose_pairwise,
    )

    wd, wd_norm = wasserstein_nd_optional(
        XA_r,
        XB_r,
        cfg.wasserstein_subsample,
        dataset,
        pair,
        space,
        k,
        model,
        bit_source,
        bit_repeat,
        len(bits),
        cfg,
    )

    n_valid_mol = nn["n_valid_a"] + nn["n_valid_b"]
    n_total_mol = nn["n_total_a"] + nn["n_total_b"]

    row = _base_row(
        dataset,
        pair,
        space,
        k,
        model,
        bit_source,
        activity_protocol,
        activity_fold,
        bit_repeat,
        len(bits),
        cfg,
    )

    row.update(
        {
            "nn_sym_distance": nn["nn_sym_distance"],
            "nn_A_to_B_mean": nn["nn_A_to_B_mean"],
            "nn_B_to_A_mean": nn["nn_B_to_A_mean"],
            "wasserstein_nd": wd,
            "wasserstein_nd_normalized": wd_norm,
            "valid_molecule_fraction": n_valid_mol / max(n_total_mol, 1),
            "n_valid_molecules": n_valid_mol,
            "n_total_molecules": n_total_mol,
            "pairwise_distance": pw["pairwise_distance"],
            "valid_pair_fraction": pw["valid_pair_fraction"],
            "n_valid_pairs": pw["n_valid_pairs"],
            "n_total_pairs": pw["n_total_pairs"],
            "pairwise_mode": pw["pairwise_mode"],
        }
    )

    dist_rows.append(row)

    if store_hist:
        for d in nn["nn_AB_array"]:
            hist_rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": cfg.dataset_labels.get(dataset, dataset),
                    "model": model,
                    "pair": pair,
                    "outer_fold": cfg.pair_to_outer_fold.get(pair, np.nan),
                    "space": space,
                    "k": k,
                    "bit_source": bit_source,
                    "activity_protocol": activity_protocol,
                    "activity_fold": activity_fold,
                    "bit_repeat": bit_repeat,
                    "bits_used": len(bits),
                    "direction": "A_to_B",
                    "nn_distance": float(d),
                }
            )

        for d in nn["nn_BA_array"]:
            hist_rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": cfg.dataset_labels.get(dataset, dataset),
                    "model": model,
                    "pair": pair,
                    "outer_fold": cfg.pair_to_outer_fold.get(pair, np.nan),
                    "space": space,
                    "k": k,
                    "bit_source": bit_source,
                    "activity_protocol": activity_protocol,
                    "activity_fold": activity_fold,
                    "bit_repeat": bit_repeat,
                    "bits_used": len(bits),
                    "direction": "B_to_A",
                    "nn_distance": float(d),
                }
            )


def compute_dataset_distances(
    dataset: str,
    fps: dict[str, tuple[np.ndarray, pd.DataFrame]],
    list_a: pd.DataFrame,
    list_b: pd.DataFrame,
    cfg: TanimotoDistanceConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print_section(f"Computing distances: {dataset.upper()}-{cfg.task}")

    dist_rows: list[dict[str, Any]] = []
    hist_rows: list[dict[str, Any]] = []

    for model in cfg.models:
        print(f"\n### Model: {model}")

        for fold_a, fold_b in cfg.pairs:
            pair = f"{fold_a}_vs_{fold_b}"
            outer_fold = cfg.pair_to_outer_fold[pair]

            XA, _ = fps[fold_a]
            XB, _ = fps[fold_b]

            print(
                f"\n  Pair {pair} (outer fold {outer_fold}) | "
                f"model={model}: nA={XA.shape[0]} nB={XB.shape[0]}"
            )

            # Full ECFP4 space.
            # This is repeated per model to keep the output table aligned with
            # the restricted-space model-wise rows.
            add_distance_row(dist_rows, hist_rows, dataset, pair, XA, XB, model, cfg)

            # Restricted top-k spaces.
            for k in cfg.k_values:
                print(
                    f"    k={k}: activity spaces + dataset-detection space "
                    "+ random-bit baseline"
                )

                # Global activity bits (pooled across protocols and folds)
                bits_g = topk_bits_global(list_a, dataset, model, k)
                print(f"      activity_global: {len(bits_g)} bits")

                add_restricted_distance_row(
                    dist_rows,
                    hist_rows,
                    dataset,
                    pair,
                    XA,
                    XB,
                    bits=bits_g,
                    k=k,
                    model=model,
                    space=f"activity_global_top{k}",
                    bit_source="activity_global",
                    activity_protocol="pooled",
                    activity_fold=np.nan,
                    bit_repeat=np.nan,
                    store_hist=True,
                    cfg=cfg,
                )

                # OOD activity bits, fold-aware
                bits_ood = topk_bits_fold_protocol(
                    list_a, dataset, model, outer_fold, "ood", k
                )
                print(f"      activity_ood fold={outer_fold}: {len(bits_ood)} bits")

                add_restricted_distance_row(
                    dist_rows,
                    hist_rows,
                    dataset,
                    pair,
                    XA,
                    XB,
                    bits=bits_ood,
                    k=k,
                    model=model,
                    space=f"activity_ood_top{k}",
                    bit_source="activity_ood",
                    activity_protocol="ood",
                    activity_fold=outer_fold,
                    bit_repeat=np.nan,
                    store_hist=True,
                    cfg=cfg,
                )

                # Random-shuffle activity bits, fold-aware
                bits_rnd = topk_bits_fold_protocol(
                    list_a, dataset, model, outer_fold, "random", k
                )
                print(
                    f"      activity_random_shuffle fold={outer_fold}: {len(bits_rnd)} bits"
                )

                add_restricted_distance_row(
                    dist_rows,
                    hist_rows,
                    dataset,
                    pair,
                    XA,
                    XB,
                    bits=bits_rnd,
                    k=k,
                    model=model,
                    space=f"activity_random_shuffle_top{k}",
                    bit_source="activity_random_shuffle",
                    activity_protocol="random",
                    activity_fold=outer_fold,
                    bit_repeat=np.nan,
                    store_hist=True,
                    cfg=cfg,
                )

                # Dataset/fold detection bits, fold-pair-aware
                bits_det = topk_bits_list_b(list_b, dataset, model, pair, k)
                print(f"      dataset_detection {pair}: {len(bits_det)} bits")

                add_restricted_distance_row(
                    dist_rows,
                    hist_rows,
                    dataset,
                    pair,
                    XA,
                    XB,
                    bits=bits_det,
                    k=k,
                    model=model,
                    space=f"dataset_detection_top{k}",
                    bit_source="dataset_detection",
                    activity_protocol="same_search_cv",
                    activity_fold=outer_fold,
                    bit_repeat=np.nan,
                    store_hist=True,
                    cfg=cfg,
                )

                # Random-bit baseline (negative control for dimensionality)
                for r in range(cfg.n_random_bit_repeats):
                    if (
                        r == 0
                        or (r + 1) % cfg.print_every_random_repeat == 0
                        or r == cfg.n_random_bit_repeats - 1
                    ):
                        print(
                            f"      random_bits repeat {r + 1}/{cfg.n_random_bit_repeats}"
                        )

                    bits_rb = random_topk_bits(dataset, pair, k, r, cfg)

                    add_restricted_distance_row(
                        dist_rows,
                        hist_rows,
                        dataset,
                        pair,
                        XA,
                        XB,
                        bits=bits_rb,
                        k=k,
                        model=model,
                        space=f"random_bits_top{k}",
                        bit_source="random_bits",
                        activity_protocol=np.nan,
                        activity_fold=np.nan,
                        bit_repeat=r,
                        store_hist=False,
                        cfg=cfg,
                    )

    dist_df = pd.DataFrame(dist_rows)
    hist_df = pd.DataFrame(hist_rows)

    n_random_bit_rows = (dist_df["bit_source"] == "random_bits").sum()
    expected_rb = (
        len(cfg.models) * len(cfg.pairs) * len(cfg.k_values) * cfg.n_random_bit_repeats
    )

    print(f"\n  Dataset {dataset}: dist_rows={len(dist_df)}, hist_rows={len(hist_df)}")
    print(
        f"  random_bits rows = {n_random_bit_rows} (expected {expected_rb}: "
        f"{len(cfg.models)} models × {len(cfg.pairs)} pairs × "
        f"{len(cfg.k_values)} k × {cfg.n_random_bit_repeats} repeats)"
    )

    expected_total = (
        len(cfg.models)
        * len(cfg.pairs)
        * (1 + len(cfg.k_values) * (1 + 1 + 1 + 1 + cfg.n_random_bit_repeats))
    )

    if len(dist_df) != expected_total:
        warnings.warn(
            f"{dataset}: got {len(dist_df)} dist rows, expected {expected_total}"
        )

    return dist_df, hist_df


def run_distance_pipeline(
    list_a: pd.DataFrame,
    list_b: pd.DataFrame,
    cfg: TanimotoDistanceConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg.ensure_paths()

    dist_parts: list[pd.DataFrame] = []
    hist_parts: list[pd.DataFrame] = []

    for ds in cfg.datasets:
        print_section(f"Running dataset: {ds.upper()}")

        fps = load_subset_fps(ds, cfg)

        t0 = time.time()

        dist_df, hist_df = compute_dataset_distances(
            dataset=ds,
            fps=fps,
            list_a=list_a,
            list_b=list_b,
            cfg=cfg,
        )

        elapsed = time.time() - t0

        dist_parts.append(dist_df)
        hist_parts.append(hist_df)

        dataset_dist_path = cfg.out_root / f"fold_distance_summary_{ds}.csv"
        dataset_hist_path = cfg.out_root / f"fold_distance_nn_distributions_{ds}.csv"

        dist_df.to_csv(dataset_dist_path, index=False)
        hist_df.to_csv(dataset_hist_path, index=False)

        print(
            f"Saved {ds}: dist_rows={len(dist_df)}, hist_rows={len(hist_df)}, "
            f"time={elapsed / 60:.1f} min"
        )
        print(f"  {dataset_dist_path}")
        print(f"  {dataset_hist_path}")

    if not dist_parts:
        raise RuntimeError("No dataset produced distance rows.")

    dist_all = pd.concat(dist_parts, ignore_index=True)
    hist_all = (
        pd.concat(hist_parts, ignore_index=True) if hist_parts else pd.DataFrame()
    )

    missing_cols = set(DIST_ROW_KEYS) - set(dist_all.columns)
    assert not missing_cols, f"dist_all missing required columns: {missing_cols}"

    dist_path = cfg.out_root / "fold_distance_summary.csv"
    hist_path = cfg.out_root / "fold_distance_nn_distributions.csv"

    dist_all.to_csv(dist_path, index=False)
    hist_all.to_csv(hist_path, index=False)

    print("Saved global tables.")
    print(f"dist_all: {dist_all.shape} -> {dist_path}")
    print(f"hist_all: {hist_all.shape} -> {hist_path}")

    return dist_all, hist_all


# ---------------------------------------------------------------------
# Diagnostics and summary
# ---------------------------------------------------------------------


def print_output_diagnostics(
    dist_all: pd.DataFrame,
    hist_all: pd.DataFrame,
    cfg: TanimotoDistanceConfig,
) -> None:
    print_section("Output diagnostics")

    print(f"dist_all shape: {dist_all.shape}")
    print(f"hist_all shape: {hist_all.shape}")

    if len(dist_all) == 0:
        raise RuntimeError("dist_all is empty — distance computation produced no rows.")

    print("\nRows by dataset / model / bit_source:")
    print(
        dist_all.groupby(["dataset", "model", "bit_source"])
        .size()
        .rename("n")
        .reset_index()
        .to_string(index=False)
    )

    expected_per_dataset = (
        len(cfg.models)
        * len(cfg.pairs)
        * (1 + len(cfg.k_values) * (4 + cfg.n_random_bit_repeats))
    )

    expected_total = expected_per_dataset * len(cfg.datasets)

    print("\nExpected row count:")
    print(f"  per dataset: {expected_per_dataset}")
    print(f"  total:       {expected_total}")
    print(f"  observed:    {len(dist_all)}")

    unexpected_datasets = set(dist_all["dataset"].unique()) - set(cfg.datasets)
    assert not unexpected_datasets, f"Unexpected datasets found: {unexpected_datasets}"

    assert "kdr" not in set(
        dist_all["dataset"].astype(str).str.lower()
    ), "KDR leaked into the analysis."

    for ds in cfg.datasets:
        got = int((dist_all["dataset"] == ds).sum())
        flag = "OK" if got == expected_per_dataset else "MISSING"
        print(f"  {ds}: got {got}, expected {expected_per_dataset} [{flag}]")

    if len(dist_all) != expected_total:
        raise RuntimeError(
            f"Unexpected total row count: got {len(dist_all)}, expected {expected_total}"
        )

    expected_bit_sources = {
        "full_ecfp4",
        "activity_global",
        "activity_ood",
        "activity_random_shuffle",
        "dataset_detection",
        "random_bits",
    }

    missing_sources = expected_bit_sources - set(dist_all["bit_source"].unique())
    if missing_sources:
        raise RuntimeError(f"Missing bit sources: {missing_sources}")

    missing_models = set(cfg.models) - set(dist_all["model"].unique())
    if missing_models:
        raise RuntimeError(f"Missing models: {missing_models}")

    if "pairwise_distance" not in dist_all.columns:
        raise RuntimeError("Missing main metric column: pairwise_distance")

    if dist_all["pairwise_distance"].notna().sum() == 0:
        raise RuntimeError("All pairwise_distance values are NaN — pipeline failed.")

    print(
        f"\nFinite pairwise_distance: "
        f"{dist_all['pairwise_distance'].notna().sum()}/{len(dist_all)}"
    )

    if "valid_pair_fraction" in dist_all.columns:
        print("\nValid pair fraction summary:")
        print(
            dist_all["valid_pair_fraction"]
            .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            .to_string()
        )

    if "valid_molecule_fraction" in dist_all.columns:
        print("\nValid molecule fraction summary:")
        print(
            dist_all["valid_molecule_fraction"]
            .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            .to_string()
        )

    print("\nOK: diagnostics passed.")


def build_summary(
    dist_all: pd.DataFrame,
    cfg: TanimotoDistanceConfig,
) -> pd.DataFrame:
    print_section("Building final summary")

    summary = dist_all.groupby(
        [
            "dataset",
            "dataset_label",
            "model",
            "pair",
            "outer_fold",
            "k",
            "bit_source",
            "activity_protocol",
            "activity_fold",
        ],
        dropna=False,
        as_index=False,
    ).agg(
        pairwise_distance_mean=("pairwise_distance", "mean"),
        pairwise_distance_std=("pairwise_distance", "std"),
        wasserstein_nd_mean=("wasserstein_nd", "mean"),
        wasserstein_nd_std=("wasserstein_nd", "std"),
        wasserstein_nd_normalized_mean=("wasserstein_nd_normalized", "mean"),
        wasserstein_nd_normalized_std=("wasserstein_nd_normalized", "std"),
        valid_molecule_fraction_mean=("valid_molecule_fraction", "mean"),
        valid_molecule_fraction_std=("valid_molecule_fraction", "std"),
        valid_pair_fraction_mean=("valid_pair_fraction", "mean"),
        valid_pair_fraction_std=("valid_pair_fraction", "std"),
        bits_used_mean=("bits_used", "mean"),
        bits_used_std=("bits_used", "std"),
        n_rows=("pairwise_distance", "size"),
        n_wasserstein_rows=("wasserstein_nd", lambda x: int(x.notna().sum())),
    )

    # Full ECFP4 reference for each dataset/model/pair/fold.
    full_ref = summary[summary["bit_source"] == "full_ecfp4"][
        [
            "dataset",
            "model",
            "pair",
            "outer_fold",
            "pairwise_distance_mean",
            "wasserstein_nd_normalized_mean",
            "valid_molecule_fraction_mean",
            "valid_pair_fraction_mean",
        ]
    ].rename(
        columns={
            "pairwise_distance_mean": "full_pairwise_distance_mean",
            "wasserstein_nd_normalized_mean": "full_wasserstein_nd_normalized_mean",
            "valid_molecule_fraction_mean": "full_valid_molecule_fraction_mean",
            "valid_pair_fraction_mean": "full_valid_pair_fraction_mean",
        }
    )

    summary = summary.merge(
        full_ref,
        on=["dataset", "model", "pair", "outer_fold"],
        how="left",
    )

    summary["delta_minus_full_pairwise"] = (
        summary["pairwise_distance_mean"] - summary["full_pairwise_distance_mean"]
    )

    summary["delta_minus_full_wasserstein_nd_normalized"] = (
        summary["wasserstein_nd_normalized_mean"]
        - summary["full_wasserstein_nd_normalized_mean"]
    )

    # Random-bit reference for each dataset/model/pair/fold/k.
    # This is the key baseline for dimensionality control.
    random_ref = summary[summary["bit_source"] == "random_bits"][
        [
            "dataset",
            "model",
            "pair",
            "outer_fold",
            "k",
            "pairwise_distance_mean",
            "pairwise_distance_std",
            "wasserstein_nd_normalized_mean",
            "wasserstein_nd_normalized_std",
            "valid_molecule_fraction_mean",
            "valid_pair_fraction_mean",
            "bits_used_mean",
            "n_rows",
            "n_wasserstein_rows",
        ]
    ].rename(
        columns={
            "pairwise_distance_mean": "random_bits_pairwise_distance_mean",
            "pairwise_distance_std": "random_bits_pairwise_distance_std",
            "wasserstein_nd_normalized_mean": "random_bits_wasserstein_nd_normalized_mean",
            "wasserstein_nd_normalized_std": "random_bits_wasserstein_nd_normalized_std",
            "valid_molecule_fraction_mean": "random_bits_valid_molecule_fraction_mean",
            "valid_pair_fraction_mean": "random_bits_valid_pair_fraction_mean",
            "bits_used_mean": "random_bits_bits_used_mean",
            "n_rows": "random_bits_n_repeats",
            "n_wasserstein_rows": "random_bits_n_wasserstein_rows",
        }
    )

    summary = summary.merge(
        random_ref,
        on=["dataset", "model", "pair", "outer_fold", "k"],
        how="left",
    )

    # Raw deltas against random bits.
    summary["delta_minus_random_bits_pairwise"] = (
        summary["pairwise_distance_mean"]
        - summary["random_bits_pairwise_distance_mean"]
    )

    summary["delta_minus_random_bits_wasserstein_nd_normalized"] = (
        summary["wasserstein_nd_normalized_mean"]
        - summary["random_bits_wasserstein_nd_normalized_mean"]
    )

    # Z-scores against the random-bit repeat distribution.
    # Negative z means the restricted activity/detection space has lower distance
    # than expected under random bits of the same nominal dimensionality.
    summary["delta_minus_random_bits_pairwise_z"] = np.where(
        summary["random_bits_pairwise_distance_std"] > 0,
        summary["delta_minus_random_bits_pairwise"]
        / summary["random_bits_pairwise_distance_std"],
        np.nan,
    )

    summary["delta_minus_random_bits_wasserstein_nd_normalized_z"] = np.where(
        summary["random_bits_wasserstein_nd_normalized_std"] > 0,
        summary["delta_minus_random_bits_wasserstein_nd_normalized"]
        / summary["random_bits_wasserstein_nd_normalized_std"],
        np.nan,
    )

    # Coverage diagnostics.
    # Restricted-space distances are conditional on molecules/pairs with non-zero
    # restricted fingerprints. For this reason, activity-vs-random comparisons
    # are clean only when both restricted space and random-bit baseline have high
    # and comparable coverage.
    summary["coverage_gap_vs_random_molecule"] = (
        summary["valid_molecule_fraction_mean"]
        - summary["random_bits_valid_molecule_fraction_mean"]
    )

    summary["coverage_gap_vs_random_pair"] = (
        summary["valid_pair_fraction_mean"]
        - summary["random_bits_valid_pair_fraction_mean"]
    )

    summary["dimension_gap_vs_random_bits"] = (
        summary["bits_used_mean"] - summary["random_bits_bits_used_mean"]
    )

    summary["dimension_ok_vs_random"] = (
        summary["dimension_gap_vs_random_bits"].abs() < 1e-9
    )

    summary["coverage_ok_vs_random"] = (
        (summary["valid_molecule_fraction_mean"] >= cfg.coverage_threshold)
        & (summary["valid_pair_fraction_mean"] >= cfg.coverage_threshold)
        & (
            summary["random_bits_valid_molecule_fraction_mean"]
            >= cfg.coverage_threshold
        )
        & (summary["random_bits_valid_pair_fraction_mean"] >= cfg.coverage_threshold)
    )

    summary["high_coverage_recommended_for_main_plot"] = (
        summary["coverage_ok_vs_random"]
        & summary["dimension_ok_vs_random"]
        & summary["bit_source"].isin(
            [
                "activity_global",
                "activity_ood",
                "activity_random_shuffle",
                "dataset_detection",
            ]
        )
    )

    # Wasserstein is most interpretable for same-k comparisons against random bits.
    # Delta-vs-full Wasserstein is kept only as a secondary diagnostic.
    summary["wasserstein_same_k_random_comparison_ok"] = (
        summary["bit_source"].isin(
            [
                "activity_global",
                "activity_ood",
                "activity_random_shuffle",
                "dataset_detection",
            ]
        )
        & summary["random_bits_wasserstein_nd_normalized_mean"].notna()
        & summary["coverage_ok_vs_random"]
        & summary["dimension_ok_vs_random"]
    )

    # Main-safe versions: these are the columns that should be used directly in
    # final plots. Raw deltas are kept for diagnostics, but the main-safe values
    # become NaN whenever coverage or dimensionality is not comparable.
    summary["delta_minus_random_bits_pairwise_main"] = np.where(
        summary["high_coverage_recommended_for_main_plot"],
        summary["delta_minus_random_bits_pairwise"],
        np.nan,
    )

    summary["delta_minus_random_bits_pairwise_z_main"] = np.where(
        summary["high_coverage_recommended_for_main_plot"],
        summary["delta_minus_random_bits_pairwise_z"],
        np.nan,
    )

    summary["delta_minus_random_bits_wasserstein_nd_normalized_main"] = np.where(
        summary["wasserstein_same_k_random_comparison_ok"],
        summary["delta_minus_random_bits_wasserstein_nd_normalized"],
        np.nan,
    )

    summary["delta_minus_random_bits_wasserstein_nd_normalized_z_main"] = np.where(
        summary["wasserstein_same_k_random_comparison_ok"],
        summary["delta_minus_random_bits_wasserstein_nd_normalized_z"],
        np.nan,
    )

    summary = summary.sort_values(
        [
            "dataset",
            "model",
            "outer_fold",
            "k",
            "bit_source",
            "activity_protocol",
        ],
        na_position="last",
    ).reset_index(drop=True)

    out_path = (
        cfg.out_root / "fold_distance_activity_detection_vs_random_bits_summary.csv"
    )
    summary.to_csv(out_path, index=False)

    print(f"Saved summary: {out_path} ({len(summary)} rows)")

    display_cols = [
        "dataset",
        "model",
        "pair",
        "outer_fold",
        "k",
        "bit_source",
        "activity_protocol",
        "bits_used_mean",
        "random_bits_bits_used_mean",
        "dimension_ok_vs_random",
        "pairwise_distance_mean",
        "random_bits_pairwise_distance_mean",
        "delta_minus_random_bits_pairwise",
        "delta_minus_random_bits_pairwise_z",
        "delta_minus_random_bits_pairwise_z_main",
        "valid_molecule_fraction_mean",
        "random_bits_valid_molecule_fraction_mean",
        "valid_pair_fraction_mean",
        "random_bits_valid_pair_fraction_mean",
        "coverage_ok_vs_random",
        "high_coverage_recommended_for_main_plot",
        "wasserstein_nd_normalized_mean",
        "random_bits_wasserstein_nd_normalized_mean",
        "delta_minus_random_bits_wasserstein_nd_normalized",
        "delta_minus_random_bits_wasserstein_nd_normalized_z",
        "delta_minus_random_bits_wasserstein_nd_normalized_z_main",
        "wasserstein_same_k_random_comparison_ok",
        "n_rows",
        "random_bits_n_repeats",
        "n_wasserstein_rows",
        "random_bits_n_wasserstein_rows",
    ]

    print("\nSummary preview:")
    print(summary[display_cols].head(40).to_string(index=False))

    print("\nCoverage eligibility by k and bit_source:")
    coverage_check = (
        summary[
            summary["bit_source"].isin(
                [
                    "activity_global",
                    "activity_ood",
                    "activity_random_shuffle",
                    "dataset_detection",
                ]
            )
        ]
        .groupby(["k", "bit_source"], as_index=False)
        .agg(
            n_rows=("coverage_ok_vs_random", "size"),
            n_coverage_ok=("coverage_ok_vs_random", "sum"),
            n_high_coverage_main=("high_coverage_recommended_for_main_plot", "sum"),
            mean_valid_molecule_fraction=("valid_molecule_fraction_mean", "mean"),
            mean_random_valid_molecule_fraction=(
                "random_bits_valid_molecule_fraction_mean",
                "mean",
            ),
            mean_valid_pair_fraction=("valid_pair_fraction_mean", "mean"),
            mean_random_valid_pair_fraction=(
                "random_bits_valid_pair_fraction_mean",
                "mean",
            ),
            mean_bits_used=("bits_used_mean", "mean"),
            mean_random_bits_used=("random_bits_bits_used_mean", "mean"),
        )
    )

    print(coverage_check.to_string(index=False))

    return summary


def coverage_and_random_control_checks(final_summary: pd.DataFrame) -> None:
    print_section("Coverage and random-control checks")

    required_summary_cols = [
        "random_bits_pairwise_distance_mean",
        "random_bits_pairwise_distance_std",
        "delta_minus_random_bits_pairwise",
        "delta_minus_random_bits_pairwise_z",
        "delta_minus_random_bits_pairwise_z_main",
        "random_bits_valid_molecule_fraction_mean",
        "random_bits_valid_pair_fraction_mean",
        "coverage_ok_vs_random",
        "dimension_ok_vs_random",
        "high_coverage_recommended_for_main_plot",
    ]

    missing_summary_cols = [
        c for c in required_summary_cols if c not in final_summary.columns
    ]

    assert (
        not missing_summary_cols
    ), f"Missing required random-control columns in final_summary: {missing_summary_cols}"

    restricted = final_summary[
        final_summary["bit_source"].isin(
            [
                "activity_global",
                "activity_ood",
                "activity_random_shuffle",
                "dataset_detection",
            ]
        )
    ].copy()

    print("Coverage-ok rows by k:")
    print(
        restricted.groupby("k")["coverage_ok_vs_random"]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .to_string(index=False)
    )

    print("\nRecommended high-coverage rows by k:")
    print(
        restricted.groupby("k")["high_coverage_recommended_for_main_plot"]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .to_string(index=False)
    )

    print("\nPairwise z-score summary:")
    print(
        restricted["delta_minus_random_bits_pairwise_z"]
        .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        .to_string()
    )

    low_coverage = restricted[~restricted["coverage_ok_vs_random"]].copy()

    if len(low_coverage) > 0:
        print(
            "\nWARNING: some restricted-space rows are not coverage-comparable with random bits."
        )
        print(
            low_coverage[
                [
                    "dataset",
                    "model",
                    "pair",
                    "k",
                    "bit_source",
                    "valid_molecule_fraction_mean",
                    "random_bits_valid_molecule_fraction_mean",
                    "valid_pair_fraction_mean",
                    "random_bits_valid_pair_fraction_mean",
                    "coverage_ok_vs_random",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    print("\nOK: random-control coverage diagnostics completed.")


def final_checks(dist_all: pd.DataFrame, cfg: TanimotoDistanceConfig) -> None:
    print_section("Final checks")

    print("Rows by dataset / model / bit_source:")
    print(
        dist_all.groupby(["dataset", "model", "bit_source"])
        .size()
        .rename("n")
        .reset_index()
        .to_string(index=False)
    )

    print("\nUnique models:")
    print(sorted(dist_all["model"].unique()))

    print("\nUnique bit sources:")
    print(sorted(dist_all["bit_source"].unique()))

    print("\nK values:")
    print(sorted(dist_all["k"].unique()))

    print("\nDatasets:")
    print(sorted(dist_all["dataset"].unique()))

    unexpected_datasets = set(dist_all["dataset"].unique()) - set(cfg.datasets)
    assert not unexpected_datasets, f"Unexpected datasets found: {unexpected_datasets}"

    assert "kdr" not in set(
        dist_all["dataset"].astype(str).str.lower()
    ), "KDR leaked into the analysis."

    observed_k = set(dist_all.loc[dist_all["bit_source"] != "full_ecfp4", "k"].unique())
    expected_k = set(cfg.k_values)
    assert (
        observed_k == expected_k
    ), f"Unexpected k values. Observed={sorted(observed_k)}, expected={sorted(expected_k)}"

    expected_bit_sources = {
        "full_ecfp4",
        "activity_global",
        "activity_ood",
        "activity_random_shuffle",
        "dataset_detection",
        "random_bits",
    }

    missing_sources = expected_bit_sources - set(dist_all["bit_source"].unique())
    assert not missing_sources, f"Missing bit sources: {missing_sources}"

    missing_models = set(cfg.models) - set(dist_all["model"].unique())
    assert not missing_models, f"Missing models: {missing_models}"

    if "pairwise_distance" not in dist_all.columns:
        raise RuntimeError("Missing main metric column: pairwise_distance")

    if dist_all["pairwise_distance"].notna().sum() == 0:
        raise RuntimeError("All pairwise_distance values are NaN — pipeline failed.")

    print(
        f"\nFinite pairwise_distance: "
        f"{dist_all['pairwise_distance'].notna().sum()}/{len(dist_all)}"
    )

    if "valid_pair_fraction" in dist_all.columns:
        print("\nValid pair fraction summary:")
        print(
            dist_all["valid_pair_fraction"]
            .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            .to_string()
        )

    if "valid_molecule_fraction" in dist_all.columns:
        print("\nValid molecule fraction summary:")
        print(
            dist_all["valid_molecule_fraction"]
            .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            .to_string()
        )

    print("\nOK: model-wise and task-wise Tanimoto distance table completed.")


def wasserstein_checks(dist_all: pd.DataFrame, cfg: TanimotoDistanceConfig) -> None:
    print_section("Wasserstein checks")

    if "wasserstein_nd" not in dist_all.columns:
        raise RuntimeError("Missing wasserstein_nd column.")

    n_w = int(dist_all["wasserstein_nd"].notna().sum())
    print(f"Finite wasserstein_nd rows: {n_w}/{len(dist_all)}")

    if cfg.run_wasserstein and n_w == 0:
        raise RuntimeError(
            "RUN_WASSERSTEIN=True but no finite Wasserstein values were computed."
        )

    if cfg.run_wasserstein:
        print("\nRows with finite Wasserstein by bit_source:")
        print(
            dist_all[dist_all["wasserstein_nd"].notna()]
            .groupby(["bit_source", "k"])
            .size()
            .rename("n")
            .reset_index()
            .to_string(index=False)
        )

        print("\nNormalized Wasserstein summary:")
        print(
            dist_all["wasserstein_nd_normalized"]
            .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            .to_string()
        )

    print("\nOK: Wasserstein diagnostics completed.")


def run_full_pipeline(
    cfg: TanimotoDistanceConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience runner for the full model-wise Tanimoto-distance pipeline.

    Returns:
        list_a, list_b, dist_all, hist_all, final_summary
    """
    if cfg is None:
        cfg = TanimotoDistanceConfig()

    cfg.ensure_paths()
    ensure_project_on_path(cfg.project_root)

    print(f"Project root: {cfg.project_root}")
    print(f"scipy wasserstein_distance_nd available: {HAS_WASSERSTEIN}")
    print(f"OUT_ROOT: {cfg.out_root}")
    print(f"FIG_ROOT: {cfg.fig_root}")
    print(f"CACHE_ROOT: {cfg.cache_root}")
    print(f"RUN_WASSERSTEIN: {cfg.run_wasserstein}")
    print(f"WASSERSTEIN_K_VALUES: {cfg.wasserstein_k_values}")
    print(
        f"W_RANDOM_BIT_REPEATS: "
        f"{cfg.wasserstein_random_bit_repeats}/{cfg.n_random_bit_repeats}"
    )

    list_a = load_list_a(cfg)
    list_b = load_list_b(cfg)

    pre_run_sanity_checks(list_a, list_b, cfg)

    print_section("Model-wise analysis")
    print(
        "The Tanimoto restricted-space analysis is now computed model-wise. "
        "No single best ECFP4 model is selected per dataset."
    )
    print(f"Models included: {cfg.models}")

    build_selected_bits_table(list_a, list_b, cfg)

    dist_all, hist_all = run_distance_pipeline(list_a, list_b, cfg)

    print_output_diagnostics(dist_all, hist_all, cfg)

    final_summary = build_summary(dist_all, cfg)

    final_checks(dist_all, cfg)
    wasserstein_checks(dist_all, cfg)
    coverage_and_random_control_checks(final_summary)

    return list_a, list_b, dist_all, hist_all, final_summary


__all__ = [
    "TanimotoDistanceConfig",
    "HAS_WASSERSTEIN",
    "DIST_ROW_KEYS",
    "find_project_root",
    "ensure_project_on_path",
    "stable_seed",
    "local_rng",
    "print_section",
    "load_subset_fps",
    "load_list_a",
    "load_list_b",
    "pre_run_sanity_checks",
    "build_selected_bits_table",
    "complete_pairwise_distance",
    "nn_distances",
    "topk_bits_global",
    "topk_bits_fold_protocol",
    "topk_bits_list_b",
    "random_topk_bits",
    "should_compute_wasserstein",
    "wasserstein_nd_optional",
    "compute_dataset_distances",
    "run_distance_pipeline",
    "print_output_diagnostics",
    "build_summary",
    "coverage_and_random_control_checks",
    "final_checks",
    "wasserstein_checks",
    "run_full_pipeline",
]
