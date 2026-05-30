#!/usr/bin/env python
# coding: utf-8
"""
Cross-dataset tables for Hi OOD-vs-Random-Shuffle analysis.

Reads per-fold artifacts (params, complexity, feature importance) for all four
Hi datasets (DRD2, HIV, KDR, Sol) and produces unified cross-dataset CSV tables
consumed by the companion plotting script.

Outputs (saved to results/results_ood_vs_random_shuffle/hi/cross_dataset/):
  - cross_dataset_protocol_per_fold.csv
  - cross_dataset_protocol_summary.csv
  - cross_dataset_protocol_delta.csv
  - cross_dataset_complexity_all.csv
  - cross_dataset_complexity_summary.csv
  - cross_dataset_feature_importance_all.csv
  - cross_dataset_feature_topk.csv
  - cross_dataset_feature_overlap.csv
  - cross_dataset_feature_concentration.csv
"""

import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_ROOT = PROJECT_ROOT / "results" / "hi"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "results_ood_vs_random_shuffle"
    / "hi"
    / "cross_dataset"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FOLDS = [1, 2, 3]
TOP_K_VALUES = [10, 20, 50]

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
# Each entry: (model, model_short, fingerprint, protocol, dir_name_template)
# dir_name_template uses {dataset} as placeholder.

_EXPERIMENT_TEMPLATES = [
    # Decision Tree
    ("Decision Tree", "DT", "ecfp4", "ECFP4", "OOD holdout",
     "dt_{dataset}_hi_inner_ood_holdout_ecfp4"),
    ("Decision Tree", "DT", "ecfp4", "ECFP4", "Random shuffle",
     "dt_{dataset}_hi_random_shuffle_ecfp4"),
    ("Decision Tree", "DT", "maccs", "MACCS", "OOD holdout",
     "dt_{dataset}_hi_inner_ood_holdout_maccs"),
    ("Decision Tree", "DT", "maccs", "MACCS", "Random shuffle",
     "dt_{dataset}_hi_random_shuffle_maccs"),
    # Logistic Regression
    ("Logistic Regression", "LR", "ecfp4", "ECFP4", "OOD holdout",
     "lr_{dataset}_hi_inner_ood_holdout_ecfp4"),
    ("Logistic Regression", "LR", "ecfp4", "ECFP4", "Random shuffle",
     "lr_{dataset}_hi_random_shuffle_ecfp4"),
    ("Logistic Regression", "LR", "maccs", "MACCS", "OOD holdout",
     "lr_{dataset}_hi_inner_ood_holdout_maccs"),
    ("Logistic Regression", "LR", "maccs", "MACCS", "Random shuffle",
     "lr_{dataset}_hi_random_shuffle_maccs"),
    ("Logistic Regression", "LR", "rdkit_desc", "RDKit desc", "OOD holdout",
     "lr_{dataset}_hi_inner_ood_holdout_rdkit_desc"),
    ("Logistic Regression", "LR", "rdkit_desc", "RDKit desc", "Random shuffle",
     "lr_{dataset}_hi_random_shuffle_rdkit_desc"),
    # Linear SVM
    ("Linear SVM", "SVM", "ecfp4", "ECFP4", "OOD holdout",
     "svm_linear_{dataset}_hi_inner_ood_holdout_ecfp4"),
    ("Linear SVM", "SVM", "ecfp4", "ECFP4", "Random shuffle",
     "svm_linear_{dataset}_hi_random_shuffle_ecfp4"),
    ("Linear SVM", "SVM", "maccs", "MACCS", "OOD holdout",
     "svm_linear_{dataset}_hi_inner_ood_holdout_maccs"),
    ("Linear SVM", "SVM", "maccs", "MACCS", "Random shuffle",
     "svm_linear_{dataset}_hi_random_shuffle_maccs"),
]

DATASETS = ["drd2", "hiv", "kdr", "sol"]

DATASET_LABELS = {
    "drd2": "DRD2",
    "hiv": "HIV",
    "kdr": "KDR",
    "sol": "Sol",
}

ORDER_MODEL = {"Decision Tree": 0, "Logistic Regression": 1, "Linear SVM": 2}
ORDER_FP = {"ECFP4": 0, "MACCS": 1, "RDKit desc": 2}
ORDER_PROTOCOL = {"OOD holdout": 0, "Random shuffle": 1}
ORDER_DATASET = {"drd2": 0, "hiv": 1, "kdr": 2, "sol": 3}


def _build_registry() -> pd.DataFrame:
    """Build the full experiment registry across all datasets."""
    rows = []
    for dataset in DATASETS:
        dataset_dir = RESULTS_ROOT / dataset
        for (model, model_short, fp_type, fp_label, protocol,
             dir_template) in _EXPERIMENT_TEMPLATES:
            dir_name = dir_template.format(dataset=dataset)
            result_dir = dataset_dir / dir_name
            rows.append({
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "model": model,
                "model_short": model_short,
                "fp_type": fp_type,
                "fingerprint": fp_label,
                "protocol": protocol,
                "dir_name": dir_name,
                "result_dir": result_dir,
                "exists": result_dir.exists(),
            })
    df = pd.DataFrame(rows)
    n_found = df["exists"].sum()
    n_total = len(df)
    print(f"Registry: {n_found}/{n_total} experiment directories found.")
    missing = df[~df["exists"]]
    if len(missing) > 0:
        for _, r in missing.iterrows():
            warnings.warn(
                f"Missing: {r['dataset']}/{r['dir_name']}"
            )
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _add_ordering_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add sort-order columns for consistent ordering."""
    df = df.copy()
    if "model" in df.columns:
        df["model_order"] = df["model"].map(ORDER_MODEL)
    if "fingerprint" in df.columns:
        df["fingerprint_order"] = df["fingerprint"].map(ORDER_FP)
    if "protocol" in df.columns:
        df["protocol_order"] = df["protocol"].map(ORDER_PROTOCOL)
    if "dataset" in df.columns:
        df["dataset_order"] = df["dataset"].map(ORDER_DATASET)
    return df


# ---------------------------------------------------------------------------
# 1. Protocol per-fold table
# ---------------------------------------------------------------------------

def build_protocol_per_fold(registry: pd.DataFrame) -> pd.DataFrame:
    """Load params_fold_i.json for every experiment and build per-fold table."""
    rows = []
    for _, exp in registry[registry["exists"]].iterrows():
        for fold in FOLDS:
            params_path = exp["result_dir"] / f"params_fold_{fold}.json"
            if not params_path.exists():
                warnings.warn(f"Missing: {params_path}")
                continue

            data = _read_json(params_path)
            train_m = data.get("train_metrics", {})
            test_m = data.get("test_metrics", {})

            inner = data.get("inner_selection_score", np.nan)
            inner_train = data.get("inner_train_score", np.nan)
            train_pr = train_m.get("pr_auc", np.nan)
            test_pr = test_m.get("pr_auc", np.nan)

            rows.append({
                "dataset": exp["dataset"],
                "dataset_label": exp["dataset_label"],
                "model": exp["model"],
                "model_short": exp["model_short"],
                "fingerprint": exp["fingerprint"],
                "protocol": exp["protocol"],
                "fold": fold,
                "inner_pr_auc": inner,
                "inner_train_pr_auc": inner_train,
                "train_pr_auc": train_pr,
                "test_pr_auc": test_pr,
                "inner_test_gap": inner - test_pr,
                "train_test_gap": train_pr - test_pr,
            })

    df = pd.DataFrame(rows)
    df = _add_ordering_columns(df)
    df = df.sort_values(
        ["dataset_order", "model_order", "fingerprint_order",
         "protocol_order", "fold"]
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Protocol summary (mean ± std across folds)
# ---------------------------------------------------------------------------

def build_protocol_summary(per_fold: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "dataset_label", "model", "model_short",
                  "fingerprint", "protocol"]
    agg = (
        per_fold
        .groupby(group_cols, as_index=False)
        .agg(
            inner_mean=("inner_pr_auc", "mean"),
            inner_std=("inner_pr_auc", "std"),
            train_mean=("train_pr_auc", "mean"),
            train_std=("train_pr_auc", "std"),
            test_mean=("test_pr_auc", "mean"),
            test_std=("test_pr_auc", "std"),
            inner_test_gap_mean=("inner_test_gap", "mean"),
            inner_test_gap_std=("inner_test_gap", "std"),
            train_test_gap_mean=("train_test_gap", "mean"),
            train_test_gap_std=("train_test_gap", "std"),
        )
    )
    agg = _add_ordering_columns(agg)
    agg = agg.sort_values(
        ["dataset_order", "model_order", "fingerprint_order", "protocol_order"]
    ).reset_index(drop=True)
    return agg


# ---------------------------------------------------------------------------
# 3. Protocol delta table
# ---------------------------------------------------------------------------

def build_protocol_delta(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Compute deltas between Random shuffle and OOD holdout.

    Sign conventions (explicit column names):
      delta_inner_optimism = inner_random - inner_OOD   (positive = random inflated)
      delta_test_benefit   = test_OOD   - test_random   (positive = OOD better)
      delta_gap            = gap_random - gap_OOD       (positive = random more optimistic)
    """
    pivot = summary.pivot_table(
        index=["dataset", "dataset_label", "model", "model_short", "fingerprint"],
        columns="protocol",
        values=["inner_mean", "test_mean", "inner_test_gap_mean",
                "train_test_gap_mean"],
    )

    rows = []
    for idx in pivot.index:
        dataset, dataset_label, model, model_short, fingerprint = idx
        try:
            inner_ood = pivot.loc[idx, ("inner_mean", "OOD holdout")]
            inner_rnd = pivot.loc[idx, ("inner_mean", "Random shuffle")]
            test_ood = pivot.loc[idx, ("test_mean", "OOD holdout")]
            test_rnd = pivot.loc[idx, ("test_mean", "Random shuffle")]
            gap_ood = pivot.loc[idx, ("inner_test_gap_mean", "OOD holdout")]
            gap_rnd = pivot.loc[idx, ("inner_test_gap_mean", "Random shuffle")]
            train_gap_ood = pivot.loc[idx, ("train_test_gap_mean", "OOD holdout")]
            train_gap_rnd = pivot.loc[idx, ("train_test_gap_mean", "Random shuffle")]
        except KeyError:
            continue

        rows.append({
            "dataset": dataset,
            "dataset_label": dataset_label,
            "model": model,
            "model_short": model_short,
            "fingerprint": fingerprint,
            "inner_ood": inner_ood,
            "inner_random": inner_rnd,
            "delta_inner_optimism": inner_rnd - inner_ood,
            "test_ood": test_ood,
            "test_random": test_rnd,
            "delta_test_benefit": test_ood - test_rnd,
            "gap_ood": gap_ood,
            "gap_random": gap_rnd,
            "delta_gap": gap_rnd - gap_ood,
            "train_gap_ood": train_gap_ood,
            "train_gap_random": train_gap_rnd,
        })

    df = pd.DataFrame(rows)
    df = _add_ordering_columns(df)
    df = df.sort_values(
        ["dataset_order", "model_order", "fingerprint_order"]
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 4. Complexity tables
# ---------------------------------------------------------------------------

def build_complexity_all(registry: pd.DataFrame) -> pd.DataFrame:
    """Load complexity_fold_i.json + params for all experiments."""
    rows = []
    for _, exp in registry[registry["exists"]].iterrows():
        for fold in FOLDS:
            params_path = exp["result_dir"] / f"params_fold_{fold}.json"
            complexity_path = exp["result_dir"] / f"complexity_fold_{fold}.json"

            if not params_path.exists() or not complexity_path.exists():
                continue

            params = _read_json(params_path)
            complexity = _read_json(complexity_path)

            train_m = params.get("train_metrics", {})
            test_m = params.get("test_metrics", {})

            inner = params.get("inner_selection_score", np.nan)
            train_pr = train_m.get("pr_auc", np.nan)
            test_pr = test_m.get("pr_auc", np.nan)

            row = {
                "dataset": exp["dataset"],
                "dataset_label": exp["dataset_label"],
                "model": exp["model"],
                "model_short": exp["model_short"],
                "fingerprint": exp["fingerprint"],
                "protocol": exp["protocol"],
                "fold": fold,
                "inner_pr_auc": inner,
                "train_pr_auc": train_pr,
                "test_pr_auc": test_pr,
                "inner_test_gap": inner - test_pr,
                "train_test_gap": train_pr - test_pr,
            }
            # Add all complexity fields
            for key, value in complexity.items():
                row[key] = value

            rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df = _add_ordering_columns(df)
    df = df.sort_values(
        ["dataset_order", "model_order", "fingerprint_order",
         "protocol_order", "fold"]
    ).reset_index(drop=True)
    return df


def build_complexity_summary(complexity_all: pd.DataFrame) -> pd.DataFrame:
    """Aggregate complexity metrics across folds."""
    if len(complexity_all) == 0:
        return pd.DataFrame()

    indicator_cols = [
        "inner_pr_auc", "test_pr_auc", "inner_test_gap", "train_test_gap",
        "l2_norm", "l1_norm", "n_nonzero_coefficients", "sparsity",
        "approx_margin",
        "effective_depth", "n_nodes", "n_leaves", "n_features_used",
        "feature_min_depth_mean",
    ]
    available = [c for c in indicator_cols if c in complexity_all.columns]

    group_cols = ["dataset", "dataset_label", "model", "model_short",
                  "fingerprint", "protocol"]

    agg = (
        complexity_all
        .groupby(group_cols, as_index=False)[available]
        .agg(["mean", "std"])
    )
    # Flatten multi-level columns
    agg.columns = [
        "_".join([c for c in col if c]) for col in agg.columns
    ]
    agg = agg.reset_index()
    agg = _add_ordering_columns(agg)
    return agg


# ---------------------------------------------------------------------------
# 5. Feature importance tables
# ---------------------------------------------------------------------------

def _unify_importance(fi: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Create unified importance columns:
      importance_value  — the primary importance score
      importance_rank   — rank (1 = most important)
      importance_source — which raw column was used

    Decision Tree:
      primary = permutation_importance_mean (falls back to tree_importance)
    Logistic Regression / Linear SVM:
      primary = normalized_abs_importance (falls back to abs_weight)
    """
    fi = fi.copy()

    if model == "Decision Tree":
        if "permutation_importance_mean" in fi.columns and fi["permutation_importance_mean"].notna().any():
            fi["importance_value"] = fi["permutation_importance_mean"]
            fi["importance_source"] = "permutation_importance_mean"
            # Use existing permutation rank if available
            if "permutation_importance_rank" in fi.columns:
                fi["importance_rank"] = fi["permutation_importance_rank"]
            else:
                fi["importance_rank"] = (
                    fi["importance_value"]
                    .rank(ascending=False, method="first")
                    .astype(int)
                )
        elif "tree_importance" in fi.columns:
            fi["importance_value"] = fi["tree_importance"]
            fi["importance_source"] = "tree_importance"
            if "rank_tree_importance" in fi.columns:
                fi["importance_rank"] = fi["rank_tree_importance"]
            else:
                fi["importance_rank"] = (
                    fi["importance_value"]
                    .rank(ascending=False, method="first")
                    .astype(int)
                )
        else:
            fi["importance_value"] = np.nan
            fi["importance_rank"] = np.nan
            fi["importance_source"] = "none"
    else:
        # Linear models: LR, SVM
        if "normalized_abs_importance" in fi.columns and fi["normalized_abs_importance"].notna().any():
            fi["importance_value"] = fi["normalized_abs_importance"]
            fi["importance_source"] = "normalized_abs_importance"
        elif "abs_weight" in fi.columns:
            fi["importance_value"] = fi["abs_weight"]
            fi["importance_source"] = "abs_weight"
        else:
            fi["importance_value"] = np.nan
            fi["importance_source"] = "none"

        if "rank_abs_weight" in fi.columns and fi["importance_source"] != "none":
            fi["importance_rank"] = fi["rank_abs_weight"]
        else:
            fi["importance_rank"] = (
                fi["importance_value"]
                .rank(ascending=False, method="first")
                .astype(int)
            )

    fi["importance_rank"] = fi["importance_rank"].astype("Int64")
    return fi


def build_feature_importance_all(registry: pd.DataFrame) -> pd.DataFrame:
    """Load and unify feature importance CSVs for all experiments."""
    parts = []
    for _, exp in registry[registry["exists"]].iterrows():
        for fold in FOLDS:
            fi_path = exp["result_dir"] / f"feature_importance_fold_{fold}.csv"
            if not fi_path.exists():
                continue

            fi = pd.read_csv(fi_path)
            fi = _unify_importance(fi, exp["model"])

            fi["dataset"] = exp["dataset"]
            fi["dataset_label"] = exp["dataset_label"]
            fi["model"] = exp["model"]
            fi["model_short"] = exp["model_short"]
            fi["fingerprint"] = exp["fingerprint"]
            fi["protocol"] = exp["protocol"]
            fi["fold"] = fold

            parts.append(fi)

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    return df


def build_feature_topk(fi_all: pd.DataFrame) -> pd.DataFrame:
    """Extract top-k features per experiment × fold using unified rank."""
    if len(fi_all) == 0:
        return pd.DataFrame()

    group = ["dataset", "model", "fingerprint", "protocol", "fold"]
    parts = []
    for k in TOP_K_VALUES:
        topk = (
            fi_all
            .sort_values(group + ["importance_rank"])
            .groupby(group, as_index=False)
            .head(k)
            .copy()
        )
        topk["top_k"] = k
        parts.append(topk)

    return pd.concat(parts, ignore_index=True)


def build_feature_overlap(fi_topk: pd.DataFrame) -> pd.DataFrame:
    """Compute top-k overlap between OOD and Random for matched experiments."""
    if len(fi_topk) == 0:
        return pd.DataFrame()

    rows = []
    for dataset in fi_topk["dataset"].unique():
        for model in fi_topk["model"].unique():
            for fp in fi_topk["fingerprint"].unique():
                for fold in FOLDS:
                    for k in TOP_K_VALUES:
                        sub = fi_topk[
                            (fi_topk["dataset"] == dataset)
                            & (fi_topk["model"] == model)
                            & (fi_topk["fingerprint"] == fp)
                            & (fi_topk["fold"] == fold)
                            & (fi_topk["top_k"] == k)
                        ]
                        ood_feats = set(
                            sub.loc[sub["protocol"] == "OOD holdout",
                                    "feature_idx"].astype(int)
                        )
                        rnd_feats = set(
                            sub.loc[sub["protocol"] == "Random shuffle",
                                    "feature_idx"].astype(int)
                        )
                        if not ood_feats or not rnd_feats:
                            continue

                        n_overlap = len(ood_feats & rnd_feats)
                        overlap_frac = n_overlap / k

                        rows.append({
                            "dataset": dataset,
                            "model": model,
                            "fingerprint": fp,
                            "fold": fold,
                            "top_k": k,
                            "n_overlap": n_overlap,
                            "overlap_percent": round(100 * overlap_frac, 2),
                        })

    return pd.DataFrame(rows)


def build_feature_concentration(fi_all: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative importance captured by top-k features.

    Uses the unified importance_value column.
    """
    if len(fi_all) == 0:
        return pd.DataFrame()

    group = ["dataset", "model", "model_short", "fingerprint", "protocol",
             "fold"]
    rows = []

    for keys, sub in fi_all.groupby(group):
        dataset, model, model_short, fp, protocol, fold = keys
        imp = sub["importance_value"].fillna(0).values
        sorted_imp = np.sort(imp)[::-1]
        total = sorted_imp.sum()

        row = {
            "dataset": dataset,
            "model": model,
            "model_short": model_short,
            "fingerprint": fp,
            "protocol": protocol,
            "fold": fold,
            "n_features": len(imp),
            "n_nonzero": int(np.sum(imp > 0)),
            "total_importance": float(total),
        }

        for k in TOP_K_VALUES:
            cum = float(np.sum(sorted_imp[:k]))
            row[f"cumulative_top_{k}"] = cum
            if total > 0:
                row[f"fraction_top_{k}"] = round(cum / total, 6)
            else:
                row[f"fraction_top_{k}"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Results root: {RESULTS_ROOT}")
    print(f"Output dir:   {OUTPUT_DIR}\n")

    registry = _build_registry()

    # --- Protocol tables ---
    print("\n--- Building protocol per-fold table ---")
    per_fold = build_protocol_per_fold(registry)
    per_fold.to_csv(OUTPUT_DIR / "cross_dataset_protocol_per_fold.csv", index=False)
    print(f"  Saved: cross_dataset_protocol_per_fold.csv  ({len(per_fold)} rows)")

    print("--- Building protocol summary ---")
    summary = build_protocol_summary(per_fold)
    summary.to_csv(OUTPUT_DIR / "cross_dataset_protocol_summary.csv", index=False)
    print(f"  Saved: cross_dataset_protocol_summary.csv  ({len(summary)} rows)")

    print("--- Building protocol delta ---")
    delta = build_protocol_delta(summary)
    delta.to_csv(OUTPUT_DIR / "cross_dataset_protocol_delta.csv", index=False)
    print(f"  Saved: cross_dataset_protocol_delta.csv  ({len(delta)} rows)")

    # --- Complexity tables ---
    print("\n--- Building complexity tables ---")
    complexity_all = build_complexity_all(registry)
    complexity_all.to_csv(OUTPUT_DIR / "cross_dataset_complexity_all.csv", index=False)
    print(f"  Saved: cross_dataset_complexity_all.csv  ({len(complexity_all)} rows)")

    complexity_summary = build_complexity_summary(complexity_all)
    complexity_summary.to_csv(OUTPUT_DIR / "cross_dataset_complexity_summary.csv", index=False)
    print(f"  Saved: cross_dataset_complexity_summary.csv  ({len(complexity_summary)} rows)")

    # --- Feature importance tables ---
    print("\n--- Building feature importance tables ---")
    fi_all = build_feature_importance_all(registry)
    fi_all.to_csv(OUTPUT_DIR / "cross_dataset_feature_importance_all.csv", index=False)
    print(f"  Saved: cross_dataset_feature_importance_all.csv  ({len(fi_all)} rows)")

    fi_topk = build_feature_topk(fi_all)
    fi_topk.to_csv(OUTPUT_DIR / "cross_dataset_feature_topk.csv", index=False)
    print(f"  Saved: cross_dataset_feature_topk.csv  ({len(fi_topk)} rows)")

    overlap = build_feature_overlap(fi_topk)
    overlap.to_csv(OUTPUT_DIR / "cross_dataset_feature_overlap.csv", index=False)
    print(f"  Saved: cross_dataset_feature_overlap.csv  ({len(overlap)} rows)")

    concentration = build_feature_concentration(fi_all)
    concentration.to_csv(OUTPUT_DIR / "cross_dataset_feature_concentration.csv", index=False)
    print(f"  Saved: cross_dataset_feature_concentration.csv  ({len(concentration)} rows)")

    print(f"\nDone. All tables saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
