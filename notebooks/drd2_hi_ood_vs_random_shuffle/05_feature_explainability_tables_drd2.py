#!/usr/bin/env python
# coding: utf-8

# # Feature Explainability Tables for DRD2-Hi
# 
# This notebook builds the tabular datasets required for feature-level explainability.
# 
#  The goal is to compare whether the two inner validation protocols (OOD holdout vs random shuffle)
#  lead to models that rely on similar or different features.
# 
#  Models: Decision Tree, Logistic Regression, Linear SVM.
#  Fingerprints: ECFP4, MACCS, RDKit descriptors (LR only).
# 
#  **Outputs** (saved to `results/results_ood_vs_random_shuffle/hi/drd2/`):
#  - `feature_importance_all.csv`
#  - `feature_topk.csv`
#  - `feature_overlap_protocol.csv`
#  - `feature_stability_intra_protocol.csv`
#  - `feature_importance_summary.csv`
#  - `local_molecule_candidates.csv`
#  - `local_feature_contributions.csv`

# In[7]:


import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

PROJECT_ROOT = Path("../..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from utils.io_utils import load_fold
from utils.fingerprints import compute_fingerprints
from utils.explainability import (
    extract_base_model,
    transform_features_if_pipeline,
    compute_linear_contributions,
    compute_topk_overlap,
)

TASK = "hi"
DATASET = "drd2"
FOLDS = [1, 2, 3]
TOP_K_VALUES = [10, 20, 50]
MAX_LOCAL_CANDIDATES_PER_GROUP = 4
TOP_LOCAL_FEATURES = 20

RESULTS_ROOT = PROJECT_ROOT / "results" / TASK / DATASET
OUT_DIR = PROJECT_ROOT / "results" / "results_ood_vs_random_shuffle" / TASK / DATASET
OUT_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", 200)


# In[8]:


experiments = [
    # Decision Tree
    ("Decision Tree", "dt", "ecfp4", "OOD holdout", "dt_drd2_hi_inner_ood_holdout_ecfp4"),
    ("Decision Tree", "dt", "ecfp4", "Random shuffle", "dt_drd2_hi_random_shuffle_ecfp4"),
    ("Decision Tree", "dt", "maccs", "OOD holdout", "dt_drd2_hi_inner_ood_holdout_maccs"),
    ("Decision Tree", "dt", "maccs", "Random shuffle", "dt_drd2_hi_random_shuffle_maccs"),
    # Logistic Regression
    ("Logistic Regression", "lr", "ecfp4", "OOD holdout", "lr_drd2_hi_inner_ood_holdout_ecfp4"),
    ("Logistic Regression", "lr", "ecfp4", "Random shuffle", "lr_drd2_hi_random_shuffle_ecfp4"),
    ("Logistic Regression", "lr", "maccs", "OOD holdout", "lr_drd2_hi_inner_ood_holdout_maccs"),
    ("Logistic Regression", "lr", "maccs", "Random shuffle", "lr_drd2_hi_random_shuffle_maccs"),
    ("Logistic Regression", "lr", "rdkit_desc", "OOD holdout", "lr_drd2_hi_inner_ood_holdout_rdkit_desc"),
    ("Logistic Regression", "lr", "rdkit_desc", "Random shuffle", "lr_drd2_hi_random_shuffle_rdkit_desc"),
    # Linear SVM
    ("SVM linear", "svm", "ecfp4", "OOD holdout", "svm_linear_drd2_hi_inner_ood_holdout_ecfp4"),
    ("SVM linear", "svm", "ecfp4", "Random shuffle", "svm_linear_drd2_hi_random_shuffle_ecfp4"),
    ("SVM linear", "svm", "maccs", "OOD holdout", "svm_linear_drd2_hi_inner_ood_holdout_maccs"),
    ("SVM linear", "svm", "maccs", "Random shuffle", "svm_linear_drd2_hi_random_shuffle_maccs"),
]

registry = pd.DataFrame(experiments, columns=["model", "model_short", "fingerprint", "protocol", "dir_name"])
registry["result_dir"] = registry["dir_name"].apply(lambda d: RESULTS_ROOT / d)
registry["exists"] = registry["result_dir"].apply(lambda p: p.exists())


# In[9]:


required_files = [f"{stem}_{fold}.{ext}" for fold in FOLDS
                  for stem, ext in [("params_fold", "json"), ("model_fold", "joblib"),
                                     ("feature_importance_fold", "csv"), ("test", "csv")]]

missing = []
for _, row in registry.iterrows():
    for f in required_files:
        if not (row["result_dir"] / f).exists():
            missing.append(f"{row['dir_name']}/{f}")

if missing:
    print(f"WARNING: {len(missing)} missing files:")
    for m in missing[:10]:
        print(f"  - {m}")
else:
    print(f"All files present for {len(registry)} experiments × {len(FOLDS)} folds.")

registry[["model", "fingerprint", "protocol", "exists"]]


# In[10]:


all_fi = []

for _, exp in registry.iterrows():
    for fold in FOLDS:
        fi = pd.read_csv(exp["result_dir"] / f"feature_importance_fold_{fold}.csv")

        with open(exp["result_dir"] / f"params_fold_{fold}.json") as f:
            params = json.load(f)

        # Metadata
        fi["model"] = exp["model"]
        fi["model_short"] = exp["model_short"]
        fi["fingerprint"] = exp["fingerprint"]
        fi["protocol"] = exp["protocol"]
        fi["fold"] = fold
        fi["inner_pr_auc"] = params.get("inner_selection_score", np.nan)
        fi["inner_train_pr_auc"] = params.get("inner_train_score", np.nan)
        fi["train_pr_auc"] = params.get("train_metrics", {}).get("pr_auc", np.nan)
        fi["test_pr_auc"] = params.get("test_metrics", {}).get("pr_auc", np.nan)

        # Standardize importance columns across model types
        fi["importance"] = fi.get("normalized_abs_importance", fi.get("normalized_tree_importance", np.nan))
        fi["raw_importance"] = fi.get("raw_weight", fi.get("tree_importance", np.nan))
        fi["abs_importance"] = fi.get("abs_weight", fi.get("tree_importance", pd.Series(dtype=float)).abs())
        fi["rank"] = fi.get("rank_abs_weight", fi.get("rank_tree_importance", np.nan))

        # Fill any remaining NaN in rank
        if fi["rank"].isna().any():
            fi = fi.sort_values("abs_importance", ascending=False).reset_index(drop=True)
            fi["rank"] = np.arange(1, len(fi) + 1)
        else:
            fi["rank"] = fi["rank"].astype(int)

        all_fi.append(fi)

feature_importance_all = pd.concat(all_fi, ignore_index=True)

feature_importance_all.to_csv(OUT_DIR / "feature_importance_all.csv", index=False)
print(f"Saved feature_importance_all.csv — {feature_importance_all.shape}")


# In[13]:


GROUP = ["model", "model_short", "fingerprint", "protocol", "fold"]

# --- Top-k ---
topk_parts = []
for k in TOP_K_VALUES:
    topk = (
        feature_importance_all
        .sort_values(GROUP + ["rank"])
        .groupby(GROUP, as_index=False)
        .head(k)
        .copy()
    )
    topk["top_k"] = k
    topk_parts.append(topk)

feature_topk = pd.concat(topk_parts, ignore_index=True)
feature_topk.to_csv(OUT_DIR / "feature_topk.csv", index=False)
print(f"Saved feature_topk.csv — {feature_topk.shape}")


# In[14]:


overlap_rows = []
for model in feature_importance_all["model"].unique():
    for fp in feature_importance_all["fingerprint"].unique():
        for fold in FOLDS:
            for k in TOP_K_VALUES:
                sub = feature_topk[
                    (feature_topk["model"] == model)
                    & (feature_topk["fingerprint"] == fp)
                    & (feature_topk["fold"] == fold)
                    & (feature_topk["top_k"] == k)
                ]
                ood_feats = set(sub.loc[sub["protocol"] == "OOD holdout", "feature_idx"].astype(int))
                rnd_feats = set(sub.loc[sub["protocol"] == "Random shuffle", "feature_idx"].astype(int))

                if not ood_feats or not rnd_feats:
                    continue

                ov = compute_topk_overlap(ood_feats, rnd_feats, k)
                overlap_rows.append({
                    "model": model, "fingerprint": fp, "fold": fold,
                    "top_k": k, "n_overlap": int(ov * k),
                    "overlap": round(ov, 4), "overlap_percent": round(100 * ov, 2),
                })

feature_overlap_protocol = pd.DataFrame(overlap_rows)
feature_overlap_protocol.to_csv(OUT_DIR / "feature_overlap_protocol.csv", index=False)
print(f"Saved feature_overlap_protocol.csv — {feature_overlap_protocol.shape}")


# In[15]:


stability_rows = []

for model in feature_importance_all["model"].unique():
    for fp in feature_importance_all["fingerprint"].unique():
        for protocol in feature_importance_all["protocol"].unique():
            for fold_a, fold_b in combinations(FOLDS, 2):
                for k in TOP_K_VALUES:
                    sub = feature_topk[
                        (feature_topk["model"] == model)
                        & (feature_topk["fingerprint"] == fp)
                        & (feature_topk["protocol"] == protocol)
                        & (feature_topk["top_k"] == k)
                    ]
                    feats_a = set(sub.loc[sub["fold"] == fold_a, "feature_idx"].astype(int))
                    feats_b = set(sub.loc[sub["fold"] == fold_b, "feature_idx"].astype(int))

                    if not feats_a or not feats_b:
                        continue

                    ov = compute_topk_overlap(feats_a, feats_b, k)
                    stability_rows.append({
                        "model": model, "fingerprint": fp, "protocol": protocol,
                        "fold_a": fold_a, "fold_b": fold_b,
                        "fold_pair": f"{fold_a} vs {fold_b}",
                        "top_k": k, "n_overlap": int(ov * k),
                        "overlap": round(ov, 4), "overlap_percent": round(100 * ov, 2),
                    })

feature_stability = pd.DataFrame(stability_rows)
feature_stability.to_csv(OUT_DIR / "feature_stability_intra_protocol.csv", index=False)
print(f"Saved feature_stability_intra_protocol.csv — {feature_stability.shape}")


# In[16]:


summary_rows = []

for keys, sub in feature_importance_all.groupby(GROUP):
    model, model_short, fp, protocol, fold = keys
    imp = sub["importance"].fillna(0).values

    row = {
        "model": model, "model_short": model_short, "fingerprint": fp,
        "protocol": protocol, "fold": fold,
        "n_features": len(imp),
        "n_nonzero": int(np.sum(imp > 0)),
        "mean_importance": float(np.mean(imp)),
        "max_importance": float(np.max(imp)) if len(imp) > 0 else np.nan,
    }

    # Cumulative importance at different k
    sorted_imp = np.sort(imp)[::-1]
    for k in TOP_K_VALUES:
        row[f"cumulative_top_{k}"] = float(np.sum(sorted_imp[:k]))

    # Decision Tree minimum depth stats
    if "minimum_depth" in sub.columns:
        depths = sub["minimum_depth"].dropna()
        if len(depths) > 0:
            row["min_depth_mean"] = float(depths.mean())
            row["min_depth_std"] = float(depths.std())
            row["min_depth_min"] = int(depths.min())
            row["min_depth_max"] = int(depths.max())

    summary_rows.append(row)

feature_importance_summary = pd.DataFrame(summary_rows)
feature_importance_summary.to_csv(OUT_DIR / "feature_importance_summary.csv", index=False)
print(f"Saved feature_importance_summary.csv — {feature_importance_summary.shape}")


# In[17]:


candidate_rows = []

for _, exp in registry.iterrows():
    for fold in FOLDS:
        test_df = pd.read_csv(exp["result_dir"] / f"test_{fold}.csv")

        preds = test_df["preds"].values
        y_true = test_df["value"].astype(bool).values

        # SVM decision scores can be outside [0,1]
        threshold = 0.5 if (np.nanmin(preds) >= 0 and np.nanmax(preds) <= 1) else 0.0
        correct = (preds >= threshold) == y_true

        test_df["protocol"] = exp["protocol"]
        test_df["model"] = exp["model"]
        test_df["fingerprint"] = exp["fingerprint"]
        test_df["fold"] = fold
        test_df["correct"] = correct
        test_df["threshold"] = threshold

        candidate_rows.append(test_df)

all_preds = pd.concat(candidate_rows, ignore_index=True)

# For each model × fp × fold, find molecules where protocols disagree
local_candidates = []

for (model, fp, fold), group in all_preds.groupby(["model", "fingerprint", "fold"]):
    ood = group[group["protocol"] == "OOD holdout"].set_index("smiles")
    rnd = group[group["protocol"] == "Random shuffle"].set_index("smiles")

    common = ood.index.intersection(rnd.index)
    if len(common) == 0:
        continue

    merged = pd.DataFrame({
        "pred_ood": ood.loc[common, "preds"],
        "pred_random": rnd.loc[common, "preds"],
        "correct_ood": ood.loc[common, "correct"],
        "correct_random": rnd.loc[common, "correct"],
        "true_label": ood.loc[common, "value"],
    })
    merged["pred_diff_abs"] = (merged["pred_ood"] - merged["pred_random"]).abs()

    categories = {
        "both_correct": merged["correct_ood"] & merged["correct_random"],
        "both_wrong": ~merged["correct_ood"] & ~merged["correct_random"],
        "ood_correct_random_wrong": merged["correct_ood"] & ~merged["correct_random"],
        "ood_wrong_random_correct": ~merged["correct_ood"] & merged["correct_random"],
    }

    for cat_name, mask in categories.items():
        subset = merged[mask]
        if len(subset) == 0:
            continue
        best = subset.sort_values("pred_diff_abs", ascending=False).head(1)
        for smiles, row in best.iterrows():
            local_candidates.append({
                "model": model, "fingerprint": fp, "fold": fold,
                "category": cat_name, "smiles": smiles,
                "true_label": row["true_label"],
                "pred_ood": row["pred_ood"], "pred_random": row["pred_random"],
                "correct_ood": row["correct_ood"], "correct_random": row["correct_random"],
                "pred_diff_abs": row["pred_diff_abs"],
            })

local_molecule_candidates = pd.DataFrame(local_candidates)
local_molecule_candidates.to_csv(OUT_DIR / "local_molecule_candidates.csv", index=False)
print(f"Saved local_molecule_candidates.csv — {local_molecule_candidates.shape}")



# In[18]:


local_contributions = []
linear_models = {"Logistic Regression", "SVM linear"}

# Group candidates by (result_dir, fold) to load each model only once
for _, exp in registry[registry["model"].isin(linear_models)].iterrows():
    for fold in FOLDS:
        # Get candidates for this experiment
        cands = local_molecule_candidates[
            (local_molecule_candidates["model"] == exp["model"])
            & (local_molecule_candidates["fingerprint"] == exp["fingerprint"])
            & (local_molecule_candidates["fold"] == fold)
        ].head(MAX_LOCAL_CANDIDATES_PER_GROUP)

        if len(cands) == 0:
            continue

        # Load model ONCE for all candidates in this group
        model = joblib.load(exp["result_dir"] / f"model_fold_{fold}.joblib")

        for _, cand in cands.iterrows():
            X_single = compute_fingerprints([cand["smiles"]], exp["fingerprint"], cache_path=None)
            x, coef, contribs = compute_linear_contributions(model, X_single)

            if contribs is None:
                continue

            n = len(contribs)
            local_df = pd.DataFrame({
                "feature_idx": np.arange(n, dtype=int),
                "feature_name": [f"{exp['fingerprint']}_feature_{i}" for i in range(n)],
                "feature_value": x,
                "feature_weight": coef,
                "contribution": contribs,
                "abs_contribution": np.abs(contribs),
            })

            # Keep only top contributing features
            local_df = local_df.nlargest(TOP_LOCAL_FEATURES, "abs_contribution").copy()
            local_df["direction"] = np.where(
                local_df["contribution"] > 0, "toward_active",
                np.where(local_df["contribution"] < 0, "toward_inactive", "zero")
            )

            # Add metadata
            for col in ["model", "fingerprint", "protocol", "fold"]:
                local_df[col] = exp[col] if col != "fold" else fold
            local_df["model_short"] = exp["model_short"]
            local_df["smiles"] = cand["smiles"]
            local_df["category"] = cand["category"]
            local_df["true_label"] = cand["true_label"]
            local_df["pred_ood"] = cand["pred_ood"]
            local_df["pred_random"] = cand["pred_random"]

            local_contributions.append(local_df)

local_feature_contributions = pd.concat(local_contributions, ignore_index=True) if local_contributions else pd.DataFrame()
local_feature_contributions.to_csv(OUT_DIR / "local_feature_contributions.csv", index=False)
print(f"Saved local_feature_contributions.csv — {local_feature_contributions.shape}")


# In[19]:


outputs = [
    "feature_importance_all.csv",
    "feature_topk.csv",
    "feature_overlap_protocol.csv",
    "feature_stability_intra_protocol.csv",
    "feature_importance_summary.csv",
    "local_molecule_candidates.csv",
    "local_feature_contributions.csv",
]

for f in outputs:
    path = OUT_DIR / f
    status = "OK" if path.exists() else "MISSING"
    size = f"({path.stat().st_size / 1024:.0f} KB)" if path.exists() else ""
    print(f"  {f}: {status} {size}")

