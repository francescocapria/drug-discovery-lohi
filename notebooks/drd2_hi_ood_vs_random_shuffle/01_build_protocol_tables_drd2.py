#!/usr/bin/env python
# coding: utf-8

# # DRD2-Hi — OOD Holdout vs Random Shuffle Protocol Comparison
# 
# This notebook compares two inner hyperparameter-selection protocols on the DRD2-Hi task:
# 
# 1. **OOD holdout**: the validation subset is chemically separated from the inner training subset.
# 2. **Random shuffle**: the validation subset is obtained by randomly splitting the outer training set.
# 
# The main experimental question is:
# 
# > Does in-distribution hyperparameter selection produce higher internal validation scores without improving final out-of-distribution test performance?
# 
# The analysis focuses on:
# 
# - inner validation PR-AUC;
# - inner train PR-AUC;
# - outer train PR-AUC after refitting;
# - final OOD test PR-AUC;
# - inner-to-test generalization gap;
# - train-to-test generalization gap;
# - selected hyperparameters across folds.
# 
# The models considered are:
# 
# - Decision Tree;
# - Logistic Regression;
# - Linear SVM.
# 
# The fingerprints considered are:
# 
# - ECFP4;
# - MACCS;
# - RDKit descriptors, where available.

# In[1]:


import json
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 120)

PROJECT_ROOT = Path("../..").resolve()
RESULTS_ROOT = PROJECT_ROOT / "results" / "hi" / "drd2"

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"RESULTS_ROOT: {RESULTS_ROOT}")
print(f"RESULTS_ROOT exists: {RESULTS_ROOT.exists()}")


# # Experiment registry

# In[2]:


EXPERIMENTS = [
    # ---------------------------------------------------------------------
    # Decision Tree
    # ---------------------------------------------------------------------
    {
        "model": "Decision Tree",
        "model_short": "DT",
        "fingerprint": "ECFP4",
        "protocol": "OOD holdout",
        "result_dir": "dt_drd2_hi_inner_ood_holdout_ecfp4",
    },
    {
        "model": "Decision Tree",
        "model_short": "DT",
        "fingerprint": "MACCS",
        "protocol": "OOD holdout",
        "result_dir": "dt_drd2_hi_inner_ood_holdout_maccs",
    },
    {
        "model": "Decision Tree",
        "model_short": "DT",
        "fingerprint": "ECFP4",
        "protocol": "Random shuffle",
        "result_dir": "dt_drd2_hi_random_shuffle_ecfp4",
    },
    {
        "model": "Decision Tree",
        "model_short": "DT",
        "fingerprint": "MACCS",
        "protocol": "Random shuffle",
        "result_dir": "dt_drd2_hi_random_shuffle_maccs",
    },

    # ---------------------------------------------------------------------
    # Logistic Regression
    # ---------------------------------------------------------------------
    {
        "model": "Logistic Regression",
        "model_short": "LR",
        "fingerprint": "ECFP4",
        "protocol": "OOD holdout",
        "result_dir": "lr_drd2_hi_inner_ood_holdout_ecfp4",
    },
    {
        "model": "Logistic Regression",
        "model_short": "LR",
        "fingerprint": "MACCS",
        "protocol": "OOD holdout",
        "result_dir": "lr_drd2_hi_inner_ood_holdout_maccs",
    },
    {
        "model": "Logistic Regression",
        "model_short": "LR",
        "fingerprint": "RDKit desc",
        "protocol": "OOD holdout",
        "result_dir": "lr_drd2_hi_inner_ood_holdout_rdkit_desc",
    },
    {
        "model": "Logistic Regression",
        "model_short": "LR",
        "fingerprint": "ECFP4",
        "protocol": "Random shuffle",
        "result_dir": "lr_drd2_hi_random_shuffle_ecfp4",
    },
    {
        "model": "Logistic Regression",
        "model_short": "LR",
        "fingerprint": "MACCS",
        "protocol": "Random shuffle",
        "result_dir": "lr_drd2_hi_random_shuffle_maccs",
    },
    {
        "model": "Logistic Regression",
        "model_short": "LR",
        "fingerprint": "RDKit desc",
        "protocol": "Random shuffle",
        "result_dir": "lr_drd2_hi_random_shuffle_rdkit_desc",
    },

    # ---------------------------------------------------------------------
    # Linear SVM
    # ---------------------------------------------------------------------
    {
        "model": "Linear SVM",
        "model_short": "SVM",
        "fingerprint": "ECFP4",
        "protocol": "OOD holdout",
        "result_dir": "svm_linear_drd2_hi_inner_ood_holdout_ecfp4",
    },
    {
        "model": "Linear SVM",
        "model_short": "SVM",
        "fingerprint": "MACCS",
        "protocol": "OOD holdout",
        "result_dir": "svm_linear_drd2_hi_inner_ood_holdout_maccs",
    },
    {
        "model": "Linear SVM",
        "model_short": "SVM",
        "fingerprint": "ECFP4",
        "protocol": "Random shuffle",
        "result_dir": "svm_linear_drd2_hi_random_shuffle_ecfp4",
    },
    {
        "model": "Linear SVM",
        "model_short": "SVM",
        "fingerprint": "MACCS",
        "protocol": "Random shuffle",
        "result_dir": "svm_linear_drd2_hi_random_shuffle_maccs",
    },
]

registry = pd.DataFrame(EXPERIMENTS)
registry["path"] = registry["result_dir"].apply(lambda d: RESULTS_ROOT / d)
registry["exists"] = registry["path"].apply(lambda p: p.exists())

registry


# In[3]:


missing = registry.loc[~registry["exists"], ["model", "fingerprint", "protocol", "path"]]

if len(missing) > 0:
    print("Missing result folders:")
    display(missing)
else:
    print("All registered result folders exist.")


# # Load params_fold_i.json into df_folds

# In[4]:


def load_params_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def safe_get_metric(metrics: dict, key: str):
    if metrics is None:
        return np.nan
    return metrics.get(key, np.nan)


rows = []

for exp in EXPERIMENTS:
    result_path = RESULTS_ROOT / exp["result_dir"]

    for fold in [1, 2, 3]:
        params_path = result_path / f"params_fold_{fold}.json"

        if not params_path.exists():
            print(f"Missing file: {params_path}")
            continue

        data = load_params_json(params_path)

        train_metrics = data.get("train_metrics", {})
        test_metrics = data.get("test_metrics", {})
        best_params = data.get("best_params", {})

        inner_pr_auc = data.get("inner_selection_score", np.nan)
        inner_train_pr_auc = data.get("inner_train_score", np.nan)
        train_pr_auc = safe_get_metric(train_metrics, "pr_auc")
        test_pr_auc = safe_get_metric(test_metrics, "pr_auc")

        row = {
            "model": exp["model"],
            "model_short": exp["model_short"],
            "fingerprint": exp["fingerprint"],
            "protocol": exp["protocol"],
            "result_dir": exp["result_dir"],
            "fold": fold,

            "inner_pr_auc": inner_pr_auc,
            "inner_train_pr_auc": inner_train_pr_auc,
            "train_pr_auc": train_pr_auc,
            "test_pr_auc": test_pr_auc,

            "inner_test_gap": inner_pr_auc - test_pr_auc,
            "train_test_gap": train_pr_auc - test_pr_auc,

            "best_params": best_params,
            "inner_split_strategy": data.get("inner_split_strategy", None),
            "time_seconds": data.get("time_seconds", np.nan),
        }

        rows.append(row)

df_folds = pd.DataFrame(rows)

order_model = {
    "Decision Tree": 0,
    "Logistic Regression": 1,
    "Linear SVM": 2,
}

order_fp = {
    "ECFP4": 0,
    "MACCS": 1,
    "RDKit desc": 2,
}

order_protocol = {
    "OOD holdout": 0,
    "Random shuffle": 1,
}

df_folds["model_order"] = df_folds["model"].map(order_model)
df_folds["fingerprint_order"] = df_folds["fingerprint"].map(order_fp)
df_folds["protocol_order"] = df_folds["protocol"].map(order_protocol)

df_folds = df_folds.sort_values(
    ["model_order", "fingerprint_order", "protocol_order", "fold"]
).reset_index(drop=True)

print(f"Loaded rows: {len(df_folds)}")
df_folds.head(2)


# # Per-fold table

# In[5]:


per_fold_table = df_folds[
    [
        "model",
        "fingerprint",
        "protocol",
        "fold",
        "inner_pr_auc",
        "inner_train_pr_auc",
        "train_pr_auc",
        "test_pr_auc",
        "inner_test_gap",
        "train_test_gap",
    ]
].copy()

numeric_cols = [
    "inner_pr_auc",
    "inner_train_pr_auc",
    "train_pr_auc",                                                                                         
    "test_pr_auc",
    "inner_test_gap",
    "train_test_gap",
]

per_fold_table[numeric_cols] = per_fold_table[numeric_cols].round(4)

per_fold_table


# # Aggregated summary table

# In[6]:


def mean_std_string(mean_value, std_value, digits=4):
    if pd.isna(mean_value):
        return ""
    if pd.isna(std_value):
        return f"{mean_value:.{digits}f}"
    return f"{mean_value:.{digits}f} ± {std_value:.{digits}f}"


summary_numeric = (
    df_folds
    .groupby(["model", "model_short", "fingerprint", "protocol"], as_index=False)
    .agg(
        inner_mean=("inner_pr_auc", "mean"),
        inner_std=("inner_pr_auc", "std"),
        inner_train_mean=("inner_train_pr_auc", "mean"),
        inner_train_std=("inner_train_pr_auc", "std"),
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

summary_numeric["model_order"] = summary_numeric["model"].map(order_model)
summary_numeric["fingerprint_order"] = summary_numeric["fingerprint"].map(order_fp)
summary_numeric["protocol_order"] = summary_numeric["protocol"].map(order_protocol)

summary_numeric = summary_numeric.sort_values(
    ["model_order", "fingerprint_order", "protocol_order"]
).reset_index(drop=True)

summary_table = summary_numeric[
    ["model", "fingerprint", "protocol"]
].copy()

summary_table["Inner PR-AUC"] = summary_numeric.apply(
    lambda r: mean_std_string(r["inner_mean"], r["inner_std"]), axis=1
)

summary_table["Inner train PR-AUC"] = summary_numeric.apply(
    lambda r: mean_std_string(r["inner_train_mean"], r["inner_train_std"]), axis=1
)

summary_table["Train PR-AUC"] = summary_numeric.apply(
    lambda r: mean_std_string(r["train_mean"], r["train_std"]), axis=1
)

summary_table["Final OOD test PR-AUC"] = summary_numeric.apply(
    lambda r: mean_std_string(r["test_mean"], r["test_std"]), axis=1
)

summary_table["Inner-test gap"] = summary_numeric.apply(
    lambda r: mean_std_string(r["inner_test_gap_mean"], r["inner_test_gap_std"]), axis=1
)

summary_table["Train-test gap"] = summary_numeric.apply(
    lambda r: mean_std_string(r["train_test_gap_mean"], r["train_test_gap_std"]), axis=1
)

summary_table


# # Delta table

# In[7]:


pivot_summary = summary_numeric.pivot_table(
    index=["model", "model_short", "fingerprint"],
    columns="protocol",
    values=[
        "inner_mean",
        "test_mean",
        "inner_test_gap_mean",
        "train_test_gap_mean",
    ],
)

delta_rows = []

for idx, row in pivot_summary.iterrows():
    model, model_short, fingerprint = idx

    try:
        random_inner = row[("inner_mean", "Random shuffle")]
        ood_inner = row[("inner_mean", "OOD holdout")]

        random_test = row[("test_mean", "Random shuffle")]
        ood_test = row[("test_mean", "OOD holdout")]

        random_inner_gap = row[("inner_test_gap_mean", "Random shuffle")]
        ood_inner_gap = row[("inner_test_gap_mean", "OOD holdout")]

        random_train_gap = row[("train_test_gap_mean", "Random shuffle")]
        ood_train_gap = row[("train_test_gap_mean", "OOD holdout")]

    except KeyError:
        continue

    delta_rows.append({
        "model": model,
        "model_short": model_short,
        "fingerprint": fingerprint,

        "ood_inner_mean": ood_inner,
        "random_inner_mean": random_inner,
        "delta_inner": random_inner - ood_inner,

        "ood_test_mean": ood_test,
        "random_test_mean": random_test,
        "delta_test": random_test - ood_test,

        "ood_inner_test_gap": ood_inner_gap,
        "random_inner_test_gap": random_inner_gap,
        "delta_inner_test_gap": random_inner_gap - ood_inner_gap,

        "ood_train_test_gap": ood_train_gap,
        "random_train_test_gap": random_train_gap,
        "delta_train_test_gap": random_train_gap - ood_train_gap,
    })

delta_table = pd.DataFrame(delta_rows)

delta_table["model_order"] = delta_table["model"].map(order_model)
delta_table["fingerprint_order"] = delta_table["fingerprint"].map(order_fp)

delta_table = delta_table.sort_values(
    ["model_order", "fingerprint_order"]
).reset_index(drop=True)

delta_display = delta_table[
    [
        "model",
        "fingerprint",
        "ood_inner_mean",
        "random_inner_mean",
        "delta_inner",
        "ood_test_mean",
        "random_test_mean",
        "delta_test",
        "ood_inner_test_gap",
        "random_inner_test_gap",
        "delta_inner_test_gap",
        "delta_train_test_gap",
    ]
].copy()

delta_numeric_cols = delta_display.select_dtypes(include=[np.number]).columns
delta_display[delta_numeric_cols] = delta_display[delta_numeric_cols].round(4)

delta_display


# # Hyperparameter summary tables

# In[8]:


def flatten_best_params(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        base = {
            "model": row["model"],
            "fingerprint": row["fingerprint"],
            "protocol": row["protocol"],
            "fold": row["fold"],
            "inner_pr_auc": row["inner_pr_auc"],
            "test_pr_auc": row["test_pr_auc"],
        }

        params = row["best_params"]

        if isinstance(params, dict):
            for key, value in params.items():
                clean_key = key.replace("model__", "")
                base[clean_key] = value

        rows.append(base)

    return pd.DataFrame(rows)


hp_df = flatten_best_params(df_folds)

hp_df["model_order"] = hp_df["model"].map(order_model)
hp_df["fingerprint_order"] = hp_df["fingerprint"].map(order_fp)
hp_df["protocol_order"] = hp_df["protocol"].map(order_protocol)

hp_df = hp_df.sort_values(
    ["model_order", "fingerprint_order", "protocol_order", "fold"]
).reset_index(drop=True)

for col in ["inner_pr_auc", "test_pr_auc"]:
    hp_df[col] = hp_df[col].round(4)

hp_df.head()


# # Logistic Regression hyperparameters

# In[9]:


lr_hp_table = hp_df.loc[
    hp_df["model"] == "Logistic Regression",
    [
        "protocol",
        "fingerprint",
        "fold",
        "C",
        "class_weight",
        "l1_ratio",
        "inner_pr_auc",
        "test_pr_auc",
    ]
].copy()

lr_hp_table = lr_hp_table.sort_values(
    ["fingerprint", "protocol", "fold"]
).reset_index(drop=True)

lr_hp_table


# # Svm liner hyperparameters

# In[10]:


svm_hp_table = hp_df.loc[
    hp_df["model"] == "Linear SVM",
    [
        "protocol",
        "fingerprint",
        "fold",
        "C",
        "class_weight",
        "inner_pr_auc",
        "test_pr_auc",
    ]
].copy()

svm_hp_table = svm_hp_table.sort_values(
    ["fingerprint", "protocol", "fold"]
).reset_index(drop=True)

svm_hp_table


# # Decision Tree hyperparameters

# In[11]:


dt_hp_table = hp_df.loc[
    hp_df["model"] == "Decision Tree",
    [
        "protocol",
        "fingerprint",
        "fold",
        "criterion",
        "max_depth",
        "max_features",
        "min_samples_leaf",
        "min_samples_split",
        "ccp_alpha",
        "class_weight",
        "inner_pr_auc",
        "test_pr_auc",
    ]
].copy()

dt_hp_table = dt_hp_table.sort_values(
    ["fingerprint", "protocol", "fold"]
).reset_index(drop=True)

dt_hp_table


# # Compact hyperparameter set summary

# In[12]:


def unique_values_as_string(series: pd.Series) -> str:
    values = []
    for value in series.dropna().tolist():
        if value not in values:
            values.append(value)
    return "{" + ", ".join(str(v) for v in values) + "}"


hp_set_summary_rows = []

for model_name, model_df in hp_df.groupby("model"):
    param_cols = [
        c for c in model_df.columns
        if c not in [
            "model",
            "fingerprint",
            "protocol",
            "fold",
            "inner_pr_auc",
            "test_pr_auc",
            "model_order",
            "fingerprint_order",
            "protocol_order",
        ]
    ]

    for protocol, protocol_df in model_df.groupby("protocol"):
        row = {
            "model": model_name,
            "protocol": protocol,
        }

        for col in param_cols:
            if protocol_df[col].notna().any():
                row[col] = unique_values_as_string(protocol_df[col])

        hp_set_summary_rows.append(row)

hp_set_summary = pd.DataFrame(hp_set_summary_rows)
hp_set_summary["model_order"] = hp_set_summary["model"].map(order_model)
hp_set_summary["protocol_order"] = hp_set_summary["protocol"].map(order_protocol)

hp_set_summary = hp_set_summary.sort_values(
    ["model_order", "protocol_order"]
).drop(columns=["model_order", "protocol_order"]).reset_index(drop=True)

hp_set_summary


# # Save processed tables for the plotting notebook

# In[13]:


TASK = "hi"
DATASET = "drd2"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "results_ood_vs_random_shuffle"
    / TASK
    / DATASET
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Main protocol comparison tables
df_folds.to_csv(
    OUTPUT_DIR / "protocol_per_fold.csv",
    index=False,
)

summary_numeric.to_csv(
    OUTPUT_DIR / "protocol_summary_numeric.csv",
    index=False,
)

summary_table.to_csv(
    OUTPUT_DIR / "protocol_summary_display.csv",
    index=False,
)

delta_table.to_csv(
    OUTPUT_DIR / "protocol_delta.csv",
    index=False,
)

# Hyperparameter tables
hp_df.to_csv(
    OUTPUT_DIR / "hyperparameters_all.csv",
    index=False,
)

lr_hp_table.to_csv(
    OUTPUT_DIR / "hyperparameters_lr.csv",
    index=False,
)

svm_hp_table.to_csv(
    OUTPUT_DIR / "hyperparameters_svm.csv",
    index=False,
)

dt_hp_table.to_csv(
    OUTPUT_DIR / "hyperparameters_dt.csv",
    index=False,
)

hp_set_summary.to_csv(
    OUTPUT_DIR / "hyperparameters_set_summary.csv",
    index=False,
)

print("Saved processed files in:")
print(OUTPUT_DIR)

print("\nFiles saved:")
for file in sorted(OUTPUT_DIR.glob("*.csv")):
    print(f"- {file.name}")

