#!/usr/bin/env python
# coding: utf-8

# # DRD2 Hi — Model complexity analysis
# 
# This notebook analyses whether the internal validation protocol used for hyperparameter selection changes the complexity of the selected models.
# 
# The comparison is between:
# 
# - OOD holdout inner validation
# - Random shuffle inner validation
# 
# The main question is:
# 
# **Does random shuffle select models that are more complex and less calibrated for final OOD test generalization?**
# 
# The analysis uses the saved per-fold artifacts:
# 
# - `params_fold_i.json`
# - `complexity_fold_i.json`

# In[3]:


from pathlib import Path
import json

import numpy as np
import pandas as pd


# In[4]:


PROJECT_ROOT = Path("../..").resolve()

RAW_RESULTS_DIR = PROJECT_ROOT / "results" / "hi" / "drd2"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "results_ood_vs_random_shuffle"
    / "hi"
    / "drd2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Raw results:", RAW_RESULTS_DIR)
print("Output dir:", OUTPUT_DIR)


# ## Experiment registry
# 
# The registry defines which result folders correspond to each combination of:
# 
# - model
# - fingerprint
# - protocol

# In[5]:


EXPERIMENTS = [
    # Decision Tree
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
        "fingerprint": "ECFP4",
        "protocol": "Random shuffle",
        "result_dir": "dt_drd2_hi_random_shuffle_ecfp4",
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
        "fingerprint": "MACCS",
        "protocol": "Random shuffle",
        "result_dir": "dt_drd2_hi_random_shuffle_maccs",
    },

    # Logistic Regression
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
        "fingerprint": "ECFP4",
        "protocol": "Random shuffle",
        "result_dir": "lr_drd2_hi_random_shuffle_ecfp4",
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
        "fingerprint": "MACCS",
        "protocol": "Random shuffle",
        "result_dir": "lr_drd2_hi_random_shuffle_maccs",
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
        "fingerprint": "RDKit desc",
        "protocol": "Random shuffle",
        "result_dir": "lr_drd2_hi_random_shuffle_rdkit_desc",
    },

    # Linear SVM
    {
        "model": "SVM linear",
        "model_short": "SVM",
        "fingerprint": "ECFP4",
        "protocol": "OOD holdout",
        "result_dir": "svm_linear_drd2_hi_inner_ood_holdout_ecfp4",
    },
    {
        "model": "SVM linear",
        "model_short": "SVM",
        "fingerprint": "ECFP4",
        "protocol": "Random shuffle",
        "result_dir": "svm_linear_drd2_hi_random_shuffle_ecfp4",
    },
    {
        "model": "SVM linear",
        "model_short": "SVM",
        "fingerprint": "MACCS",
        "protocol": "OOD holdout",
        "result_dir": "svm_linear_drd2_hi_inner_ood_holdout_maccs",
    },
    {
        "model": "SVM linear",
        "model_short": "SVM",
        "fingerprint": "MACCS",
        "protocol": "Random shuffle",
        "result_dir": "svm_linear_drd2_hi_random_shuffle_maccs",
    },
]


# In[6]:


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def safe_get(dct, key, default=np.nan):
    return dct.get(key, default)


# In[7]:


def load_complexity_rows(experiments, raw_results_dir):
    rows = []

    for exp in experiments:
        result_dir = raw_results_dir / exp["result_dir"]

        if not result_dir.exists():
            print(f"Missing directory: {result_dir}")
            continue

        for fold in [1, 2, 3]:
            params_path = result_dir / f"params_fold_{fold}.json"
            complexity_path = result_dir / f"complexity_fold_{fold}.json"

            if not params_path.exists():
                print(f"Missing params file: {params_path}")
                continue

            if not complexity_path.exists():
                print(f"Missing complexity file: {complexity_path}")
                continue

            params = read_json(params_path)
            complexity = read_json(complexity_path)

            train_metrics = params.get("train_metrics", {})
            test_metrics = params.get("test_metrics", {})

            inner_pr_auc = params.get("inner_selection_score", np.nan)
            inner_train_pr_auc = params.get("inner_train_score", np.nan)
            train_pr_auc = train_metrics.get("pr_auc", np.nan)
            test_pr_auc = test_metrics.get("pr_auc", np.nan)

            row = {
                "model": exp["model"],
                "model_short": exp["model_short"],
                "fingerprint": exp["fingerprint"],
                "protocol": exp["protocol"],
                "result_dir": exp["result_dir"],
                "fold": fold,

                # Scores
                "inner_pr_auc": inner_pr_auc,
                "inner_train_pr_auc": inner_train_pr_auc,
                "train_pr_auc": train_pr_auc,
                "test_pr_auc": test_pr_auc,

                # Gaps
                "train_inner_gap": train_pr_auc - inner_pr_auc,
                "inner_test_gap": inner_pr_auc - test_pr_auc,
                "train_test_gap": train_pr_auc - test_pr_auc,
            }

            # Add all complexity fields
            for key, value in complexity.items():
                row[key] = value

            rows.append(row)

    return pd.DataFrame(rows)


# In[9]:


complexity_all = load_complexity_rows(EXPERIMENTS, RAW_RESULTS_DIR)

complexity_all = complexity_all.sort_values(
    ["model_short", "fingerprint", "protocol", "fold"]
).reset_index(drop=True)

complexity_all.head(3)


# In[10]:


print("Rows loaded:", len(complexity_all))
print("Expected rows:", len(EXPERIMENTS) * 3)

complexity_all[
    [
        "model",
        "fingerprint",
        "protocol",
        "fold",
        "inner_pr_auc",
        "train_pr_auc",
        "test_pr_auc",
        "inner_test_gap",
        "train_test_gap",
        "model_class",
    ]
]


# ## Logistic Regression complexity table
# 
# For Logistic Regression complexity is described by:
# 
# - selected `C` - larger --> weaker regularization
# - selected `l1_ratio`
# - number of non-zero coefficients
# - sparsity
# - L1 norm of the coefficient vector
# - L2 norm of the coefficient vector - 

# In[11]:


lr_cols = [
    "model",
    "fingerprint",
    "protocol",
    "fold",
    "C",
    "l1_ratio",
    "class_weight",
    "n_nonzero_coefficients",
    "sparsity",
    "l1_norm",
    "l2_norm",
    "inner_pr_auc",
    "test_pr_auc",
    "inner_test_gap",
    "train_test_gap",
]

lr_table = (
    complexity_all[complexity_all["model_short"] == "LR"]
    [lr_cols]
    .sort_values(["fingerprint", "protocol", "fold"])
    .reset_index(drop=True)
)

lr_table


# ## Linear SVM complexity table
# 
# For linear SVM the indicators are:
# 
# - selected `C`
# - L2 norm of the weight vector
# - approximate margin

# In[12]:


svm_cols = [
    "model",
    "fingerprint",
    "protocol",
    "fold",
    "C",
    "class_weight",
    "l2_norm",
    "approx_margin",
    "n_nonzero_coefficients",
    "inner_pr_auc",
    "test_pr_auc",
    "inner_test_gap",
    "train_test_gap",
]

svm_table = (
    complexity_all[complexity_all["model_short"] == "SVM"]
    [svm_cols]
    .sort_values(["fingerprint", "protocol", "fold"])
    .reset_index(drop=True)
)

svm_table


# ## Decision Tree complexity table
# 
# For Decision Trees:
# 
# - selected `ccp_alpha`
# - selected `max_depth`
# - effective tree depth
# - number of nodes
# - number of leaves
# - number of features used in the tree
# - average minimum depth of the used features
# 
# The number of nodes and leaves gives a direct indication of how large the fitted tree is.

# In[13]:


dt_cols = [
    "model",
    "fingerprint",
    "protocol",
    "fold",
    "ccp_alpha",
    "max_depth",
    "effective_depth",
    "n_nodes",
    "n_leaves",
    "n_features_used",
    "used_feature_fraction",
    "feature_min_depth_mean",
    "feature_min_depth_std",
    "inner_pr_auc",
    "test_pr_auc",
    "inner_test_gap",
    "train_test_gap",
]

dt_table = (
    complexity_all[complexity_all["model_short"] == "DT"]
    [dt_cols]
    .sort_values(["fingerprint", "protocol", "fold"])
    .reset_index(drop=True)
)

dt_table


# ## Gap analysis table
# 
# This table collects the three main performance levels:
# 
# - train PR-AUC
# - inner validation PR-AUC
# - final OOD test PR-AUC
# 
# and the corresponding gaps:
# 
# $$\text{train-inner gap} = \text{train PR-AUC} - \text{inner PR-AUC}$$
# 
# $$\text{inner-test gap} = \text{inner PR-AUC} - \text{test PR-AUC}$$
# 
# $$\text{train-test gap} = \text{train PR-AUC} - \text{test PR-AUC}$$
# 
# 
# This table connects model selection, overfitting and OOD generalization.

# In[14]:


gap_cols = [
    "model",
    "model_short",
    "fingerprint",
    "protocol",
    "fold",
    "train_pr_auc",
    "inner_pr_auc",
    "inner_train_pr_auc",
    "test_pr_auc",
    "train_inner_gap",
    "inner_test_gap",
    "train_test_gap",
]

gap_analysis = (
    complexity_all[gap_cols]
    .sort_values(["model_short", "fingerprint", "protocol", "fold"])
    .reset_index(drop=True)
)

gap_analysis


# ## Aggregated complexity summary
# 

# In[15]:


summary_metrics = [
    "inner_pr_auc",
    "test_pr_auc",
    "inner_test_gap",
    "train_test_gap",
    "l2_norm",
    "n_nonzero_coefficients",
    "approx_margin",
    "effective_depth",
    "n_nodes",
    "n_leaves",
    "n_features_used",
    "feature_min_depth_mean",
]

available_summary_metrics = [
    col for col in summary_metrics
    if col in complexity_all.columns
]

complexity_summary = (
    complexity_all
    .groupby(["model", "model_short", "fingerprint", "protocol"], as_index=False)
    [available_summary_metrics]
    .agg(["mean", "std"])
)

complexity_summary.columns = [
    "_".join([c for c in col if c])
    for col in complexity_summary.columns
]

complexity_summary = complexity_summary.reset_index()

complexity_summary


# In[16]:


complexity_all.to_csv(OUTPUT_DIR / "complexity_all.csv", index=False)
lr_table.to_csv(OUTPUT_DIR / "complexity_lr.csv", index=False)
svm_table.to_csv(OUTPUT_DIR / "complexity_svm.csv", index=False)
dt_table.to_csv(OUTPUT_DIR / "complexity_dt.csv", index=False)
gap_analysis.to_csv(OUTPUT_DIR / "complexity_gap_analysis.csv", index=False)
complexity_summary.to_csv(OUTPUT_DIR / "complexity_summary.csv", index=False)

print("Saved complexity tables in:")
print(OUTPUT_DIR)

