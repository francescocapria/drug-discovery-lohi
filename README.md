# Drug Discovery Lo-Hi

A rigorous re-implementation and extension of the **Lo-Hi benchmark** for practical
machine learning in drug discovery, originally proposed by Steshin (NeurIPS 2023).

This repository provides a clean, modular pipeline to evaluate a wide range of
machine learning models, from classical baselines to multi-layer perceptrons, on the two complementary tasks defined by the benchmark:

- **Hi (Hit Identification)** — binary classification on test molecules that
  are dissimilar from the training set (Tanimoto ECFP4 < 0.4), simulating the
  search for novel, patentable hits during virtual screening.
- **Lo (Lead Optimization)** — intra-cluster ranking of structurally similar
  molecules, simulating the optimization of a known hit by minor chemical
  modifications.

## References

- **Original paper:** Simon Steshin, *Lo-Hi: Practical ML Drug Discovery
  Benchmark*, NeurIPS 2023 — Datasets and Benchmarks Track.
  [OpenReview](https://openreview.net/forum?id=H2Yb28qGLV)
- **Original benchmark code:** <https://github.com/SteshinSS/lohi_neurips2023>
- **Lo-Hi splitter library:** <https://github.com/SteshinSS/lohi_splitter>

The datasets (DRD2, HIV, KDR, Sol, KCNH2) and their pre-defined three-fold
splits are taken from the original benchmark repository.

---

## What this repository adds with respect to the original paper

- **Nested cross-validation without test-set leakage.**  
  Hyperparameters are selected independently inside each outer fold using only the
  training partition. The test fold is used only once, for the final evaluation.

- **Extended molecular representations.**  
  In addition to ECFP4 and MACCS fingerprints, this repository evaluates RDKit
  topological fingerprints and RDKit 2D descriptors. 

- **Modular classical ML pipeline.**  
  Featurization, model selection, training, evaluation and result saving are
  organized into reusable modules, making the code easier to extend to new models,
  datasets and future ChEMBL versions.

- **Larger hyperparameter searches.**  
  Classical models are evaluated with more systematic grid or random searches,
  performed separately for each outer fold.

- **Additional models.**
  In addition to the models evaluated in the original paper, the repository also tests Logistic Regression, Decision Tree, Random Forest, XGBoost, and SVM with linear, polynomial, and Tanimoto kernels.

- **Improved MLP pipeline.**  
  The MLP implementation supports configurable depth and width, flat or pyramidal
  architectures, multiple activation functions, Dropout, optional Batch
  Normalization, AdamW, learning-rate scheduling, gradient clipping, early stopping
  on an internal validation split, and multi-seed ensembling.

- **Task-specific MLP training.**  
  Hi is treated as binary classification, while Lo is treated as regression/ranking
  with target standardization and final intra-cluster Spearman evaluation.

- **OOD vs in-distribution inner validation protocol comparison.**
  A systematic study of how the choice of inner validation strategy during
  hyperparameter selection affects model complexity, optimism bias, feature
  reliance, and final out-of-distribution generalization. This comparison is
  conducted on all four Hi datasets (DRD2, HIV, KDR, Sol) across Decision Tree,
  Logistic Regression, and Linear SVM, with ECFP4, MACCS, and RDKit descriptor
  representations. The analysis includes protocol-level performance tables,
  model complexity metrics, and feature-level explainability comparisons. See the
  dedicated sections below for details.

---

## Repository structure
 
```
drug-discovery-lohi/
├── configs/                  # YAML experiment configs (one per model × task × dataset)
│   └── hi/
│       ├── drd2/
│       │   ├── svm/
│       │   │   ├── svm_linear_drd2_hi.yaml
│       │   │   ├── svm_linear_drd2_hi_inner_ood_holdout.yaml
│       │   │   ├── svm_linear_drd2_hi_random_shuffle.yaml
│       │   │   └── ...
│       │   ├── lr/
│       │   ├── decision_tree/
│       │   └── ...
│       ├── hiv/
│       ├── kdr/
│       └── sol/
├── data/                     # Lo-Hi dataset folds (CSV)
│   ├── hi/{drd2,hiv,kdr,sol}/{train,test}_{1,2,3}.csv
│   └── lo/{drd2,kcnh2,kdr}/{train,test}_{1,2,3}.csv
├── features/                 # Cached precomputed fingerprints (.npz)
├── notebooks/
│   ├── drd2_hi_ood_vs_random_shuffle/
│   │   ├── 01_build_protocol_tables_drd2.ipynb
│   │   ├── 02_protocol_plots_drd2.ipynb
│   │   ├── 03_model_complexity_tables_drd2.ipynb
│   │   ├── 04_model_complexity_plots_drd2.ipynb
│   │   ├── 05_feature_explainability_tables_drd2.ipynb
│   │   └── 06_feature_explainability_plots_drd2.ipynb
│   ├── hiv_hi_ood_vs_random_shuffle/      # Same 6-notebook structure
│   ├── kdr_hi_ood_vs_random_shuffle/
│   ├── sol_hi_ood_vs_random_shuffle/
│   └── mlp/
│       ├── mlp_drd2_hi_ecfp4.ipynb
│       ├── mlp_drd2_hi_rdkit.ipynb
│       └── ...
├── results/
│   ├── hi/{dataset}/{model_fp}/
│   │   ├── train_{1,2,3}.csv
│   │   ├── test_{1,2,3}.csv
│   │   ├── params_fold_{1,2,3}.json
│   │   ├── model_fold_{1,2,3}.joblib      # When artifacts.save_model: true
│   │   ├── complexity_fold_{1,2,3}.json   # When artifacts.save_complexity: true
│   │   ├── feature_importance_fold_{1,2,3}.csv  # When artifacts.save_feature_importance: true
│   │   └── cv_results_fold_{1,2,3}.csv    # When artifacts.save_cv_results: true
│   ├── lo/{dataset}/{model_fp}/
│   └── results_ood_vs_random_shuffle/
│       └── hi/{drd2,hiv,kdr,sol}/
│           ├── protocol_summary_numeric.csv
│           ├── protocol_summary_display.csv
│           ├── protocol_per_fold.csv
│           ├── protocol_delta.csv
│           ├── hyperparameters_all.csv
│           ├── hyperparameters_{dt,lr,svm}.csv
│           ├── hyperparameters_set_summary.csv
│           ├── complexity_all.csv
│           ├── complexity_{dt,lr,svm}.csv
│           ├── complexity_gap_analysis.csv
│           ├── complexity_summary.csv
│           ├── feature_importance_all.csv
│           ├── feature_topk.csv
│           ├── feature_overlap_protocol.csv
│           ├── feature_stability_intra_protocol.csv
│           ├── feature_importance_summary.csv
│           ├── local_molecule_candidates.csv
│           ├── local_feature_contributions.csv
│           ├── figures/                   # 4 protocol comparison plots
│           ├── figures_complexity/        # 4 complexity plots
│           └── figures_feature_explainability/  # 5 types of explainability plots
├── splits/
├── training/
│   └── train_model.py
└── utils/
    ├── fingerprints.py
    ├── metrics.py
    ├── cv_pipeline.py        # kfold / holdout / random_shuffle, artifact saving
    ├── explainability.py     # topk overlap, linear contributions, pipeline unwrapping
    ├── io_utils.py
    ├── config_loader.py
    ├── mlp_utils.py
    └── mlp_utils_lo.py
```
 
---

## Installation

The code requires Python 3.10 or newer. A CUDA-capable GPU is recommended
for the MLP notebooks but not required for the classical models in
`training/train_model.py`.

If you need a specific CUDA build of PyTorch, install it explicitly before
running `pip install -r requirements.txt`.

The experiments in this repository were originally run on a machine equipped
with two NVIDIA Tesla V100S 32 GB GPUs.

---

## Datasets

The dataset folds are not bundled with this repository. They must be obtained
from the original Lo-Hi benchmark repository
(<https://github.com/SteshinSS/lohi_neurips2023>) and placed under `data/`
with the following layout:

```
data/
├── hi/
│   ├── drd2/{train,test}_{1,2,3}.csv
│   ├── hiv/{train,test}_{1,2,3}.csv
│   ├── kdr/{train,test}_{1,2,3}.csv
│   └── sol/{train,test}_{1,2,3}.csv
└── lo/
    ├── drd2/{train,test}_{1,2,3}.csv
    ├── kcnh2/{train,test}_{1,2,3}.csv
    └── kdr/{train,test}_{1,2,3}.csv
```

Each CSV file contains a `smiles` column, a `value` column (binary for Hi,
continuous for Lo) and, for Lo only, a `cluster` column identifying the
cluster of each test molecule.

---

## Running experiments

### 1. Classical machine learning models

Classical models (KNN, SVM, Logistic / Linear Regression, Decision Tree,
Random Forest, Gradient Boosting, XGBoost, dummy baseline) are trained from
the command line through `training/train_model.py`, which reads a YAML
configuration file describing the experiment.

```bash
python training/train_model.py \
    --config configs/hi/drd2/svm/svm_ecfp4_drd2_hi.yaml
```

Optional arguments:

```bash
# Run only a subset of the outer folds
python training/train_model.py --config <path> --folds 1 2

# Validate the configuration without running anything
python training/train_model.py --config <path> --dry-run
```

A YAML config has the following structure:

```yaml
experiment:
  name: svm_ecfp4_drd2_hi
  task: hi               # "hi" or "lo"
  dataset: drd2          # drd2, hiv, kdr, sol, kcnh2

fingerprint:
  type: ecfp4            # ecfp4, maccs, rdkit_topo, rdkit_desc

model:
  name: svm              # knn, svm, gb, rf, lr, linreg, dt, dummy, xgb
  fixed:
    kernel: rbf
    probability: true
  search:
    C: [0.1, 0.5, 1.0, 2.0, 5.0]
    gamma: [scale, auto]

cv:
  inner_k: 3
  scoring: average_precision
  search_strategy: grid   # "grid" or "random"
  n_iter: 50              # only used when search_strategy == "random"
  random_state: 42
```

For each outer fold the pipeline featurizes the training and test molecules
(or loads cached fingerprints from `features/`), runs the inner
cross-validation on the training partition only to select hyperparameters,
refits the best estimator on the full training partition and evaluates it on
the test partition. Predictions and per-fold best hyperparameters are saved
under `results/{task}/{dataset}/{model_fp}/`.

### 2. Multi-layer perceptron experiments

The MLP experiments are implemented as Jupyter notebooks under `notebooks/`,
one per dataset × fingerprint combination, for example:

- `notebooks/mlp_drd2_hi_ecfp4.ipynb`
- `notebooks/mlp_drd2_hi_rdkit.ipynb`
- `notebooks/mlp_drd2_lo_ecfp4.ipynb`
- `notebooks/mlp_kcnh2_lo_rdkit.ipynb`
- `notebooks/mlp_hiv_hi_ecfp4.ipynb`
- ...

The MLP pipeline is implemented in `utils/mlp_utils.py` for Hi tasks and
`utils/mlp_utils_lo.py` for Lo tasks. These files include inline comments and
docstrings explaining the main steps of the pipeline.

---

## Evaluation metrics

- **Hi task** — PR-AUC (primary), ROC-AUC, BEDROC (α = 70), F1 at threshold
  0.5 when applicable, and positive rate for reference. Metrics are
  aggregated across the three outer folds as mean ± standard deviation.

- **Lo task** — Mean Spearman rank correlation computed inside each test
  cluster of structurally similar molecules and then averaged across
  clusters, following the original Lo-Hi benchmark protocol. Mean intra-cluster
  R² and MAE are also reported as auxiliary metrics. Clusters with fewer
  than three molecules are skipped.

All metric definitions are in `utils/metrics.py`.

---

# Results

## Hi Task (Hit Identification) — PR-AUC 

| Model | DRD2-Hi | HIV-Hi | KDR-Hi | Sol-Hi |
|---|---:|---:|---:|---:|
| Dummy | 0.6765 ± 0.0614 | 0.0399 ± 0.0143 | 0.6092 ± 0.0814 | 0.2154 ± 0.0083 |
| LR (ECFP4) | 0.7743 ± 0.0841 | 0.0714 ± 0.0307 | 0.6603 ± 0.0642 | 0.4813 ± 0.0502 |
| LR (MACCS) | 0.7327 ± 0.0641 | 0.0988 ± 0.0491 | 0.6018 ± 0.0907 | 0.4740 ± 0.0172 |
| LR (RDKit Desc.) | 🥇  **0.7922 ± 0.0603** | 0.1123 ± 0.0538 | 0.6450 ± 0.0784 | 🥈 **0.5955 ± 0.0246** |
| KNN (ECFP4) | 0.7078 ± 0.0486 | 0.0656 ± 0.0261 | 0.6473 ± 0.0812 | 0.4466 ± 0.0300 |
| KNN (MACCS) | 0.7100 ± 0.0496 | 0.0717 ± 0.0335 | 0.6284 ± 0.0886 | 0.4264 ± 0.0341 |
| DT (ECFP4) | 0.7145 ± 0.0712 | 0.0450 ± 0.0140 | 0.6421 ± 0.0252 | 0.3298 ± 0.0220 |
| DT (MACCS) | 0.6872 ± 0.0481 | 0.0682 ± 0.0353 | 0.6302 ± 0.0903 | 0.3317 ± 0.0283 |
| SVM Linear (ECFP4) | 0.7732 ± 0.0876 | 0.0623 ± 0.0315 | 0.6498 ± 0.0788 | 0.4791 ± 0.0645 |
| SVM Linear (MACCS) | 0.7293 ± 0.0561 | 0.1125 ± 0.0510 | 0.5824 ± 0.0854 | 0.4766 ± 0.0191 |
| SVM Poly (ECFP4) | 0.7454 ± 0.0869 | 0.0851 ± 0.0405 | 0.6104 ± 0.1532 | 0.5021 ± 0.0442 |
| SVM Poly (MACCS) | 0.7326 ± 0.0463 | 0.1058 ± 0.0530 | 0.5960 ± 0.1101 | 0.4671 ± 0.0369 |
| SVM Poly (RDKit Desc.) | 🥉 **0.7750 ± 0.0615** | 0.0840 ± 0.0397 | 🥇  **0.6744 ± 0.0608** | 0.5781 ± 0.0244 |
| SVM RBF (ECFP4) | 0.7728 ± 0.0790 | 0.0953 ± 0.0426 | 🥉 **0.6690 ± 0.0249** | 0.4916 ± 0.0513 |
| SVM RBF (MACCS) | 0.7321 ± 0.0477 | 0.1101 ± 0.0580 | 0.6178 ± 0.0534 | 0.4729 ± 0.0368 |
| SVM RBF (RDKit Desc.) | 0.7574 ± 0.0609 | 0.1020 ± 0.0505 | 0.6547 ± 0.0627 | 🥉 **0.5875 ± 0.0431** |
| SVM Tanimoto (ECFP4) | 0.7745 ± 0.0782 | 0.0827 ± 0.0331 | 🥈 **0.6723 ± 0.0436** | 0.4849 ± 0.0473 |
| SVM Tanimoto (MACCS) | 0.7319 ± 0.0498 | 0.0966 ± 0.0468 | 0.5829 ± 0.0815 | 0.4747 ± 0.0400 |
| RF (ECFP4) | 0.7471 ± 0.0646 | 0.1105 ± 0.0624 | 0.6547 ± 0.0701 | 0.4824 ± 0.0391 |
| RF (MACCS) | 0.7238 ± 0.0516 | 🥉 **0.1348 ± 0.0766** | 0.6125 ± 0.0772 | 0.4646 ± 0.0263 |
| RF (RDKit Desc.) | 0.7671 ± 0.0689 | 🥈 **0.1491 ± 0.1148** | 0.6556 ± 0.0574 | 0.5476 ± 0.0301 |
| GB (ECFP4) | 0.7450 ± 0.0835 | 0.1053 ± 0.0560 | 0.6642 ± 0.0604 | 0.4680 ± 0.0199 |
| GB (MACCS) | 0.7419 ± 0.0639 | 0.1186 ± 0.0604 | 0.5891 ± 0.0938 | 0.4853 ± 0.0450 |
| GB (RDKit Desc.) | 🥈 **0.7809 ± 0.0565** | 🥇  **0.1511 ± 0.0805** | 0.6562 ± 0.0608 | 0.5691 ± 0.0178 |
| XGBoost (ECFP4) | 0.7552 ± 0.0853 | 0.0972 ± 0.0463 | 0.6255 ± 0.0662 | 0.4858 ± 0.0253 |
| XGBoost (MACCS) | 0.7376 ± 0.0643 | 0.1142 ± 0.0653 | 0.6214 ± 0.0836 | 0.5025 ± 0.0264 |
| XGBoost (RDKit Desc.) | 0.7726 ± 0.0479 | 0.1199 ± 0.0695 | 0.6615 ± 0.0734 | 0.5656 ± 0.0197 |
| MLP (ECFP4) | 0.7689 ± 0.0803 | 0.0663 ± 0.0220 | 0.6403 ± 0.0468 | 0.4575 ± 0.0254 |
| MLP (RDKit Desc.) | 0.7621 ± 0.0748 | 0.1044 ± 0.0457 | 0.6291 ± 0.0886 | 🥇  **0.5972 ± 0.0260** |

---

## Lo Task (Lead Optimization) — Mean Spearman 

| Model | DRD2-Lo | KCNH2-Lo | KDR-Lo |
|---|---:|---:|---:|
| Dummy | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| LinReg (ECFP4) | 0.0968 ± 0.0494 | 0.1795 ± 0.0198 | 0.0214 ± 0.0704 |
| LinReg (MACCS) | 0.1315 ± 0.0390 | 0.1222 ± 0.0048 | 0.0540 ± 0.0866 |
| LinReg (RDKit Desc.) | 0.2046 ± 0.0175 | 0.3765 ± 0.0093 | 0.1049 ± 0.0711 |
| KNN (ECFP4) | 0.2188 ± 0.0323 | 0.1903 ± 0.0505 | 0.0787 ± 0.0490 |
| KNN (MACCS) | 0.1478 ± 0.0421 | 0.1618 ± 0.0329 | 0.0376 ± 0.0113 |
| DT (ECFP4) | 0.1500 ± 0.0541 | 0.1696 ± 0.0664 | 0.0083 ± 0.0351 |
| DT (MACCS) | 0.0353 ± 0.0348 | 0.0460 ± 0.0011 | 0.0292 ± 0.0815 |
| SVM Linear (ECFP4) | 0.1299 ± 0.0170 | 0.2113 ± 0.0127 | 0.0957 ± 0.0127 |
| SVM Linear (MACCS) | 0.1300 ± 0.0562 | 0.1479 ± 0.0113 | −0.0067 ± 0.0471 |
| SVM Poly (ECFP4) | 0.2286 ± 0.0264 | 🥉 **0.4426 ± 0.0235** | 0.1371 ± 0.0189 |
| SVM Poly (MACCS) | 0.2301 ± 0.0353 | 0.0797 ± 0.0094 | 0.0982 ± 0.0064 |
| SVM Poly (RDKit Desc.) | 0.2695 ± 0.0350 | 0.3657 ± 0.0130 | 0.0738 ± 0.0247 |
| SVM RBF (ECFP4) | 0.1734 ± 0.0082 | 🥈 **0.4453 ± 0.0205** | 0.1273 ± 0.0204 |
| SVM RBF (MACCS) | 0.2483 ± 0.0224 | 0.1335 ± 0.0244 | 0.0651 ± 0.0171 |
| SVM RBF (RDKit Desc.) | 0.2021 ± 0.0195 | 0.0170 ± 0.0275 | 0.1391 ± 0.0179 |
| SVM Tanimoto (ECFP4) | 0.1820 ± 0.0131 | 0.3919 ± 0.0137 | 0.1398 ± 0.0153 |
| SVM Tanimoto (MACCS) | 0.2480 ± 0.0418 | 0.1186 ± 0.0202 | 0.0575 ± 0.0330 |
| RF (ECFP4) | 🥇  **0.3188 ± 0.0255** | 0.3458 ± 0.0263 | 0.1070 ± 0.0207 |
| RF (MACCS) | 0.1878 ± 0.0388 | 0.1598 ± 0.0208 | 0.1227 ± 0.0491 |
| RF (RDKit Desc.) | 0.2141 ± 0.0222 | 0.4127 ± 0.0421 | 🥇  **0.1691 ± 0.0268** |
| GB (ECFP4) | 0.2660 ± 0.0362 | 0.4008 ± 0.0272 | 0.0748 ± 0.0098 |
| GB (MACCS) | 0.2015 ± 0.0298 | 0.1914 ± 0.0251 | 0.1184 ± 0.0146 |
| GB (RDKit Desc.) | 0.1742 ± 0.0538 | 0.3914 ± 0.0553 | 🥈 **0.1527 ± 0.0456** |
| XGBoost (ECFP4) | 🥈 **0.2943 ± 0.0530** | 0.4188 ± 0.0100 | 0.0715 ± 0.0102 |
| XGBoost (MACCS) | 0.1967 ± 0.0240 | 0.1358 ± 0.0558 | 0.0853 ± 0.0526 |
| XGBoost (RDKit Desc.) | 0.2314 ± 0.0809 | 🥇  **0.4508 ± 0.0287** | 0.1434 ± 0.0268 |
| MLP (ECFP4) | 🥉 **0.2757 ± 0.0406** | 0.3970 ± 0.0188 | 🥉 **0.1507 ± 0.0293** |
| MLP (RDKit Desc.) | 0.2732 ± 0.0248 | 0.4147 ± 0.0551 | 0.1098 ± 0.0167 |

---

## OOD vs in-distribution validation — Hi task

### Motivation and research questions

The Hi task requires models to generalize to molecules that are structurally
dissimilar from anything seen during training (Tanimoto ECFP4 < 0.4). Standard
nested cross-validation selects hyperparameters using a random inner validation
split, where the validation molecules may be chemically similar to the inner
training molecules. The concern is concrete: a model validated on its own
chemical neighborhood might look much better internally than it actually is on
truly novel structures, and might be guided toward higher capacity and weaker
regularization precisely because those configurations are better at memorizing
local patterns.

The analysis investigates five questions. First, whether the inner-to-test
optimism gap (inner PR-AUC − final OOD test PR-AUC) is systematically larger
with random shuffle validation. Second, whether random shuffle selects more
complex models — deeper and larger trees, linear models with larger coefficient
norms and less sparsity, SVMs with weaker regularization. Third, whether the
two protocols select models that rely on different features, measured as top-k
feature overlap. Fourth, whether random shuffle leads to less reproducible
feature rankings across the three outer folds. Fifth, whether these effects
are consistent across molecular targets and fingerprint types or are
dataset-specific.

### Protocol design

The benchmark fold structure provides a natural OOD inner holdout without any
new splitting logic. The three Hi subsets F1, F2, F3 are mutually dissimilar
by construction, and since `train_i = Fⱼ ∪ Fₖ`, the two constituent subsets
can be recovered exactly from the test files of the other outer folds. For
outer fold 1, for example, `inner_train = test_3.csv = F1` and `inner_val =
test_2.csv = F2` — two chemically distinct groups, with no leakage since the
only forbidden file for fold 1 is `test_1.csv`. The random shuffle protocol
takes the same outer training set, shuffles it, and splits it 80/20 with
stratification. In both cases `GridSearchCV(refit=True)` with `PredefinedSplit`
ensures the selected model is always refitted on the full outer training set
before test evaluation.

### Analysis structure and plots

The comparison is organized into six notebooks per dataset, covering three
levels of analysis.

**Protocol-level performance** (notebooks 01–02). The key figure is a scatter
plot of mean inner validation PR-AUC against mean final OOD test PR-AUC, with
the identity diagonal separating well-calibrated from optimistic protocols.
Complementary figures show the optimism gap over the three outer folds per
model, and a delta heatmap summarizing the differences across all four metrics
(inner score, test score, inner-test gap, train-test gap) for all model and
fingerprint combinations.

**Model complexity** (notebooks 03–04). Strip plots with fold-level points
show how the key complexity indicator per model family — number of nodes for
Decision Tree, L2 norm of the coefficient vector for Logistic Regression and
SVM — distributes across fingerprints and protocols. Scatter plots relate
complexity directly to the optimism gap, and a log-ratio heatmap shows which
hyperparameters (C, ccp_alpha, L2 norm, sparsity, margin) are systematically
higher or lower under random shuffle. A three-stage line plot connecting Train,
Inner val, and OOD test PR-AUC for every fold makes the drop pattern
immediately visible.

**Feature-level explainability** (notebooks 05–06). Barplots show the mean
top-k overlap between features selected by the two protocols, with individual
fold values as scatter points. A single stability heatmap summarizes
intra-protocol feature consistency across fold pairs for all model and
fingerprint combinations. Cumulative importance curves reveal whether
importance is concentrated on few features or spread across many. For Decision
Trees, depth distribution plots show how deep the most important features
appear in the tree. Finally, side-by-side local contribution plots compare
the feature-level evidence produced by the two protocols for the same test
molecules, focusing on cases where the protocols disagree.

### Results

The effect of the validation protocol on final OOD test PR-AUC is strongly
dataset-dependent. On KDR-Hi, OOD holdout outperforms random shuffle by 15–35
percentage points across all models and fingerprints — the sharpest example of
in-distribution validation being actively misleading. On DRD2-Hi the effect is
model-dependent, with OOD holdout better for Decision Tree and the picture
mixed for linear models. On HIV-Hi and Sol-Hi the differences are small and
inconsistent in direction, suggesting the severity of the effect scales with
the chemical dissimilarity between inner validation and final test.

#### DRD2-Hi

| Model + Fingerprint | OOD holdout | Random shuffle |
|---|---:|---:|
| DT + ECFP4 | **0.7164 ± 0.0383** | 0.6774 ± 0.0556 |
| DT + MACCS | **0.6802 ± 0.0557** | 0.6739 ± 0.0674 |
| LR + ECFP4 | 0.7572 ± 0.0813 | **0.7677 ± 0.0900** |
| LR + MACCS | 0.7461 ± 0.0751 | **0.7516 ± 0.0652** |
| LR + RDKit desc | **0.7883 ± 0.0660** | 0.7713 ± 0.0730 |
| SVM + ECFP4 | 0.7532 ± 0.0890 | **0.7725 ± 0.0875** |
| SVM + MACCS | 0.7394 ± 0.0640 | **0.7410 ± 0.0602** |

#### HIV-Hi

| Model + Fingerprint | OOD holdout | Random shuffle |
|---|---:|---:|
| DT + ECFP4 | 0.0467 ± 0.0207 | **0.0596 ± 0.0251** |
| DT + MACCS | **0.0778 ± 0.0511** | 0.0536 ± 0.0121 |
| LR + ECFP4 | 0.0577 ± 0.0197 | **0.0714 ± 0.0307** |
| LR + MACCS | **0.1129 ± 0.0633** | 0.1001 ± 0.0485 |
| LR + RDKit desc | 0.0885 ± 0.0386 | **0.0887 ± 0.0370** |
| SVM + ECFP4 | 0.0532 ± 0.0162 | **0.0728 ± 0.0401** |
| SVM + MACCS | 0.1034 ± 0.0434 | **0.1061 ± 0.0558** |

#### KDR-Hi

| Model + Fingerprint | OOD holdout | Random shuffle |
|---|---:|---:|
| DT + ECFP4 | **0.9423 ± 0.0254** | 0.6489 ± 0.0721 |
| DT + MACCS | **0.9456 ± 0.0073** | 0.6062 ± 0.0859 |
| LR + ECFP4 | **0.9563 ± 0.0089** | 0.6664 ± 0.0517 |
| LR + MACCS | **0.8431 ± 0.0305** | 0.6105 ± 0.0927 |
| LR + RDKit desc | **0.8776 ± 0.0240** | 0.6578 ± 0.0643 |
| SVM + ECFP4 | **0.9440 ± 0.0076** | 0.6943 ± 0.0433 |
| SVM + MACCS | **0.8339 ± 0.0303** | 0.5840 ± 0.1176 |

#### Sol-Hi

| Model + Fingerprint | OOD holdout | Random shuffle |
|---|---:|---:|
| DT + ECFP4 | **0.3038 ± 0.0105** | 0.2974 ± 0.0125 |
| DT + MACCS | 0.3326 ± 0.0134 | **0.3553 ± 0.0384** |
| LR + ECFP4 | 0.4778 ± 0.0403 | **0.4820 ± 0.0393** |
| LR + MACCS | **0.4768 ± 0.0172** | 0.4612 ± 0.0271 |
| LR + RDKit desc | **0.5908 ± 0.0331** | 0.5875 ± 0.0221 |
| SVM + ECFP4 | **0.4968 ± 0.0314** | 0.4629 ± 0.0591 |
| SVM + MACCS | **0.4756 ± 0.0144** | 0.4513 ± 0.0099 |

---

## OOD vs in-distribution validation — Lo task

The same OOD holdout strategy cannot be applied to Lo. Hi folds are
constructed around global chemical dissimilarity, F1, F2, F3 are mutually
dissimilar subsets,  which is exactly what makes the fold reconstruction work.
Lo folds are organized around clusters of chemical analogues: the split
criterion is cluster membership, not pairwise molecular distance, so the test
files of other outer folds are not chemically separated from the training set
in the same meaningful sense. Using them as an inner OOD holdout for Lo would
not replicate the right generalization scenario.

The code blocks the holdout strategy for Lo explicitly, with a ValueError that
suggests either `kfold` or `random_shuffle`, or a future cluster-aware holdout
where entire clusters of analogues are held out during inner validation — an
experiment that would need its own careful design around cluster sizes and the
intra-cluster Spearman evaluation protocol.

## Proposed extension: LLM-assisted chemical explanation audit

A natural extension of this work is to move from predictive evaluation to
diagnostic evaluation. The current OOD vs in-distribution analysis measures
whether different inner validation protocols lead to different test performance,
model complexity and feature reliance. PR-AUC alone does not tell us whether a
correct prediction is chemically meaningful, or whether a model is exploiting
shortcuts that happen to work on a given split.

The proposed extension is an LLM-assisted chemical explanation audit. The LLM
would not be used as a molecular property predictor. Instead, it would act as a
structured judge of the explanations produced by already trained modelsthe
model predictions remain entirely produced by classical ML (Logistic Regression,
Linear SVM, Decision Tree), and the LLM only evaluates whether the evidence
supporting those predictions is chemically plausible. This distinction is
methodologically important: the LLM has no access to the benchmark labels
during the audit, and its task is assessment rather than prediction.

The framework is inspired by the LLM-as-a-Judge methodology introduced by
Zheng et al. in "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
(NeurIPS 2023). From that work we adopt three core methodological ideas:
using a strong LLM as a scalable evaluator of structured outputs, distinguishing
between single grading and pairwise comparison, and controlling for evaluation
artifacts — particularly position bias, where the judge systematically favors
the explanation presented first regardless of content.

### What the judge receives

For each selected test molecule, the audit case packages structured evidence
extracted from the existing pipeline: the test SMILES, the true label and model
prediction, the predicted score as a proxy for confidence, the top local feature
contributions (xⱼ × wⱼ for linear models, the molecule-specific decision path
for Decision Tree via `sklearn.tree.decision_path`), the nearest training
neighbors by ECFP4 Tanimoto similarity with their labels, and a compact set of
interpretable RDKit descriptors (molecular weight, LogP, TPSA, HBD, HBA,
rotatable bonds, ring count, fraction Csp3). Cases are selected to cover
informative scenarios: correct by OOD holdout only, correct by random shuffle
only, high-confidence wrong, both correct with very different confidence, and
both wrong.

### Single grading and pairwise comparison

In single grading, the judge evaluates one explanation at a time and assigns a
plausibility score and a shortcut flag — indicating whether the highlighted
features seem chemically generic, uninformative or inconsistent with the
neighbor labels. In pairwise comparison, the judge receives two anonymous
explanations A and B for the same molecule, same fold, same model and same
fingerprint, one from the OOD holdout model and one from the random shuffle
model, without knowing which is which. It then selects the more chemically
plausible explanation or declares a tie. To control for position bias, each
pair is also evaluated in the swapped order (B first, A second): if the
judge changes its preference, the case is flagged as inconsistent.

### Metrics

From the single grading, the main metrics could be  the Plausible Correct Rate (PCR),
defined as the fraction of correct predictions that receive a plausible
explanation, the Shortcut Rate (SR), defined as the fraction of all judgements
where the shortcut flag is raised, and the Mean Plausibility Score (MPS),
defined as the average numeric plausibility rating across all cases. From the
pairwise comparison, the key metric is the OOD Preference Rate (OPR), defined
as the fraction of pairs where the OOD holdout explanation is preferred over
the random shuffle explanation net of position effects. A Plausibility Gap (PG)
is also computed as the difference in MPS between the two protocols across
matched pairs. Position consistency is measured as the fraction of pairs where
the judge gives the same answer before and after swapping.

### Bias controls

Beyond position bias, the audit includes two additional controls taken from
the MT-Bench methodology. Verbosity bias is controlled by keeping the
explanation format strictly structured and equal in length for both protocols.
A blind ablation removes the true label from the prompt entirely, allowing
comparison of judge behaviour with and without knowledge of the ground truth,
which tests whether the judge is reasoning about chemical plausibility
independently or anchoring on the correct answer.

### Future direction: graph neural networks

This audit framework becomes significantly more powerful when applied to graph
neural network models. GNN explanations produced by methods such as GNNExplainer
or attention weights provide subgraph-level evidence, specific bonds and
atoms driving the prediction, rather than bit-level fingerprint contributions.
An LLM trained on chemical literature can evaluate whether a highlighted
subgraph corresponds to a known pharmacophore, a reactive group or a structural
motif consistent with the target biology. This would turn the audit from a
pattern-consistency check into a genuine chemical knowledge assessment, and
would make the LLM-as-judge framework substantially more informative than it
can be with anonymous fingerprint bits. The modular structure of the audit
pipeline (evidence packaging, single and pairwise prompting, bias controls,
metrics) is designed to accommodate this extension without architectural changes.

---

## Important papers to study
 
**Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", NeurIPS 2023.**


**Kifer, Ben-David, Gehrke, "Detecting Change in Data Streams", Cornell University.**

 
**Ben-David, Blitzer, Crammer, Kulesza, Pereira, Vaughan, "A Theory of Learning
from Different Domains".**

## Next steps
 
**Evaluate and implement the LLM audit.** The infrastructure for building
structured judge cases is already in place in `utils/llm_audit/`. The natural
next step is to implement `02_run_llm_judge_drd2_hi.ipynb`, run a pilot audit
on DRD2-Hi with a small case set, and evaluate whether the LLM plausibility
judgements are internally consistent and correlate with the quantitative
metrics from the OOD vs in-distribution analysis.
 
**Study Kifer et al. and Ben-David et al. and integrate them into the analysis.**
The two distribution shift papers provide the theoretical language to
describe what the OOD vs in-distribution experiments are actually measuring.
Concretely, this means quantifying the distribution shift between the inner
validation set and the outer test set for each dataset and protocol, and
testing whether the magnitude of that shift predicts the size of the optimism
gap. KDR-Hi is the most extreme case and would be a good starting point.
 
**Extend the analysis to graph neural networks.** 

---

## Use of large language models

Anthropic's Claude was used during the development of this repository,
mainly to refactor exploratory code into a more modular and professional
structure (Claude Sonnet 4.6), and to assist with debugging (Claude Opus 4.6).

---

## License

The datasets are released under the MIT license by the authors of the
original Lo-Hi benchmark. Code in this repository is also released under the
MIT license unless otherwise noted.

---






















<!-- ## Citation

If you use this repository or the Lo-Hi benchmark in your work, please cite
the original paper:

```bibtex
@inproceedings{steshin2023lohi,
  title     = {Lo-Hi: Practical ML Drug Discovery Benchmark},
  author    = {Steshin, Simon},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS),
               Datasets and Benchmarks Track},
  year      = {2023},
  url       = {https://openreview.net/forum?id=H2Yb28qGLV}
}
``` -->
