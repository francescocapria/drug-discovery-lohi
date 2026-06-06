# Drug Discovery Lo-Hi

A rigorous re-implementation and extension of the **Lo-Hi benchmark** for
practical machine learning in drug discovery, originally proposed by Steshin
(NeurIPS 2023).

This repository provides a clean, modular pipeline to evaluate a wide range of machine learning models, from classical baselines to multi-layer perceptrons, on the two complementary tasks defined by the benchmark:

- **Hi (Hit Identification)** — binary classification on test molecules that
  are structurally dissimilar from the training set (Tanimoto ECFP4 < 0.4),
  simulating the search for novel, patentable hits during virtual screening.
- **Lo (Lead Optimization)** — intra-cluster ranking of structurally similar
  molecules, simulating the optimization of a known hit by minor chemical
  modifications.

The additional analyses are designed to separate three questions: whether the validation protocol changes model selection, whether Lo-Hi folds are structurally distinguishable, and whether the observed structural shift overlaps with activity-relevant molecular features.

### References

- **Original paper:** Simon Steshin, *Lo-Hi: Practical ML Drug Discovery
  Benchmark*, NeurIPS 2023 — Datasets and Benchmarks Track.
  [OpenReview](https://openreview.net/forum?id=H2Yb28qGLV)
- **Original benchmark code:** <https://github.com/SteshinSS/lohi_neurips2023>
- **Lo-Hi splitter library:** <https://github.com/SteshinSS/lohi_splitter>

---

## Main training pipeline and results

### Pipeline overview

The main training pipeline is driven by YAML configuration files and executed
through `training/train_model.py`. Each config specifies dataset, task,
molecular representation, model family, hyperparameter search space, inner
cross-validation strategy, and which artifacts to save.

For each outer fold the pipeline: loads or computes cached fingerprints →
runs inner cross-validation on the training partition only → selects
hyperparameters → refits on the full training partition → evaluates on the
held-out test fold. Predictions, best parameters, trained models, complexity
metrics, and feature importances are saved under `results/`.

**Supported models:** Logistic Regression, Linear/Poly/RBF/Tanimoto SVM,
Decision Tree, Random Forest, Gradient Boosting, XGBoost, KNN, Dummy
baseline, MLP (separate notebooks).

**Molecular representations:** ECFP4 (2048 bits), MACCS (167 keys), RDKit
2D descriptors, RDKit topological fingerprints.

**Evaluation metrics:** PR-AUC (primary for Hi), ROC-AUC, BEDROC; mean
intra-cluster Spearman correlation (primary for Lo).

### Hi task — PR-AUC

| Model | DRD2-Hi | HIV-Hi | KDR-Hi | Sol-Hi |
|---|---:|---:|---:|---:|
| Dummy | 0.677 ± 0.061 | 0.040 ± 0.014 | 0.609 ± 0.081 | 0.215 ± 0.008 |
| LR (ECFP4) | 0.774 ± 0.084 | 0.071 ± 0.031 | 0.660 ± 0.064 | 0.481 ± 0.050 |
| LR (MACCS) | 0.733 ± 0.064 | 0.099 ± 0.049 | 0.602 ± 0.091 | 0.474 ± 0.017 |
| LR (RDKit Desc.) | 🥇 **0.794 ± 0.060** | 0.112 ± 0.054 | 0.645 ± 0.078 | 🥈 **0.596 ± 0.025** |
| DT (ECFP4) | 0.715 ± 0.071 | 0.045 ± 0.014 | 0.642 ± 0.025 | 0.330 ± 0.022 |
| DT (MACCS) | 0.687 ± 0.048 | 0.068 ± 0.035 | 0.630 ± 0.090 | 0.332 ± 0.028 |
| SVM Linear (ECFP4) | 0.773 ± 0.088 | 0.062 ± 0.032 | 0.650 ± 0.079 | 0.479 ± 0.065 |
| SVM Linear (MACCS) | 0.729 ± 0.056 | 0.113 ± 0.051 | 0.582 ± 0.085 | 0.477 ± 0.019 |
| SVM Poly (RDKit Desc.) | 🥉 **0.775 ± 0.062** | 0.084 ± 0.040 | 🥇 **0.674 ± 0.061** | 0.578 ± 0.024 |
| SVM RBF (ECFP4) | 0.773 ± 0.079 | 0.095 ± 0.043 | 🥉 **0.669 ± 0.025** | 0.492 ± 0.051 |
| SVM Tanimoto (ECFP4) | 0.775 ± 0.078 | 0.083 ± 0.033 | 🥈 **0.672 ± 0.044** | 0.485 ± 0.047 |
| RF (RDKit Desc.) | 0.767 ± 0.069 | 🥈 **0.149 ± 0.115** | 0.656 ± 0.057 | 0.548 ± 0.030 |
| GB (RDKit Desc.) | 🥈 **0.781 ± 0.057** | 🥇 **0.151 ± 0.081** | 0.656 ± 0.061 | 0.569 ± 0.018 |
| XGBoost (RDKit Desc.) | 0.773 ± 0.048 | 🥉 **0.120 ± 0.070** | 0.662 ± 0.073 | 0.566 ± 0.020 |
| MLP (RDKit Desc.) | 0.762 ± 0.075 | 0.104 ± 0.046 | 0.629 ± 0.089 | 🥇 **0.597 ± 0.026** |

Selected rows shown. Full results for all model × fingerprint combinations are
available in the per-dataset result directories.

### Lo task — Mean Spearman

| Model | DRD2-Lo | KCNH2-Lo | KDR-Lo |
|---|---:|---:|---:|
| Dummy | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| XGBoost (ECFP4) | 🥇 **0.354 ± 0.018** | 0.419 ± 0.010 | 0.071 ± 0.010 |
| SVR Tanimoto (ECFP4) | 🥈 **0.313 ± 0.031** | 0.392 ± 0.014 | 🥈 **0.168 ± 0.012** |
| RF (ECFP4) | 🥉 **0.306 ± 0.015** | 0.346 ± 0.026 | 0.107 ± 0.021 |
| RF (RDKit Desc.) | 0.200 ± 0.005 | 0.413 ± 0.042 | 🥇 **0.175 ± 0.036** |
| GB (RDKit Desc.) | 0.174 ± 0.054 | 🥈 **0.429 ± 0.017** | 🥉 **0.143 ± 0.032** |
| XGBoost (RDKit Desc.) | 0.208 ± 0.076 | 🥉 **0.427 ± 0.028** | 0.112 ± 0.028 |
| SVM RBF (ECFP4) | 0.180 ± 0.035 | 🥇 **0.445 ± 0.021** | 0.156 ± 0.027 |
| SVM Poly (ECFP4) | 0.237 ± 0.011 |  0.357 ± 0.064 | 0.145 ± 0.038 |
| MLP (ECFP4) | 0.276 ± 0.041 | 0.397 ± 0.019 | 0.151 ± 0.029 |

---

## OOD-holdout vs random-shuffle model selection

### Motivation

The Hi task requires generalization to molecules structurally dissimilar from
training. Standard nested CV selects hyperparameters using a random inner
validation split, where validation molecules may be chemically similar to the
inner training set. This may introduce an **optimism bias**: the inner score
overestimates OOD test performance, and the hyperparameter search may favor configurations that perform well on
in-distribution-like validation data but do not necessarily improve OOD test performance.

### Protocol design

The Lo-Hi fold structure provides a natural OOD inner holdout without any new
splitting logic. The three Hi subsets F1, F2, F3 are mutually dissimilar by
construction. For each outer fold, the two non-test Lo-Hi subsets are used as chemically distinct inner train and inner validation subsets. 
After hyperparameter selection, the model is refitted on their union, which reconstructs the full outer training set. The **random-shuffle protocol** uses the same outer training set and samples a stratified validation split whose size is matched to the corresponding OOD-holdout validation subset. In both cases the
selected model is refitted on the full outer training set before test
evaluation.

### Datasets

The cross-dataset analysis uses **DRD2-Hi, HIV-Hi, and Sol-Hi**. KDR-Hi is
excluded because its outer training folds are restricted to ~500 molecules,
making the `train_i = F_j ∪ F_k` reconstruction non-comparable with the other
datasets.

### Key quantities

| Quantity | Definition | Interpretation |
|---|---|---|
| `inner_test_gap` | inner PR-AUC − test PR-AUC | optimism of the inner estimate |
| `delta_inner_optimism` | inner_random − inner_OOD | random is more optimistic? |
| `delta_test_benefit` | test_OOD − test_random | OOD helps on final test? |

### Results

| Dataset | Model + FP | OOD holdout | Random shuffle |
|---|---|---:|---:|
| DRD2 | LR + RDKit Desc. | **0.789 ± 0.066** | 0.793 ± 0.061 |
| DRD2 | SVM + ECFP4 | 0.752 ± 0.080 | **0.764 ± 0.080** |
| HIV | LR + MACCS | **0.101 ± 0.055** | 0.100 ± 0.048 |
| HIV | SVM + MACCS | 0.096 ± 0.037 | **0.102 ± 0.074** |
| Sol | DT + MACCS | **0.333 ± 0.013** | 0.316 ± 0.034 |
| Sol | LR + RDKit Desc. | **0.590 ± 0.035** | 0.586 ± 0.023 |

### Key insights

Random shuffle tends to produce more optimistic inner-validation estimates than OOD holdout, especially on DRD2 and HIV. However, this reduction in validation optimism does not translate into a systematic improvement on the final OOD test fold: the OOD and random-shuffle test performances are usually close. This suggests a distinction between an **evaluation effect** and a **selection effect**: OOD validation gives a more realistic estimate of future performance, but does not necessarily select a consistently better model. Therefore, the OOD-holdout protocol is useful for measuring optimism, while its benefit for model selection appears dataset- and fold-dependent.

### Notebooks

- `notebooks/hi_ood_vs_random_cross_dataset/01-cross_dataset_tables_hi.ipynb` — builds tables
- `notebooks/hi_ood_vs_random_cross_dataset/02_cross_dataset_plots_hi.ipynb` — generates plots

---

## Distribution-shift analysis

### Goal

Measure whether the Lo-Hi folds are structurally distinguishable, and whether
the distinguishing features overlap with the features used by activity models.

### Method

For each dataset and fold pair (F1 vs F2, F1 vs F3, F2 vs F3), a binary
classifier is trained to discriminate which fold a molecule belongs to. The
main shift metric is:

```
shift_score = max(0, 2 × balanced_accuracy − 1)
```

where 0 = indistinguishable (chance) and 1 = perfectly separable.

The main estimate uses cross-validated `same_search_cv` discriminators (DT,
LR, SVM). High-capacity in-sample discriminators are reported only as an
upper-bound diagnostic.

### List A and List B

- **List A** = features important for predicting activity (from the activity
  models trained in the OOD-vs-random pipeline).
- **List B** = features important for discriminating folds (from the shift
  classifiers).

Their overlap quantifies whether the structural shift falls on the predictive
subspace. Overlap is reported as **enrichment over random expectation**
(1.0 = chance), which normalizes for the different dimensionalities of ECFP4
(2048 bits) and MACCS (167 keys).

### Feature importance conventions

- **DT:** permutation importance evaluated on a 20% stratified holdout of the
  discriminator data (not in-sample), clipped at zero.
- **LR / SVM:** absolute coefficient values.

### Key insights

The fold-discriminator analysis confirms that the Lo-Hi splits induce real molecular distribution shift, especially for DRD2, so the weak OOD-selection benefit cannot be explained by folds being chemically identical. However, a large structural shift alone is not sufficient to guarantee that OOD validation improves final model selection. The List A/List B overlap analysis shows that the relevant question is whether the shift affects features used for activity prediction, not only whether folds are separable in molecular space. Overall, the alignment between shift features and activity features is present in some cases, but remains heterogeneous across datasets, fingerprints and model families.

### Notebooks

- `notebooks/distribution_shift_analysis/classifier_shift_test_all_hi.ipynb` — computes tables and shift models
- `notebooks/distribution_shift_analysis/classifier_shift_test_all_hi_plots.ipynb` — generates plots

---

## Tanimoto fold-distance analysis

### Goal

Directly measure the chemical distance between folds in the full molecular
space and in activity-relevant feature subspaces, using a **random-bit
baseline** to control for dimensionality reduction artifacts.

### Method

The main metric is the **complete cross-fold pairwise Tanimoto distance**:

```
D(A, B) = mean_{x∈A, y∈B} [1 − T(x, y)]
```

computed exhaustively over all molecule pairs between two folds. Nearest-
neighbour distances are saved as a secondary diagnostic.

For each dataset and fold pair, distances are computed in:

- **Full ECFP4** (all 2048 bits) — baseline structural distance.
- **Activity top-k bits** (k = 10, 20, 50, 100, 200) — selected from List A
  importances. Computed for global (pooled), OOD-specific, and random-shuffle-
  specific activity models, using the best ECFP4 model per dataset.
- **Random top-k bits** (30 repeats) — the key negative control: random
  subsets of k bits from the 2048 ECFP4 space.

### Key quantity

```
Δ = activity top-k distance − random-bit top-k distance
```

- **Δ < 0:** the activity-relevant subspace is less shifted than a random
  subspace of the same dimensionality.
- **Δ ≈ 0:** the activity-relevant subspace behaves similarly to a random
  top-k subspace.
- **Δ > 0:** the activity-relevant subspace is more shifted than a random
  subspace of the same dimensionality.

### Valid molecule fraction

At small k, some molecules may have all-zero restricted fingerprints. The
`valid_molecule_fraction` and `valid_pair_fraction` are always reported
alongside restricted-space distances. Small-k results should be interpreted together with these coverage diagnostics,
especially when the valid molecule fraction is low.

### Key insights

The complete pairwise Tanimoto analysis shows that all Hi datasets exhibit substantial fold-to-fold separation in the full ECFP4 space, but this separation changes when distances are computed only on activity-relevant bits. DRD2 and Sol generally show lower distances in activity-restricted spaces than in the full fingerprint space, while HIV behaves differently and requires more cautious interpretation because restricted-space coverage can be lower at small k. Random-bit baselines are essential here: they show whether changes in distance are due to activity relevance or simply to using fewer bits. The main conclusion is that the relationship between global structural shift and predictive-subspace shift is strongly dataset-dependent.


### Notebooks

- `notebooks/tanimoto_distance_analysis/fold_distance_tanimoto.ipynb` — computes and saves tables
- `notebooks/tanimoto_distance_analysis/fold_distance_tanimoto_plots.ipynb` — generates plots

---

## Repository structure

```
.
├── data/
│   ├── hi/{drd2,hiv,kdr,sol}/         # train_{1,2,3}.csv, test_{1,2,3}.csv
│   └── lo/{drd2,kcnh2,kdr}/           # train_{1,2,3}.csv, test_{1,2,3}.csv
├── configs/                           # YAML experiment configs by task/dataset/model/protocol
├── features/                          # Cached fingerprint/descriptor matrices (.npz)
├── results/                           # Model artifacts, predictions, metrics, analysis outputs
├── notebooks/
│   ├── hi_ood_vs_random_cross_dataset/
│   │   ├── 01-cross_dataset_tables_hi.ipynb
│   │   └── 02_cross_dataset_plots_hi.ipynb
│   ├── distribution_shift_analysis/
│   │   ├── classifier_shift_test_all_hi.ipynb
│   │   └── classifier_shift_test_all_hi_plots.ipynb
│   ├── tanimoto_distance_analysis/
│   │   ├── fold_distance_tanimoto.ipynb
│   │   └── fold_distance_tanimoto_plots.ipynb
│   └── mlp/
│       ├── mlp_drd2_hi_ecfp4.ipynb
│       ├── mlp_drd2_hi_rdkit.ipynb
│       ├── mlp_hiv_hi_ecfp4.ipynb
│       └── ...                        # One notebook per dataset × fingerprint
├── training/
│   └── train_model.py                 # Main CLI entry point
├── utils/
│   ├── config_loader.py
│   ├── cv_pipeline.py                 # kfold / holdout / random_shuffle, artifact saving
│   ├── fingerprints.py
│   ├── metrics.py
│   ├── io_utils.py
│   ├── mlp_utils.py                   # Hi MLP pipeline
│   └── mlp_utils_lo.py               # Lo MLP pipeline
├── README.md
└── requirements.txt
```

---

## How to run

### 1. Classical model training

```bash
python training/train_model.py \
          --config <path/to/config.yaml>
```

Optional flags: `--folds 1 2` (subset of outer folds), `--dry-run` (validate
config without running).

A YAML config controls dataset, task, fingerprint, model, hyperparameter
search space, inner split protocol, and artifact settings. Outputs go to
`results/`; fingerprint caches go to `features/`.

### 2. MLP experiments

MLP experiments are implemented as Jupyter notebooks under `notebooks/mlp/`,
one per dataset × fingerprint combination. Run them in Jupyter or JupyterLab.

### 3. Analysis notebooks

Run in order — each `_plots` notebook loads tables already produced by the
preceding computation notebook:

1. Run training scripts to produce baseline results.
2. `notebooks/hi_ood_vs_random_cross_dataset/01-cross_dataset_tables_hi.ipynb`
3. `notebooks/hi_ood_vs_random_cross_dataset/02_cross_dataset_plots_hi.ipynb`
4. `notebooks/distribution_shift_analysis/classifier_shift_test_all_hi.ipynb`
5. `notebooks/distribution_shift_analysis/classifier_shift_test_all_hi_plots.ipynb`
6. `notebooks/tanimoto_distance_analysis/fold_distance_tanimoto.ipynb`
7. `notebooks/tanimoto_distance_analysis/fold_distance_tanimoto_plots.ipynb`

---

## Important methodological notes

- **KDR-Hi** is excluded from the OOD-vs-random cross-dataset comparison (its
  ~500-molecule outer training folds make fold reconstruction non-comparable),
  but is included in the general benchmark results.
- **ECFP4** fingerprints use 2048 bits. Invalid SMILES are removed, not
  replaced by dummy molecules.
- **Tanimoto SVM** is restricted to binary fingerprints (ECFP4, MACCS).
- **Feature importance:** DT uses permutation importance (clipped at zero);
  LR/SVM use absolute coefficients.
- **Outer test folds** remain untouched during model selection. Random shuffle
  validation fraction is matched to OOD holdout size.
- **Complete pairwise Tanimoto distance** is the main fold-distance metric.
  Random-bit baselines (30 repeats) control for top-k dimensionality effects.
- **OOD-vs-random applies only to Hi.** Lo folds are organized by cluster
  membership, not chemical dissimilarity, so the OOD holdout reconstruction
  does not apply.

---

## Papers

* **[Lo-Hi: Practical ML Drug Discovery Benchmark](https://arxiv.org/abs/2310.06399)**
  Reference benchmark for this project; introduces the Hi/Lo drug-discovery tasks and the Lo-Hi splitting strategy.

* **[MoleculeNet: A Benchmark for Molecular Machine Learning](https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a)**
  General reference for molecular property prediction benchmarks, datasets, metrics and molecular featurizations.

* **[Real-World Molecular Out-Of-Distribution: Specification and Investigation](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01774)**
  Reference for molecular OOD generalization and realistic distribution-shift evaluation in drug discovery.

* **[Evaluating Machine Learning Models for Molecular Property Prediction: Performance and Robustness on Out-of-Distribution Data](https://pubs.acs.org/doi/10.1021/acs.jcim.5c00475)**
  Recent molecular-property prediction study comparing ID/OOD robustness across models and splitting strategies.

* **[Accuracy on the Line: On the Strong Correlation Between Out-of-Distribution and In-Distribution Generalization](https://arxiv.org/abs/2107.04649)**
  Reference for the idea that ID and OOD performance can be strongly correlated, explaining why a more realistic validation estimate may not always improve model selection.

* **[Accuracy on the Wrong Line: On the Pitfalls of Noisy Data for Out-of-Distribution Generalisation](https://proceedings.mlr.press/v258/sanyal25a.html)**
  Follow-up work on when the ID/OOD correlation breaks, especially under nuisance features, noise or shortcut-driven shifts.

* **[A Theory of Learning from Different Domains](https://link.springer.com/article/10.1007/s10994-009-5152-4)**
  Foundational domain-adaptation theory paper connecting source/target performance to distribution divergence.

* **[Detecting Change in Data Streams](https://www.vldb.org/conf/2004/RS5P1.PDF)**
  Foundational reference for statistically detecting and quantifying distributional change.

* **[CoDrug: Conformal Drug Property Prediction with Density Estimation under Covariate Shift](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7691484a7a35d5e2742279c1d926b778-Abstract-Conference.html)**
  Reference for covariate-shift handling in drug-property prediction using conformal prediction and density estimation.

* **[SIMPD: an Algorithm for Generating Simulated Time Splits for Validating Machine Learning Approaches](https://link.springer.com/article/10.1186/s13321-023-00787-9)**
  Reference for controlled molecular splitting strategies that better approximate realistic medicinal-chemistry validation settings.

* **[Scaffold Splits Overestimate Virtual Screening Performance](https://arxiv.org/abs/2406.00873)**
  Shows that scaffold splits can still overestimate virtual-screening performance, motivating more careful structural-shift evaluation.

* **[Exposing the Limitations of Molecular Machine Learning with Activity Cliffs](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01073)**
  Reference for activity cliffs and the limitations of molecular ML when small structural changes induce large activity changes.

---

## Installation

Python 3.10 or newer. A CUDA-capable GPU is recommended for MLP notebooks but
not required for classical models.

```bash
pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install it explicitly before
running `pip install`.

---

## License

The datasets are released under the MIT license by the authors of the original
Lo-Hi benchmark. Code in this repository is also released under the MIT
license.