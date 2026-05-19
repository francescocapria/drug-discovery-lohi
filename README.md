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
  In addition to the models evaluated in the original paper, the repository also
  tests Logistic Regression, Decision Tree, Random Forest, XGBoost, and SVM with
  linear, polynomial, and Tanimoto kernels.

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

## OOD vs in-distribution validation — Hi task

### Motivation

The Hi task in the Lo-Hi benchmark is designed to evaluate generalization to
structurally novel molecules: each test fold contains molecules that are
chemically distant from the training set (Tanimoto ECFP4 similarity < 0.4).
This is the setting that matters most in real drug discovery: a model is only
valuable if it can identify active compounds that look different from anything
seen during training.

However, the standard nested cross-validation procedure selects hyperparameters
using an inner validation split that is typically random, meaning the
validation molecules may be structurally similar to the inner training
molecules. This creates a potential mismatch: the model is selected on an
"easy" in-distribution validation signal, but then evaluated on a "hard"
out-of-distribution test set.

The concern is not only about the final test score. If in-distribution
validation produces inflated inner scores, practitioners may systematically
overestimate how well their models will perform on truly novel molecules —
the exact setting where drug discovery models are expected to be useful. At
the same time, in-distribution validation may guide the hyperparameter search
toward models with higher capacity and lower regularization, since complex
models can more easily memorize the training neighborhood. Such models may
overfit to chemical substructures that are common in training but absent or
misleading in OOD test sets.

The central experimental question is:

> **Does in-distribution hyperparameter selection produce inflated internal
> validation scores without improving — or even worsening — final OOD test
> performance? And does it consistently select more complex, less calibrated
> models?**

More specifically, the analysis investigates:

- **Optimism gap:** Is the gap between inner validation score and final OOD
  test score (inner PR-AUC − test PR-AUC) systematically larger with random
  shuffle validation?
- **Model complexity:** Does random shuffle select models with higher
  complexity — deeper trees with more nodes, linear models with larger
  coefficient norms and less sparsity, SVMs with weaker regularization?
- **Feature reliance:** Do the two protocols select models that rely on
  different features? Is the overlap in the top-k most important features
  lower when protocols differ?
- **Feature stability:** Are the selected features consistent across the three
  outer folds within each protocol, or does random shuffle lead to less
  reproducible feature rankings?
- **Consistency across datasets:** Are these effects specific to a particular
  molecular target or fingerprint, or do they generalize?

### Reconstructing the OOD inner holdout

The key methodological challenge is that, in the standard Lo-Hi data format,
the training files (`train_1.csv`, `train_2.csv`, `train_3.csv`) contain the
union of two subsets but do not have a column labelling which molecule comes
from which original subset. Without this information, it is not possible to
build a truly OOD inner holdout by simply reading the training file.

The solution exploits the complementary structure of the Hi benchmark folds.
The three original subsets F1, F2, F3 are constructed to be mutually
dissimilar, and the outer fold structure is:

```
test_1.csv = F3       train_1.csv = F1 ∪ F2
test_2.csv = F2       train_2.csv = F1 ∪ F3
test_3.csv = F1       train_3.csv = F2 ∪ F3
```

This means that for each outer fold, the two constituent subsets of the
training set can be recovered exactly from the test files of the other two
folds. For outer fold 1:

```
train_outer = F1 ∪ F2    test_outer = F3  (forbidden — only used for final eval)
inner_train = F1 = test_3.csv
inner_val   = F2 = test_2.csv
```

The inner split map used in the code is:

| Outer fold | inner_train | inner_val |
|---|---|---|
| 1 | test_3.csv (= F1) | test_2.csv (= F2) |
| 2 | test_3.csv (= F1) | test_1.csv (= F3) |
| 3 | test_2.csv (= F2) | test_1.csv (= F3) |

There is no leakage: `test_2.csv` and `test_3.csv` are used as "test" only in
their respective outer folds (2 and 3). In outer fold 1, both files belong to
the training partition, and using them to build the inner train/validation
split is fully legitimate. The only forbidden file is `test_1.csv`, which is
the actual test set for fold 1.

This reconstruction was empirically verified on all four Hi datasets before
running any experiments. For DRD2-Hi fold 1, for example:
`len(train_1) = 2385`, `len(test_2 ∪ test_3) = 2385`, and the two SMILES
sets are identical. The same check holds for all datasets and all three folds.

### Three inner validation protocols

The pipeline (`utils/cv_pipeline.py`) supports three inner validation
strategies, controlled by the `inner_split_strategy` field in the YAML config:

**`holdout` — OOD holdout (the new protocol).**
The inner training set is one chemically distinct Hi subset (e.g. F1) and the
inner validation set is another (e.g. F2), both reconstructed from the test
files of the other outer folds as described above. Since F1 and F2 are
mutually dissimilar by construction of the Lo-Hi benchmark, the inner
validation is genuinely OOD with respect to the inner training set.
Hyperparameter search is run with `GridSearchCV` (or `RandomizedSearchCV`)
using `PredefinedSplit`, so that the split is fixed and the best model is
refitted on the full outer training set (`F1 ∪ F2`) before test evaluation.
This protocol is only valid for Hi tasks.

**`random_shuffle` — Random in-distribution holdout (the comparison).**
The full outer training set is shuffled and split randomly into 80% inner
training and 20% inner validation, with stratification on the binary label.
Molecules from the same chemical neighborhood may appear on both sides of the
split. The search and refit procedure is identical to the holdout protocol.
This represents the "easy" validation scenario and serves as the baseline for
the comparison.

**`kfold` — Standard nested cross-validation.**
The original protocol: stratified k-fold cross-validation (default k=2) on
the inner training set. Used as an additional baseline for completeness.

All protocols use `GridSearchCV` or `RandomizedSearchCV` with `refit=True`,
so the selected model is always retrained on the full outer training partition
before evaluation on the test set. The inner validation is only used to rank
hyperparameter configurations, not as a direct measure of test performance.

### What is saved per fold by `cv_pipeline.py`

When the `artifacts` flags are enabled in the YAML config, for each outer fold
the pipeline saves the following files inside the experiment result directory:

| File | Content |
|---|---|
| `params_fold_{i}.json` | Best hyperparameters, inner selection score, inner train score, test metrics, training metrics, time elapsed, inner split strategy |
| `model_fold_{i}.joblib` | The fitted scikit-learn estimator (or Pipeline), refitted on the full outer training set with the selected hyperparameters |
| `complexity_fold_{i}.json` | Model complexity metrics: for DT — effective depth, number of nodes, number of leaves, features used, ccp_alpha, minimum feature depth statistics; for LR — L1/L2 norm, number of non-zero coefficients, sparsity, C, l1_ratio; for SVM — L2 norm, approximate margin (1/‖w‖₂), C, number of support vectors |
| `feature_importance_fold_{i}.csv` | Per-feature importance: for linear models — raw coefficient, absolute coefficient, normalized absolute importance, rank; for Decision Tree — impurity-based importance, minimum split depth, rank |
| `cv_results_fold_{i}.csv` | Full grid/random search results from the inner hyperparameter selection, including mean and std of train and validation scores for every hyperparameter combination |

These files are required as inputs for the protocol comparison notebooks.

### Notebook structure and outputs

For each of the four Hi datasets, the analysis is organized into six notebooks
under `notebooks/{dataset}_hi_ood_vs_random_shuffle/`. The same structure is
replicated identically across DRD2, HIV, KDR, and Sol.

---

#### Notebook 01 — `01_build_protocol_tables_{dataset}.ipynb`

Loads the `params_fold_{i}.json` files for all experiments and builds the
comparison tables used by the plot notebooks.

**Tables saved** to `results/results_ood_vs_random_shuffle/hi/{dataset}/`:

| File | Content |
|---|---|
| `protocol_per_fold.csv` | One row per (model, fingerprint, protocol, fold). Columns: inner PR-AUC, inner train PR-AUC, outer train PR-AUC, final OOD test PR-AUC, inner-test gap, train-test gap, best hyperparameters (dict), inner split strategy, time |
| `protocol_summary_numeric.csv` | Aggregated over folds: mean ± std for all score and gap columns, per (model, fingerprint, protocol) |
| `protocol_summary_display.csv` | Human-readable version with "mean ± std" strings |
| `protocol_delta.csv` | Per (model, fingerprint): delta = random_shuffle − OOD_holdout for inner score, test score, inner-test gap, train-test gap |
| `hyperparameters_all.csv` | Flattened per-fold hyperparameter table for all models |
| `hyperparameters_dt.csv` | Decision Tree: ccp_alpha, max_depth, max_features, min_samples_leaf, min_samples_split, class_weight, inner score, test score |
| `hyperparameters_lr.csv` | Logistic Regression: C, l1_ratio, class_weight, inner score, test score |
| `hyperparameters_svm.csv` | Linear SVM: C, class_weight, inner score, test score |
| `hyperparameters_set_summary.csv` | Unique hyperparameter values selected across all folds, per (model, protocol) |

---

#### Notebook 02 — `02_protocol_plots_{dataset}.ipynb`

Produces four figures from the tables built by notebook 01.

**Figure 01 — Scatter: inner validation vs final OOD test.**
Each point is one (model, fingerprint, protocol) combination. The x-axis is
the mean inner validation PR-AUC; the y-axis is the mean final OOD test
PR-AUC. The identity diagonal separates well-calibrated protocols (near
diagonal) from optimistic ones (far right of diagonal). Color encodes the
protocol (blue = OOD holdout, red = random shuffle); marker shape encodes the
model family (circle = DT, square = LR, triangle = SVM). A shaded region
below the diagonal marks the "optimistic inner estimate" zone.

**Figure 02 — Grouped barplot: test PR-AUC with inner score overlay.**
Bars show the mean final OOD test PR-AUC (with standard deviation error
bars). Diamond markers overlaid on each bar show the mean inner validation
PR-AUC. A dotted vertical line connects the bar top to the diamond, making
the optimism gap directly readable as a visual distance. One bar pair (OOD,
random) per (model, fingerprint) combination.

**Figure 03 — Fold-wise gap panel (1×3 subplots, one per model).**
For each model, shows how the inner-test gap (inner PR-AUC − OOD test PR-AUC)
varies across the three outer folds. Each line is a (protocol, fingerprint)
combination. Linestyle encodes the fingerprint (solid = ECFP4, dashed =
MACCS). Color encodes the protocol. Shared y-axis across panels. This plot
reveals whether the optimism gap is consistent across folds or fold-specific.

**Figure 04 — Delta heatmap.**
A single heatmap where rows are (model, fingerprint) combinations and columns
are four delta metrics: Δ inner PR-AUC, Δ test PR-AUC, Δ inner-test gap, Δ
train-test gap (all defined as random_shuffle − OOD_holdout). A diverging
colormap (`RdBu_r`) centered at zero is used: red = random shuffle scores
higher / larger gap, blue = OOD holdout scores higher / larger gap. Cell
annotations show values with sign.

Saved to `results/results_ood_vs_random_shuffle/hi/{dataset}/figures/`.

---

#### Notebook 03 — `03_model_complexity_tables_{dataset}.ipynb`

Loads the `complexity_fold_{i}.json` and `params_fold_{i}.json` files for all
experiments and builds model-specific complexity tables.

**Tables saved** to `results/results_ood_vs_random_shuffle/hi/{dataset}/`:

| File | Content |
|---|---|
| `complexity_all.csv` | All experiments × folds, with all complexity metrics and performance scores in a single flat table |
| `complexity_dt.csv` | Decision Tree: ccp_alpha, max_depth, effective_depth, n_nodes, n_leaves, n_features_used, used_feature_fraction, feature_min_depth_mean, feature_min_depth_std, inner/test PR-AUC, gaps |
| `complexity_lr.csv` | Logistic Regression: C, l1_ratio, class_weight, n_nonzero_coefficients, sparsity, l1_norm, l2_norm, inner/test PR-AUC, gaps |
| `complexity_svm.csv` | Linear SVM: C, class_weight, l2_norm, approx_margin (= 1/‖w‖₂), n_nonzero_coefficients, inner/test PR-AUC, gaps |
| `complexity_gap_analysis.csv` | Unified view of train PR-AUC, inner PR-AUC, inner train PR-AUC, OOD test PR-AUC, train-inner gap, inner-test gap, train-test gap, per (model, fingerprint, protocol, fold) |
| `complexity_summary.csv` | Aggregated (mean ± std across folds) for all available complexity metrics, per (model, fingerprint, protocol) |

---

#### Notebook 04 — `04_model_complexity_plots_{dataset}.ipynb`

Produces four figures from the complexity tables.

**Figure 01 — Complexity by protocol (1×3 panel, one per model family).**
Strip plots: each point is one outer fold, horizontal lines show the mean.
Decision Tree panel shows number of tree nodes; Logistic Regression panel
shows L2 norm of the coefficient vector; SVM panel shows L2 norm of the
weight vector. The x-axis groups by fingerprint. Color encodes the protocol.
This figure directly answers whether random shuffle selects more complex models.

**Figure 02 — Complexity vs optimism gap (1×3 scatter panel).**
For each model family, a scatter plot with the complexity indicator on the
x-axis (n_nodes for DT, l2_norm for LR and SVM) and the inner-test gap on
the y-axis. Color encodes the protocol; marker shape encodes the fingerprint.
A horizontal reference line marks gap = 0. This figure tests whether more
complex models tend to show a larger optimism gap.

**Figure 03 — Train → inner → test PR-AUC profile (1×3 line panel).**
For each model family, every (fingerprint, protocol, fold) combination is
shown as a line connecting three x-positions: Train, Inner val, OOD test.
Linestyle encodes the fingerprint; color encodes the protocol. The y-axis is
shared across panels. A steep drop from inner val to OOD test is the visual
signature of an optimistic inner validation. This replaces three separate
bar charts and shows the full three-stage profile in a compact format.

**Figure 04 — Hyperparameter heatmap.**
A single heatmap comparing the mean hyperparameter values selected by each
protocol. Rows are (model, fingerprint) combinations; columns are the key
hyperparameters: n_nodes, effective depth, ccp_alpha for DT; C, L2 norm,
sparsity for LR; C, L2 norm, margin for SVM. Cell color shows log₂(random /
OOD) — red means random shuffle selected a higher value, blue means OOD
holdout selected a higher value. Cell text shows "OOD value | Random value"
for direct numerical comparison.

Saved to `results/results_ood_vs_random_shuffle/hi/{dataset}/figures_complexity/`.

---

#### Notebook 05 — `05_feature_explainability_tables_{dataset}.ipynb`

Loads the `feature_importance_fold_{i}.csv` and `model_fold_{i}.joblib` files
and builds feature-level comparison tables. Implements functions from
`utils/explainability.py`:
- `extract_base_model` — unwraps sklearn Pipelines (needed for RDKit
  descriptors, which use a scaler)
- `transform_features_if_pipeline` — applies preprocessing steps before the
  final estimator
- `compute_linear_contributions` — computes local contributions xⱼ × wⱼ for
  a single molecule
- `compute_topk_overlap` — computes |A ∩ B| / k for two sets of feature indices

**Tables saved** to `results/results_ood_vs_random_shuffle/hi/{dataset}/`:

| File | Content |
|---|---|
| `feature_importance_all.csv` | Full per-feature importance table for all experiments × folds: feature index, name, raw/abs/normalized importance or tree importance, minimum split depth (DT), rank, plus experiment metadata and performance scores |
| `feature_topk.csv` | Filtered to the top-k features (k ∈ {10, 20, 50}) per (model, fingerprint, protocol, fold) |
| `feature_overlap_protocol.csv` | For each (model, fingerprint, fold, k): number and fraction of top-k features shared between OOD holdout and random shuffle |
| `feature_stability_intra_protocol.csv` | For each (model, fingerprint, protocol, fold_pair, k): top-k overlap between all pairs of outer folds within the same protocol (fold pairs: 1v2, 1v3, 2v3) |
| `feature_importance_summary.csv` | Per (model, fingerprint, protocol, fold): n_features, n_nonzero, mean/max importance, cumulative importance at top-10/20/50, DT minimum depth stats |
| `local_molecule_candidates.csv` | For each (model, fingerprint, fold): up to one molecule per disagreement category — both correct, both wrong, OOD correct & random wrong, OOD wrong & random correct. Selected as the molecule with the largest absolute difference in predicted score between the two protocols |
| `local_feature_contributions.csv` | For each selected local candidate molecule: top-20 features by absolute contribution xⱼ × wⱼ, with feature index, name, value, weight, contribution, direction (toward_active / toward_inactive), and experiment metadata |

---

#### Notebook 06 — `06_feature_explainability_plots_{dataset}.ipynb`

Produces five figures from the feature tables.

**Figure 01 — Top-k protocol overlap barplot (1×3 panel, one per model).**
For each model family, bars show the mean overlap percentage between OOD and
random shuffle top-k features, across the three outer folds. Individual fold
values are shown as scatter points on each bar. The x-axis groups by
(fingerprint, k) combinations. This directly answers: do the two protocols
rank the same features as most important?

**Figure 02 — Cumulative importance curves (1×3 panel, one per model).**
For each model family, lines show the cumulative normalized importance as a
function of the number of features (sorted by decreasing importance). Each
line is one (fingerprint, protocol, fold) combination; color encodes the
protocol, linestyle encodes the fingerprint. A steep rise means importance
is concentrated on few features. Horizontal reference lines mark the 0.5 and
0.8 thresholds. X-axis is capped at 200 features for readability.

**Figure 03 — Intra-protocol stability summary heatmap.**
A single heatmap where rows are (model, fingerprint, protocol) combinations
and columns are k ∈ {10, 20, 50}. Each cell shows the mean top-k overlap
across the three fold pairs (1v2, 1v3, 2v3), averaged within the same
protocol. A `YlGnBu` colormap is used (0–100%). Horizontal lines separate
model families. This replaces seven separate per-model×fingerprint heatmaps
and answers: are the selected features stable across folds within each
protocol?

**Figure 04 — Decision Tree depth and importance (1×2 panel).**
Left panel: boxplot of minimum split depth for each feature used in the tree,
per (fingerprint, protocol). The y-axis is inverted so that shallower features
(closer to the root, more globally influential) appear at the top. Right
panel: boxplot of absolute importance for the top-50 features, per
(fingerprint, protocol). Shows whether random shuffle selects trees where
important features are used at greater depth, and whether importance is more
or less concentrated.

**Figure 05 — Local feature contributions (up to 6 case studies).**
For each selected molecule from `local_molecule_candidates.csv`, a side-by-side
horizontal barplot shows the top-15 feature contributions (xⱼ × wⱼ) for the
OOD holdout model (left) and the random shuffle model (right), using the
same molecule and the same test fold. Green bars indicate contributions toward
active, red bars toward inactive. The selection strategy prioritizes diversity
across disagreement categories, model families, and fingerprints.

Saved to `results/results_ood_vs_random_shuffle/hi/{dataset}/figures_feature_explainability/`.

---

### Results — OOD vs random shuffle final test PR-AUC

The table below reports the mean ± std final OOD test PR-AUC across the three
outer folds, for each model, fingerprint, and inner validation protocol.
Bold marks the better protocol for each row.

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

The effect of the validation protocol is strongly dataset-dependent. On
KDR-Hi, OOD holdout consistently and dramatically outperforms random shuffle
across all model families and fingerprints, by 15–35 percentage points in
final test PR-AUC. This suggests that KDR-Hi has a particularly sharp
chemical distribution shift between inner training and test, and that
in-distribution validation is especially misleading in this setting. On
DRD2-Hi, OOD holdout is better for DT but the picture is mixed for linear
models. On HIV-Hi and Sol-Hi, differences are smaller and less consistent in
direction, indicating that the severity of the validation protocol effect
depends on the specific molecular target and the difficulty of the OOD shift.

---

## OOD vs in-distribution validation — Lo task

The OOD holdout strategy described above is specific to the Hi task and cannot
be directly applied to Lo. The reason is structural: Hi folds are explicitly
constructed as three mutually dissimilar subsets (F1, F2, F3), where each
subset is chemically distant from the others by design. This complementary
structure is what allows the reconstruction of a chemically separated inner
holdout from the test sets of the other outer folds.

Lo folds have a fundamentally different logic. The Lo task evaluates the
ability to rank structurally similar molecules within clusters of chemical
analogues. The train/test split in Lo is organized around these clusters, not
around global chemical dissimilarity. As a result, the three Lo outer folds
are not three mutually dissimilar subsets: they are organized so that each
test fold contains clusters that are not present in the corresponding training
fold, but the split criterion is cluster membership, not pairwise molecular
distance.

Using `test_1.csv` and `test_2.csv` as inner train/validation for outer fold 3
in Lo would not produce a meaningful OOD holdout, because "OOD" in Lo means
"a different cluster of analogues", not "a globally novel scaffold". The inner
validation would not be chemically representative of the outer test in the
relevant sense.

For this reason, the code explicitly blocks the holdout strategy for Lo:

```python
if inner_split_strategy == "holdout" and task != "hi":
    raise ValueError(
        "The OOD holdout strategy based on test_1/test_2/test_3 reconstruction "
        "is currently valid only for Hi tasks. For Lo, use 'kfold' or 'random_shuffle', "
        "or implement a dedicated cluster-aware holdout."
    )
```

A meaningful future extension for Lo would be a **cluster-aware inner
holdout**: hold out entire clusters of analogues during inner validation, so
that the model must generalize to ranking a new cluster it has not seen during
training. This would be the Lo analogue of the OOD holdout for Hi, but it
requires careful design around cluster sizes and the intra-cluster Spearman
evaluation protocol, and represents a separate experiment.

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
│           └── figures_feature_explainability/  # 5 explainability plots
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
for the MLP notebooks but not required for the classical models.

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

```bash
python training/train_model.py \
    --config configs/hi/drd2/svm/svm_linear_drd2_hi.yaml

# Run only a subset of outer folds
python training/train_model.py --config <path> --folds 1 2

# Validate config without running
python training/train_model.py --config <path> --dry-run
```

A YAML config for the OOD vs random comparison has the following structure:

```yaml
experiment:
  name: svm_linear_drd2_hi_inner_ood_holdout
  task: hi
  dataset: drd2

fingerprint:
  types: [ecfp4, maccs]

model:
  name: svm
  fixed:
    kernel: linear
  search:
    C: [0.001, 0.01, 0.1, 0.25, 0.5, 1, 5, 10, 25, 50, 100]
    class_weight: [null, balanced]

cv:
  scoring: average_precision
  search_strategy: grid
  inner_split_strategy: holdout    # "kfold" | "holdout" | "random_shuffle"
  holdout_val_fraction: 0.2        # only used with random_shuffle
  random_state: 42

artifacts:
  save_model: true
  save_complexity: true
  save_feature_importance: true
  save_cv_results: true
```

### 2. MLP experiments

Implemented as Jupyter notebooks under `notebooks/mlp/`, one per dataset ×
fingerprint. Pipeline code in `utils/mlp_utils.py` (Hi) and
`utils/mlp_utils_lo.py` (Lo).

### 3. OOD vs random shuffle comparison

Run both protocol configs for each model and dataset, then open the analysis
notebooks in order:

```bash
python training/train_model.py \
    --config configs/hi/drd2/lr/lr_drd2_hi_inner_ood_holdout.yaml
python training/train_model.py \
    --config configs/hi/drd2/lr/lr_drd2_hi_random_shuffle.yaml

# Then run notebooks 01 through 06 in:
# notebooks/drd2_hi_ood_vs_random_shuffle/
```

---

## Evaluation metrics

- **Hi task** — PR-AUC (primary), ROC-AUC, BEDROC (α = 70), F1 at threshold
  0.5 when applicable. Aggregated as mean ± std across the three outer folds.

- **Lo task** — Mean Spearman rank correlation computed inside each test
  cluster and averaged across clusters. Mean intra-cluster R² and MAE are
  also reported. Clusters with fewer than three molecules are skipped.

All metric definitions are in `utils/metrics.py`.

---

## Results

### Hi Task (Hit Identification) — PR-AUC

| Model | DRD2-Hi | HIV-Hi | KDR-Hi | Sol-Hi |
|---|---:|---:|---:|---:|
| Dummy | 0.6765 ± 0.0614 | 0.0399 ± 0.0143 | 0.6092 ± 0.0814 | 0.2154 ± 0.0083 |
| LR (ECFP4) | 0.7743 ± 0.0841 | 0.0714 ± 0.0307 | 0.6603 ± 0.0642 | 0.4813 ± 0.0502 |
| LR (MACCS) | 0.7327 ± 0.0641 | 0.0988 ± 0.0491 | 0.6018 ± 0.0907 | 0.4740 ± 0.0172 |
| LR (RDKit Desc.) | **0.7922 ± 0.0603** | 0.1123 ± 0.0538 | 0.6450 ± 0.0784 | 0.5955 ± 0.0246 |
| KNN (ECFP4) | 0.7078 ± 0.0486 | 0.0656 ± 0.0261 | 0.6473 ± 0.0812 | 0.4466 ± 0.0300 |
| KNN (MACCS) | 0.7100 ± 0.0496 | 0.0717 ± 0.0335 | 0.6284 ± 0.0886 | 0.4264 ± 0.0341 |
| DT (ECFP4) | 0.7145 ± 0.0712 | 0.0450 ± 0.0140 | 0.6421 ± 0.0252 | 0.3298 ± 0.0220 |
| DT (MACCS) | 0.6872 ± 0.0481 | 0.0682 ± 0.0353 | 0.6302 ± 0.0903 | 0.3317 ± 0.0283 |
| SVM Linear (ECFP4) | 0.7732 ± 0.0876 | 0.0623 ± 0.0315 | 0.6498 ± 0.0788 | 0.4791 ± 0.0645 |
| SVM Linear (MACCS) | 0.7293 ± 0.0561 | 0.1125 ± 0.0510 | 0.5824 ± 0.0854 | 0.4766 ± 0.0191 |
| SVM Poly (ECFP4) | 0.7454 ± 0.0869 | 0.0851 ± 0.0405 | 0.6104 ± 0.1532 | 0.5021 ± 0.0442 |
| SVM Poly (MACCS) | 0.7326 ± 0.0463 | 0.1058 ± 0.0530 | 0.5960 ± 0.1101 | 0.4671 ± 0.0369 |
| SVM Poly (RDKit Desc.) | 0.7750 ± 0.0615 | 0.0840 ± 0.0397 | **0.6744 ± 0.0608** | 0.5781 ± 0.0244 |
| SVM RBF (ECFP4) | 0.7728 ± 0.0790 | 0.0953 ± 0.0426 | 0.6690 ± 0.0249 | 0.4916 ± 0.0513 |
| SVM RBF (MACCS) | 0.7321 ± 0.0477 | 0.1101 ± 0.0580 | 0.6178 ± 0.0534 | 0.4729 ± 0.0368 |
| SVM RBF (RDKit Desc.) | 0.7574 ± 0.0609 | 0.1020 ± 0.0505 | 0.6547 ± 0.0627 | 0.5875 ± 0.0431 |
| SVM Tanimoto (ECFP4) | 0.7745 ± 0.0782 | 0.0827 ± 0.0331 | 0.6723 ± 0.0436 | 0.4849 ± 0.0473 |
| SVM Tanimoto (MACCS) | 0.7319 ± 0.0498 | 0.0966 ± 0.0468 | 0.5829 ± 0.0815 | 0.4747 ± 0.0400 |
| RF (ECFP4) | 0.7471 ± 0.0646 | 0.1105 ± 0.0624 | 0.6547 ± 0.0701 | 0.4824 ± 0.0391 |
| RF (MACCS) | 0.7238 ± 0.0516 | 0.1348 ± 0.0766 | 0.6125 ± 0.0772 | 0.4646 ± 0.0263 |
| RF (RDKit Desc.) | 0.7671 ± 0.0689 | **0.1491 ± 0.1148** | 0.6556 ± 0.0574 | 0.5476 ± 0.0301 |
| GB (ECFP4) | 0.7450 ± 0.0835 | 0.1053 ± 0.0560 | 0.6642 ± 0.0604 | 0.4680 ± 0.0199 |
| GB (MACCS) | 0.7419 ± 0.0639 | 0.1186 ± 0.0604 | 0.5891 ± 0.0938 | 0.4853 ± 0.0450 |
| GB (RDKit Desc.) | 0.7809 ± 0.0565 | **0.1511 ± 0.0805** | 0.6562 ± 0.0608 | 0.5691 ± 0.0178 |
| XGBoost (ECFP4) | 0.7552 ± 0.0853 | 0.0972 ± 0.0463 | 0.6255 ± 0.0662 | 0.4858 ± 0.0253 |
| XGBoost (MACCS) | 0.7376 ± 0.0643 | 0.1142 ± 0.0653 | 0.6214 ± 0.0836 | 0.5025 ± 0.0264 |
| XGBoost (RDKit Desc.) | 0.7726 ± 0.0479 | 0.1199 ± 0.0695 | 0.6615 ± 0.0734 | 0.5656 ± 0.0197 |
| MLP (ECFP4) | 0.7689 ± 0.0803 | 0.0663 ± 0.0220 | 0.6403 ± 0.0468 | 0.4575 ± 0.0254 |
| MLP (RDKit Desc.) | 0.7621 ± 0.0748 | 0.1044 ± 0.0457 | 0.6291 ± 0.0886 | **0.5972 ± 0.0260** |

### Lo Task (Lead Optimization) — Mean Spearman

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
| SVM Poly (ECFP4) | 0.2286 ± 0.0264 | 0.4426 ± 0.0235 | 0.1371 ± 0.0189 |
| SVM Poly (MACCS) | 0.2301 ± 0.0353 | 0.0797 ± 0.0094 | 0.0982 ± 0.0064 |
| SVM Poly (RDKit Desc.) | 0.2695 ± 0.0350 | 0.3657 ± 0.0130 | 0.0738 ± 0.0247 |
| SVM RBF (ECFP4) | 0.1734 ± 0.0082 | **0.4453 ± 0.0205** | 0.1273 ± 0.0204 |
| SVM RBF (MACCS) | 0.2483 ± 0.0224 | 0.1335 ± 0.0244 | 0.0651 ± 0.0171 |
| SVM RBF (RDKit Desc.) | 0.2021 ± 0.0195 | 0.0170 ± 0.0275 | 0.1391 ± 0.0179 |
| SVM Tanimoto (ECFP4) | 0.1820 ± 0.0131 | 0.3919 ± 0.0137 | 0.1398 ± 0.0153 |
| SVM Tanimoto (MACCS) | 0.2480 ± 0.0418 | 0.1186 ± 0.0202 | 0.0575 ± 0.0330 |
| RF (ECFP4) | **0.3188 ± 0.0255** | 0.3458 ± 0.0263 | 0.1070 ± 0.0207 |
| RF (MACCS) | 0.1878 ± 0.0388 | 0.1598 ± 0.0208 | 0.1227 ± 0.0491 |
| RF (RDKit Desc.) | 0.2141 ± 0.0222 | 0.4127 ± 0.0421 | **0.1691 ± 0.0268** |
| GB (ECFP4) | 0.2660 ± 0.0362 | 0.4008 ± 0.0272 | 0.0748 ± 0.0098 |
| GB (MACCS) | 0.2015 ± 0.0298 | 0.1914 ± 0.0251 | 0.1184 ± 0.0146 |
| GB (RDKit Desc.) | 0.1742 ± 0.0538 | 0.3914 ± 0.0553 | 0.1527 ± 0.0456 |
| XGBoost (ECFP4) | 0.2943 ± 0.0530 | 0.4188 ± 0.0100 | 0.0715 ± 0.0102 |
| XGBoost (MACCS) | 0.1967 ± 0.0240 | 0.1358 ± 0.0558 | 0.0853 ± 0.0526 |
| XGBoost (RDKit Desc.) | 0.2314 ± 0.0809 | **0.4508 ± 0.0287** | 0.1434 ± 0.0268 |
| MLP (ECFP4) | 0.2757 ± 0.0406 | 0.3970 ± 0.0188 | 0.1507 ± 0.0293 |
| MLP (RDKit Desc.) | 0.2732 ± 0.0248 | 0.4147 ± 0.0551 | 0.1098 ± 0.0167 |

---

## Use of large language models

Anthropic's Claude was used during the development of this repository,
mainly to refactor exploratory code into a more modular and professional
structure (Claude Sonnet 4.6), and to assist with debugging and analysis
design (Claude Opus 4.6).

---

## License

The datasets are released under the MIT license by the authors of the
original Lo-Hi benchmark. Code in this repository is also released under the
MIT license unless otherwise noted.