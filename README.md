# Drug Discovery Lo-Hi

A rigorous re-implementation and extension of the **Lo-Hi benchmark** for practical
machine learning in drug discovery, originally proposed by Steshin (NeurIPS 2023).

This repository provides a clean, modular pipeline to evaluate a wide range of
machine learning models — from classical baselines to multi-layer perceptrons —
on the two complementary tasks defined by the benchmark:

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

This is not a verbatim re-run of the original code. The pipeline addresses a
number of methodological issues in the original implementation and broadens
the experimental scope:

- **Methodologically correct nested cross-validation.** In the original
  implementation, hyperparameter selection was performed by monitoring metrics
  computed on the test set itself (`val_dataloaders=test_dataloader` with
  `monitor='test_spearman'`), and the selected hyperparameters were then
  reused unchanged on the remaining folds. This is a form of test-set leakage.
  In this repository each outer fold runs its own independent inner
  cross-validation on the training partition only; the test set is never
  observed during model selection or early stopping.

- **Larger fingerprint suite.** In addition to the ECFP4 and MACCS
  fingerprints used in the paper, this repository also evaluates RDKit
  topological fingerprints (path-based) and RDKit 2D descriptors. Continuous
  descriptors are standardized inside each cross-validation split to avoid
  leakage.

- **Rigorous and modular MLP implementation.** The original paper's MLPs were
  defined ad-hoc with very small architectures (4–32 hidden units), a fixed
  learning rate of 0.01, ReLU only, no normalization, no gradient clipping,
  and a hard-coded number of training epochs in the final retraining step.
  The MLPs in this repository support flat and pyramidal architectures with
  configurable depth and width, ReLU / LeakyReLU / ELU / GELU / SiLU
  activations, optional Batch Normalization and Dropout, AdamW with weight
  decay, ReduceLROnPlateau scheduling, gradient clipping, principled early
  stopping on a held-out validation split, and multi-seed ensembling for
  the final test evaluation.

- **Larger and better-structured hyperparameter searches.** The original
  paper performed a small grid search (e.g. `C ∈ {0.1, 0.5, 1.0, 2.0, 5.0}`
  for SVM). This repository runs more extensive grid and random searches
  for every model.

- **Additional SVM kernels.** Linear, polynomial and Tanimoto kernels are
  available in addition to the RBF kernel.

- **Modular pipeline for reusability.** Featurization, splitting, model
  training and evaluation are decoupled into independent modules under
  `utils/`, which makes the code easy to extend to additional models or
  datasets (for example ChEMBL36).

---

## Repository structure

```
drug-discovery-lohi/
├── configs/                # YAML experiment configs (one per model × task × dataset)
│   └── hi/
│       ├── drd2/
│       ├── hiv/
│       ├── kdr/
│       └── sol/
├── data/                   # Lo-Hi dataset folds (CSV)
│   ├── hi/{drd2,hiv,kdr,sol}/{train,test}_{1,2,3}.csv
│   └── lo/{drd2,kcnh2,kdr}/{train,test}_{1,2,3}.csv
├── features/               # Cached precomputed fingerprints (.npz)
├── notebooks/              # MLP experiments (one notebook per dataset × fingerprint)
│   ├── mlp_drd2_hi_ecfp4.ipynb
│   ├── mlp_drd2_lo_rdkit.ipynb
│   └── ...
├── results/                # Per-fold predictions and best hyperparameters
│   └── {hi,lo}/{dataset}/{model_fp}/
│       ├── train_{1,2,3}.csv
│       ├── test_{1,2,3}.csv
│       └── params_fold_{1,2,3}.json
├── splits/                 # Pre-computed split indices
├── training/
│   └── train_model.py      # Entry point for sklearn-based models
├── utils/
│   ├── fingerprints.py     # ECFP4, MACCS, RDKit topo, RDKit descriptors
│   ├── metrics.py          # Hi (PR-AUC, ROC-AUC, BEDROC) and Lo (intra-cluster Spearman)
│   ├── cv_pipeline.py      # Nested CV for sklearn estimators
│   ├── io_utils.py         # Loading, caching, saving predictions and parameters
│   ├── config_loader.py    # YAML config parsing and validation
│   ├── mlp_utils.py        # PyTorch MLP pipeline for the Hi (classification) task
│   └── mlp_utils_lo.py     # PyTorch MLP pipeline for the Lo (regression) task
├── requirements.txt
└── README.md
```

---

## Installation

The code requires Python 3.10 or newer. A CUDA-capable GPU is recommended
for the MLP notebooks but not required for the classical models in
`training/train_model.py`.

```bash
git clone https://github.com/<your-username>/drug-discovery-lohi.git
cd drug-discovery-lohi

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install it explicitly before
running `pip install -r requirements.txt`, for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

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

Each notebook is organized in five cells:

1. Imports, project root setup, device selection, global seed.
2. Configuration dictionary (task, dataset, fingerprint, outer folds,
   inner-CV folds).
3. Hyperparameter search space and fixed hyperparameters.
4. Loading and featurization of the three pre-defined folds.
5. Execution of the nested random search and printing of aggregated results.

The MLP pipeline (defined in `utils/mlp_utils.py` for Hi and
`utils/mlp_utils_lo.py` for Lo) is documented inline in the source.

To run a notebook:

```bash
jupyter lab
```

and open the desired notebook in `notebooks/`.

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

## Results

## Hi Task (Hit Identification) — PR-AUC ↑
 
| Model | DRD2-Hi | HIV-Hi | KDR-Hi | Sol-Hi |
|---|---|---|---|---|
| Dummy | 0.6765 ± 0.0614 | 0.0399 ± 0.0143 | 0.6092 ± 0.0814 | 0.2154 ± 0.0083 |
| | | | | |
| LR (ECFP4) | 0.7743 ± 0.0841 | 0.0714 ± 0.0307 | 0.6653 ± 0.0520 | 0.4813 ± 0.0502 |
| LR (MACCS) | 0.7334 ± 0.0645 | 0.1012 ± 0.0464 | 0.6129 ± 0.0848 | 0.4686 ± 0.0189 |
| LR (RDKit Desc.) | **0.7922 ± 0.0603** | 0.1175 ± 0.0566 | 0.6450 ± 0.0784 | **0.5953 ± 0.0251** |
| | | | | |
| KNN (ECFP4) | 0.7078 ± 0.0486 | 0.0656 ± 0.0261 | 0.6473 ± 0.0812 | 0.4466 ± 0.0300 |
| KNN (MACCS) | 0.7100 ± 0.0496 | 0.0717 ± 0.0335 | 0.6284 ± 0.0886 | 0.4264 ± 0.0341 |
| | | | | |
| DT (ECFP4) | 0.7145 ± 0.0712 | 0.0450 ± 0.0140 | 0.6421 ± 0.0252 | 0.3298 ± 0.0220 |
| DT (MACCS) | 0.6872 ± 0.0481 | 0.0682 ± 0.0353 | 0.6302 ± 0.0903 | 0.3317 ± 0.0283 |
| | | | | |
| SVM Linear (ECFP4) | 0.7711 ± 0.0892 | 0.0526 ± 0.0206 | 0.6599 ± 0.0624 | 0.4136 ± 0.0389 |
| SVM Linear (MACCS) | 0.7304 ± 0.0566 | 0.1012 ± 0.0464 | 0.5883 ± 0.0792 | 0.4810 ± 0.0237 |
| SVM Poly (ECFP4) | 0.7454 ± 0.0869 | 0.0851 ± 0.0405 | 0.6104 ± 0.1532 | 0.5021 ± 0.0442 |
| SVM Poly (MACCS) | 0.7326 ± 0.0463 | 0.1058 ± 0.0530 | 0.5960 ± 0.1101 | 0.4671 ± 0.0369 |
| SVM Poly (RDKit Desc.) | 0.7750 ± 0.0615 | 0.0840 ± 0.0397 | **0.6744 ± 0.0608** | 0.5781 ± 0.0244 |
| SVM RBF (ECFP4) | 0.7728 ± 0.0790 | 0.0953 ± 0.0426 | 0.6690 ± 0.0249 | 0.4916 ± 0.0513 |
| SVM RBF (MACCS) | 0.7321 ± 0.0477 | 0.1101 ± 0.0580 | 0.6178 ± 0.0534 | 0.4729 ± 0.0368 |
| SVM RBF (RDKit Desc.) | 0.7574 ± 0.0609 | 0.1020 ± 0.0505 | 0.6547 ± 0.0627 | 0.5875 ± 0.0431 |
| SVM Tanimoto (ECFP4) | 0.7745 ± 0.0782 | 0.0827 ± 0.0331 | 0.6723 ± 0.0436 | 0.4849 ± 0.0473 |
| SVM Tanimoto (MACCS) | 0.7319 ± 0.0498 | 0.0966 ± 0.0468 | 0.5829 ± 0.0815 | 0.4747 ± 0.0400 |
| | | | | |
| RF (ECFP4) | 0.7471 ± 0.0646 | 0.1105 ± 0.0624 | 0.6547 ± 0.0701 | 0.4824 ± 0.0391 |
| RF (MACCS) | 0.7238 ± 0.0516 | 0.1348 ± 0.0766 | 0.6125 ± 0.0772 | 0.4646 ± 0.0263 |
| RF (RDKit Desc.) | 0.7671 ± 0.0689 | **0.1491 ± 0.1148** | 0.6556 ± 0.0574 | 0.5476 ± 0.0301 |
| | | | | |
| GB (ECFP4) | 0.7450 ± 0.0835 | 0.1053 ± 0.0560 | 0.6642 ± 0.0604 | 0.4680 ± 0.0199 |
| GB (MACCS) | 0.7419 ± 0.0639 | 0.1186 ± 0.0604 | 0.5891 ± 0.0938 | 0.4853 ± 0.0450 |
| GB (RDKit Desc.) | 0.7809 ± 0.0565 | 0.1511 ± 0.0805 | 0.6562 ± 0.0608 | 0.5691 ± 0.0178 |
| | | | | |
| XGBoost (ECFP4) | 0.7552 ± 0.0853 | 0.0972 ± 0.0463 | 0.6255 ± 0.0662 | 0.4858 ± 0.0253 |
| XGBoost (MACCS) | 0.7376 ± 0.0643 | 0.1142 ± 0.0653 | 0.6214 ± 0.0836 | 0.5025 ± 0.0264 |
| XGBoost (RDKit Desc.) | 0.7726 ± 0.0479 | 0.1199 ± 0.0695 | 0.6615 ± 0.0734 | 0.5656 ± 0.0197 |
| | | | | |
| MLP (ECFP4) | 0.7689 ± 0.0803 | 0.0663 ± 0.0220 | 0.6403 ± 0.0468 | 0.4575 ± 0.0254 |
| MLP (RDKit Desc.) | 0.7621 ± 0.0748 | 0.1044 ± 0.0457 | 0.6291 ± 0.0886 | 0.5972 ± 0.0260 |
 
---
 
## Lo Task (Lead Optimization) — Mean Spearman ↑
 
| Model | DRD2-Lo | KCNH2-Lo | KDR-Lo |
|---|---|---|---|
| Dummy | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| | | | |
| LinReg (ECFP4) | 0.0968 ± 0.0494 | 0.1795 ± 0.0198 | 0.0214 ± 0.0704 |
| LinReg (MACCS) | 0.1315 ± 0.0390 | 0.1222 ± 0.0048 | 0.0540 ± 0.0866 |
| LinReg (RDKit Desc.) | 0.2046 ± 0.0175 | 0.3765 ± 0.0093 | 0.1049 ± 0.0711 |
| | | | |
| KNN (ECFP4) | 0.2188 ± 0.0323 | 0.1903 ± 0.0507 | 0.0787 ± 0.0492 |
| KNN (MACCS) | 0.1478 ± 0.0424 | 0.1618 ± 0.0330 | 0.0376 ± 0.0112 |
| | | | |
| DT (ECFP4) | 0.1373 ± 0.0749 | 0.0988 ± 0.0510 | 0.0067 ± 0.0387 |
| DT (MACCS) | 0.0403 ± 0.0273 | 0.1154 ± 0.0284 | 0.0382 ± 0.0866 |
| | | | |
| SVM Linear (ECFP4) | 0.1299 ± 0.0170 | 0.2113 ± 0.0127 | 0.0957 ± 0.0127 |
| SVM Linear (MACCS) | 0.1300 ± 0.0562 | 0.1479 ± 0.0113 | −0.0067 ± 0.0471 |
| SVM Poly (ECFP4) | 0.2286 ± 0.0264 | **0.4426 ± 0.0235** | 0.1371 ± 0.0189 |
| SVM Poly (MACCS) | 0.2301 ± 0.0353 | 0.0797 ± 0.0094 | 0.0982 ± 0.0064 |
| SVM Poly (RDKit Desc.) | 0.2695 ± 0.0350 | 0.3657 ± 0.0130 | 0.0738 ± 0.0247 |
| SVM RBF (ECFP4) | 0.1734 ± 0.0082 | 0.4453 ± 0.0205 | 0.1273 ± 0.0204 |
| SVM RBF (MACCS) | 0.2483 ± 0.0224 | 0.1335 ± 0.0244 | 0.0651 ± 0.0171 |
| SVM RBF (RDKit Desc.) | 0.2021 ± 0.0195 | 0.0170 ± 0.0275 | 0.1391 ± 0.0179 |
| SVM Tanimoto (ECFP4) | 0.1820 ± 0.0131 | 0.3919 ± 0.0137 | 0.1398 ± 0.0153 |
| SVM Tanimoto (MACCS) | 0.2480 ± 0.0418 | 0.1186 ± 0.0202 | 0.0575 ± 0.0330 |
| | | | |
| RF (ECFP4) | **0.3188 ± 0.0255** | 0.3458 ± 0.0263 | 0.1070 ± 0.0207 |
| RF (MACCS) | 0.1878 ± 0.0388 | 0.1598 ± 0.0208 | 0.1227 ± 0.0491 |
| RF (RDKit Desc.) | 0.2141 ± 0.0222 | 0.4127 ± 0.0421 | **0.1691 ± 0.0268** |
| | | | |
| GB (ECFP4) | 0.2660 ± 0.0362 | 0.4008 ± 0.0272 | 0.0748 ± 0.0098 |
| GB (MACCS) | 0.2015 ± 0.0298 | 0.1914 ± 0.0251 | 0.1184 ± 0.0146 |
| GB (RDKit Desc.) | 0.1742 ± 0.0538 | 0.3914 ± 0.0553 | 0.1527 ± 0.0456 |
| | | | |
| XGBoost (ECFP4) | 0.2943 ± 0.0530 | 0.4188 ± 0.0100 | 0.0715 ± 0.0102 |
| XGBoost (MACCS) | 0.1967 ± 0.0240 | 0.1358 ± 0.0558 | 0.0853 ± 0.0526 |
| XGBoost (RDKit Desc.) | 0.2314 ± 0.0809 | **0.4508 ± 0.0287** | 0.1434 ± 0.0268 |
| | | | |
| MLP (ECFP4) | 0.2757 ± 0.0406 | 0.3970 ± 0.0188 | 0.1507 ± 0.0293 |
| MLP (RDKit Desc.) | 0.2732 ± 0.0248 | 0.4147 ± 0.0551 | 0.1098 ± 0.0167 |

---

## Use of large language models

Anthropic's Claude was used during the development of this repository,
mainly to refactor exploratory code into a more modular and professional
structure, and to assist with debugging. Claude Sonnet 4.6 was used for
ordinary tasks (code clean-up, documentation drafting, routine refactoring),
while Claude Opus 4.6 was used for more complex debugging sessions
(methodological discussions on nested cross-validation, early-stopping
strategies, edge cases in cluster-aware splits). All methodological
decisions and the final code are the author's responsibility.

---

## License

The datasets are released under the MIT license by the authors of the
original Lo-Hi benchmark. Code in this repository is also released under the
MIT license unless otherwise noted.

---

## Citation

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
```
