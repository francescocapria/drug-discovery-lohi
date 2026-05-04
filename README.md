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

Results will be added once all experiments are completed. They will be
provided as aggregated tables (mean ± std across the three outer folds)
broken down by dataset, model and fingerprint, alongside the corresponding
per-fold prediction CSVs and best-hyperparameter JSON files in `results/`.

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
