"""
Fold-to-fold Tanimoto distance analysis for the Lo-Hi Hi datasets.

We compare the distance between reconstructed Lo-Hi Hi folds in:

1. full ECFP4 2048-bit space;
2. activity-relevant ECFP4 subspaces from List A top-k bits;
3. random ECFP4 top-k subspaces as a negative baseline.

The goal is to check whether the fold shift mainly lives in the global molecular
structure, or also in the activity-relevant features used by the predictive models.

Main metric:
    symmetric nearest-neighbour Tanimoto distance

Baseline:
    random-pair Tanimoto distance

Negative control:
    random top-k ECFP4 bits, repeated several times

Optional diagnostic:
    scipy wasserstein_distance_nd on a subsample, with Euclidean ground metric.
    Disabled by default because it is not Tanimoto-based and can be expensive.
"""

# %%
import json
import time
import zlib
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

try:
    from scipy.stats import wasserstein_distance_nd
except Exception:
    wasserstein_distance_nd = None

import sys

try:
    START_DIR = Path(__file__).resolve().parent
except NameError:
    START_DIR = Path.cwd()

PROJECT_ROOT = START_DIR

while PROJECT_ROOT != PROJECT_ROOT.parent:
    if (
        (PROJECT_ROOT / "utils").exists()
        and (PROJECT_ROOT / "data").exists()
        and (PROJECT_ROOT / "training").exists()
    ):
        break

    PROJECT_ROOT = PROJECT_ROOT.parent
else:
    raise RuntimeError(
        "Could not find project root. Expected a parent directory containing "
        "'utils/', 'data/' and 'training/'."
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.fingerprints import compute_fingerprints

# %%
# Global settings

TASK = "hi"

DATASETS = ["drd2", "hiv", "sol"]

DATASET_LABELS = {
    "drd2": "DRD2",
    "hiv": "HIV",
    "sol": "Sol",
}

SUBSET_FILES = {
    "F1": "test_3.csv",
    "F2": "test_2.csv",
    "F3": "test_1.csv",
}

PAIRS = list(combinations(["F1", "F2", "F3"], 2))

PAIR_TO_OUTER_FOLD = {
    "F1_vs_F2": 1,
    "F1_vs_F3": 2,
    "F2_vs_F3": 3,
}

FP_TYPE = "ecfp4"
EXPECTED_ECFP4_BITS = 2048

K_VALUES = [10, 20, 50, 100, 200]

N_RANDOM_PAIRS = 50_000
N_RANDOM_BIT_REPEATS = 30

RUN_WASSERSTEIN = False
W_SUBSAMPLE = 400

RANDOM_STATE = 42

DATA_ROOT = PROJECT_ROOT / "data" / TASK
OOD_CROSS_DIR = (
    PROJECT_ROOT / "results" / "results_ood_vs_random_shuffle" / TASK / "cross_dataset"
)

OUT_ROOT = PROJECT_ROOT / "results" / "results_fold_distance" / TASK
FIG_ROOT = OUT_ROOT / "figures"
CACHE_ROOT = PROJECT_ROOT / "features" / "fold_distance" / TASK

for d in (OUT_ROOT, FIG_ROOT, CACHE_ROOT):
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
    "activity_importance",
    "abs_importance",
    "normalized_abs_importance",
    "normalized_importance",
    "tree_importance",
    "normalized_tree_importance",
    "importance",
    "abs_weight",
    "abs_coefficient",
]


# %%
# Utilities


def stable_seed(*parts):
    text = "|".join(str(p) for p in parts)
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def local_rng(*parts):
    return np.random.default_rng(stable_seed(RANDOM_STATE, *parts))


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def get_smiles_col(df):
    for col in ["smiles", "SMILES", "canonical_smiles"]:
        if col in df.columns:
            return col

    raise ValueError(
        f"No SMILES column found. Available columns: {df.columns.tolist()}"
    )


def find_importance_col(df):
    for col in IMPORTANCE_COL_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError(
        "No valid importance column found in List A. "
        f"Available columns: {df.columns.tolist()}"
    )


def normalize_model_name(x):
    if pd.isna(x):
        return x

    return MODEL_NAME_MAP.get(str(x), str(x))


def normalize_fingerprint_name(x):
    if pd.isna(x):
        return x

    return FP_MAP.get(str(x), str(x))


def protocol_match(series, protocol):
    s = series.astype(str).str.lower()

    if protocol == "OOD holdout":
        return s.str.contains("ood") | s.str.contains("holdout")

    if protocol == "Random shuffle":
        return s.str.contains("random") | s.str.contains("shuffle")

    raise ValueError(f"Unknown protocol: {protocol}")


# %%
# Load data


def load_subset_fps(dataset):
    """
    Return:
        {subset_name: (X, df)}

    X is ECFP4 2048-bit. Invalid SMILES are dropped using valid_mask.
    """
    out = {}

    print(f"\nLoading reconstructed subsets for {dataset.upper()}:")

    for subset_name, filename in SUBSET_FILES.items():
        path = DATA_ROOT / dataset / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing subset file: {path}")

        df = pd.read_csv(path).copy()
        smiles_col = get_smiles_col(df)
        smiles = df[smiles_col].astype(str).tolist()

        cache_path = CACHE_ROOT / dataset / f"{subset_name}_{FP_TYPE}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        X, valid_mask = compute_fingerprints(smiles, FP_TYPE, str(cache_path))
        elapsed = time.time() - t0

        valid_mask = np.asarray(valid_mask, dtype=bool)

        if len(valid_mask) != len(df):
            raise ValueError(
                f"valid_mask length mismatch for {dataset}/{subset_name}: "
                f"{len(valid_mask)} vs {len(df)}"
            )

        if X.shape[0] != int(valid_mask.sum()):
            raise ValueError(
                f"Feature row mismatch for {dataset}/{subset_name}: "
                f"X has {X.shape[0]} rows, valid rows are {int(valid_mask.sum())}"
            )

        if X.shape[1] != EXPECTED_ECFP4_BITS:
            raise ValueError(
                f"Expected ECFP4 with {EXPECTED_ECFP4_BITS} bits, "
                f"got {X.shape[1]} for {dataset}/{subset_name}. "
                "Delete stale caches if needed."
            )

        n_invalid = int((~valid_mask).sum())

        print(
            f"  {subset_name}: raw_n={len(valid_mask):>6}, "
            f"valid_n={X.shape[0]:>6}, invalid_n={n_invalid:>3}, "
            f"n_bits={X.shape[1]}, time={elapsed:.2f}s"
        )

        if n_invalid > 0:
            print(
                f"    Removed {n_invalid} invalid SMILES from {dataset}/{subset_name}"
            )

        df = df.loc[valid_mask].reset_index(drop=True)
        out[subset_name] = (X.astype(np.uint8), df)

    print(f"\n{dataset.upper()} subset size summary:")
    for subset_name, (X, df) in out.items():
        print(f"  {subset_name}: X={X.shape}, df={df.shape}")

    return out


# %%
# Tanimoto


def nn_max_sim(X, Y, chunk=512):
    """
    For each molecule in X, compute max Tanimoto similarity to any molecule in Y.
    """
    Y = Y.astype(np.float32, copy=False)
    y_sum = Y.sum(axis=1)

    out = np.empty(X.shape[0], dtype=np.float32)

    for start in range(0, X.shape[0], chunk):
        xb = X[start : start + chunk].astype(np.float32, copy=False)

        inter = xb @ Y.T
        union = xb.sum(axis=1)[:, None] + y_sum[None, :] - inter

        sim = np.divide(
            inter,
            union,
            out=np.zeros_like(inter, dtype=np.float32),
            where=union > 0,
        )

        out[start : start + chunk] = sim.max(axis=1)

    return out


def pair_sim(X, Y, idx_x, idx_y):
    """
    Tanimoto similarity for selected cross-fold pairs.

    Returns NaN when both restricted fingerprints are empty.
    """
    a = X[idx_x].astype(np.float32, copy=False)
    b = Y[idx_y].astype(np.float32, copy=False)

    inter = (a * b).sum(axis=1)
    union = a.sum(axis=1) + b.sum(axis=1) - inter

    sim = np.full(inter.shape, np.nan, dtype=np.float32)
    np.divide(inter, union, out=sim, where=union > 0)

    return sim


def nn_distances(XA, XB):
    """
    Symmetric nearest-neighbour Tanimoto distance.
    """
    d_ab = 1.0 - nn_max_sim(XA, XB)
    d_ba = 1.0 - nn_max_sim(XB, XA)

    sym = 0.5 * (float(d_ab.mean()) + float(d_ba.mean()))
    pooled = np.concatenate([d_ab, d_ba])

    return pooled, sym, d_ab, d_ba


def random_pair_distance(XA, XB, n_pairs, dataset, pair, space, repeat=np.nan):
    """
    Random-pair Tanimoto distance between two folds.
    """
    rng = local_rng(dataset, pair, space, repeat, "random_pairs")

    idx_a = rng.integers(0, XA.shape[0], size=n_pairs)
    idx_b = rng.integers(0, XB.shape[0], size=n_pairs)

    sim = pair_sim(XA, XB, idx_a, idx_b)
    dist = 1.0 - sim

    valid_fraction = float(np.isfinite(dist).mean())

    if np.isfinite(dist).any():
        mean_dist = float(np.nanmean(dist))
    else:
        mean_dist = np.nan

    return mean_dist, valid_fraction, int(np.isfinite(dist).sum()), int(len(dist))


# %%
# List A


def load_list_a():
    """
    Load activity-model feature importance table from OOD-vs-random analysis.
    Only ECFP4 features are kept.
    """
    path = OOD_CROSS_DIR / "cross_dataset_feature_importance_all.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"List A table not found: {path}\n"
            "Run the cross-dataset OOD-vs-random tables notebook first."
        )

    df = pd.read_csv(path, low_memory=False).copy()

    for col in ["model", "fingerprint", "feature_idx"]:
        if col not in df.columns:
            raise ValueError(
                f"List A has no '{col}' column. Columns: {df.columns.tolist()}"
            )

    importance_col = find_importance_col(df)

    df["model"] = df["model"].map(normalize_model_name)
    df["fingerprint"] = df["fingerprint"].map(normalize_fingerprint_name)

    df = df[df["fingerprint"] == FP_TYPE].copy()

    df["abs_importance"] = df[importance_col].fillna(0.0).astype(float).abs()

    df["feature_idx"] = df["feature_idx"].astype(int)

    print("\nList A loaded:")
    print(f"  path: {path}")
    print(f"  shape after ECFP4 filter: {df.shape}")
    print(f"  importance column used: {importance_col}")
    print(f"  datasets: {sorted(df['dataset'].dropna().unique())}")
    print(f"  models: {sorted(df['model'].dropna().unique())}")
    print(f"  fingerprints: {sorted(df['fingerprint'].dropna().unique())}")

    if "protocol" in df.columns:
        print("\nList A protocol counts:")
        print(df["protocol"].value_counts(dropna=False))
    else:
        print(
            "\nWARNING: List A has no protocol column. Protocol-specific top-k bits will be skipped."
        )

    if "fold" in df.columns:
        print("\nList A fold counts:")
        print(df["fold"].value_counts(dropna=False).sort_index())
    else:
        print(
            "\nWARNING: List A has no fold column. Fold-aware top-k bits will be skipped."
        )

    print("\nList A rows by dataset/model:")
    print(
        df.groupby(["dataset", "model"])
        .size()
        .reset_index(name="n_rows")
        .sort_values(["dataset", "model"])
        .to_string(index=False)
    )

    if df.empty:
        raise ValueError("List A is empty after filtering to ECFP4.")

    return df


def best_ecfp4_activity_models():
    """
    Best ECFP4 activity model per dataset, by mean final test PR-AUC.
    """
    path = OOD_CROSS_DIR / "cross_dataset_protocol_summary.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Protocol summary not found: {path}\n"
            "Run the cross-dataset OOD-vs-random tables notebook first."
        )

    df = pd.read_csv(path).copy()

    df["model"] = df["model"].map(normalize_model_name)
    df["fingerprint"] = df["fingerprint"].map(normalize_fingerprint_name)

    df = df[df["fingerprint"] == FP_TYPE].copy()

    if "test_mean" in df.columns:
        score_col = "test_mean"
    elif "test_pr_auc_mean" in df.columns:
        score_col = "test_pr_auc_mean"
    else:
        raise ValueError(
            "Could not find test score column in protocol summary. "
            f"Available columns: {df.columns.tolist()}"
        )

    rank = (
        df.groupby(["dataset", "model"], as_index=False)[score_col]
        .mean()
        .sort_values(["dataset", score_col], ascending=[True, False])
    )

    best = rank.loc[rank.groupby("dataset")[score_col].idxmax()].copy()

    print("\nBest-model ranking by dataset:")
    print(rank.to_string(index=False))

    return dict(zip(best["dataset"], best["model"]))


def topk_bits_global(list_a, dataset, model, k):
    """
    Top-k ECFP4 bits from List A, pooled over protocols and folds.
    """
    sub = list_a[
        (list_a["dataset"] == dataset)
        & (list_a["model"] == model)
        & (list_a["fingerprint"] == FP_TYPE)
    ].copy()

    if sub.empty:
        return None

    ranked = (
        sub.groupby("feature_idx", as_index=False)["abs_importance"]
        .mean()
        .sort_values("abs_importance", ascending=False)
    )

    return ranked.head(k)["feature_idx"].to_numpy(dtype=int)


def topk_bits_fold_protocol(list_a, dataset, model, fold, protocol, k):
    """
    Top-k ECFP4 bits from List A for a specific outer fold and protocol.
    """
    if "fold" not in list_a.columns or "protocol" not in list_a.columns:
        return None

    sub = list_a[
        (list_a["dataset"] == dataset)
        & (list_a["model"] == model)
        & (list_a["fingerprint"] == FP_TYPE)
        & (list_a["fold"].astype(int) == int(fold))
        & protocol_match(list_a["protocol"], protocol)
    ].copy()

    if sub.empty:
        return None

    ranked = (
        sub.groupby("feature_idx", as_index=False)["abs_importance"]
        .mean()
        .sort_values("abs_importance", ascending=False)
    )

    return ranked.head(k)["feature_idx"].to_numpy(dtype=int)


def random_topk_bits(dataset, pair, k, repeat):
    """
    Random ECFP4 bit subset baseline.

    This is the negative control used to check whether the reduction in fold
    distance is due only to dimensionality reduction, or specifically to the
    activity-relevant bits selected from List A.
    """
    rng = local_rng(dataset, pair, "random_bits", k, repeat)
    return rng.choice(EXPECTED_ECFP4_BITS, size=k, replace=False).astype(int)


# %%
# Restricted spaces and optional Wasserstein


def restrict_to_bits(X, bits):
    Xr = X[:, bits]
    valid_molecule_mask = Xr.sum(axis=1) > 0
    return Xr, valid_molecule_mask


def wasserstein_nd_optional(XA, XB, n, dataset, pair, space):
    """
    Optional Wasserstein diagnostic with Euclidean ground metric.
    """
    if not RUN_WASSERSTEIN:
        return np.nan

    if wasserstein_distance_nd is None:
        warnings.warn("scipy.stats.wasserstein_distance_nd is not available.")
        return np.nan

    rng = local_rng(dataset, pair, space, "wasserstein")

    if XA.shape[0] > n:
        XA = XA[rng.choice(XA.shape[0], size=n, replace=False)]

    if XB.shape[0] > n:
        XB = XB[rng.choice(XB.shape[0], size=n, replace=False)]

    try:
        return float(
            wasserstein_distance_nd(XA.astype(np.float64), XB.astype(np.float64))
        )
    except Exception as exc:
        warnings.warn(f"Wasserstein failed for {dataset}|{pair}|{space}: {exc}")
        return np.nan


# %%
# Distance computation


def add_distance_row(
    rows,
    hist_rows,
    dataset,
    pair,
    space,
    k,
    model,
    bit_source,
    activity_protocol,
    activity_fold,
    XA,
    XB,
    bits_used=None,
    bit_repeat=np.nan,
):
    t0 = time.time()

    pooled, sym, d_ab, d_ba = nn_distances(XA, XB)

    rp_mean, rp_valid_frac, rp_valid_n, rp_total_n = random_pair_distance(
        XA,
        XB,
        N_RANDOM_PAIRS,
        dataset,
        pair,
        space,
        bit_repeat,
    )

    wdist = wasserstein_nd_optional(XA, XB, W_SUBSAMPLE, dataset, pair, space)

    rows.append(
        {
            "dataset": dataset,
            "dataset_label": DATASET_LABELS.get(dataset, dataset.upper()),
            "pair": pair,
            "space": space,
            "k": int(k),
            "model": model,
            "bit_source": bit_source,
            "activity_protocol": activity_protocol,
            "activity_fold": activity_fold,
            "bit_repeat": bit_repeat,
            "bits_used": json.dumps(
                [] if bits_used is None else [int(b) for b in bits_used]
            ),
            "nn_sym_distance": float(sym),
            "nn_A_to_B_mean": float(d_ab.mean()),
            "nn_B_to_A_mean": float(d_ba.mean()),
            "random_pair_distance": rp_mean,
            "wasserstein_nd": wdist,
            "valid_molecule_fraction": 1.0,
            "valid_random_pair_fraction": rp_valid_frac,
            "n_valid_molecules": int(XA.shape[0] + XB.shape[0]),
            "n_total_molecules": int(XA.shape[0] + XB.shape[0]),
            "n_valid_random_pairs": rp_valid_n,
            "n_random_pairs": rp_total_n,
        }
    )

    hist_rows.append(
        pd.DataFrame(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS.get(dataset, dataset.upper()),
                "pair": pair,
                "space": space,
                "k": int(k),
                "model": model,
                "bit_source": bit_source,
                "activity_protocol": activity_protocol,
                "activity_fold": activity_fold,
                "distance": pooled,
            }
        )
    )

    print(
        f"    {space:<20} XA={str(XA.shape):>14}, XB={str(XB.shape):>14}, "
        f"nn_sym={sym:.4f}, random={rp_mean:.4f}, "
        f"valid_random_pairs={rp_valid_frac:.3f}, time={time.time() - t0:.1f}s"
    )


def add_restricted_distance_row(
    rows,
    hist_rows,
    dataset,
    pair,
    space,
    k,
    model,
    bit_source,
    activity_protocol,
    activity_fold,
    XA,
    XB,
    bits,
    bit_repeat=np.nan,
    save_hist=True,
):
    t0 = time.time()

    Ar, valid_A = restrict_to_bits(XA, bits)
    Br, valid_B = restrict_to_bits(XB, bits)

    n_valid_molecules = int(valid_A.sum() + valid_B.sum())
    n_total_molecules = int(len(valid_A) + len(valid_B))
    valid_molecule_fraction = n_valid_molecules / n_total_molecules

    rp_mean, rp_valid_frac, rp_valid_n, rp_total_n = random_pair_distance(
        Ar,
        Br,
        N_RANDOM_PAIRS,
        dataset,
        pair,
        space,
        bit_repeat,
    )

    if valid_A.sum() < 2 or valid_B.sum() < 2:
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS.get(dataset, dataset.upper()),
                "pair": pair,
                "space": space,
                "k": int(k),
                "model": model,
                "bit_source": bit_source,
                "activity_protocol": activity_protocol,
                "activity_fold": activity_fold,
                "bit_repeat": bit_repeat,
                "bits_used": json.dumps([int(b) for b in bits]),
                "nn_sym_distance": np.nan,
                "nn_A_to_B_mean": np.nan,
                "nn_B_to_A_mean": np.nan,
                "random_pair_distance": rp_mean,
                "wasserstein_nd": np.nan,
                "valid_molecule_fraction": float(valid_molecule_fraction),
                "valid_random_pair_fraction": float(rp_valid_frac),
                "n_valid_molecules": n_valid_molecules,
                "n_total_molecules": n_total_molecules,
                "n_valid_random_pairs": rp_valid_n,
                "n_random_pairs": rp_total_n,
            }
        )

        print(
            f"    {space:<20} skipped NN: valid_A={valid_A.sum()}, valid_B={valid_B.sum()}, "
            f"valid_mol={valid_molecule_fraction:.3f}, "
            f"valid_random_pairs={rp_valid_frac:.3f}, time={time.time() - t0:.1f}s"
        )
        return

    Ar_valid = Ar[valid_A]
    Br_valid = Br[valid_B]

    pooled, sym, d_ab, d_ba = nn_distances(Ar_valid, Br_valid)
    wdist = wasserstein_nd_optional(
        Ar_valid, Br_valid, W_SUBSAMPLE, dataset, pair, space
    )

    rows.append(
        {
            "dataset": dataset,
            "dataset_label": DATASET_LABELS.get(dataset, dataset.upper()),
            "pair": pair,
            "space": space,
            "k": int(k),
            "model": model,
            "bit_source": bit_source,
            "activity_protocol": activity_protocol,
            "activity_fold": activity_fold,
            "bit_repeat": bit_repeat,
            "bits_used": json.dumps([int(b) for b in bits]),
            "nn_sym_distance": float(sym),
            "nn_A_to_B_mean": float(d_ab.mean()),
            "nn_B_to_A_mean": float(d_ba.mean()),
            "random_pair_distance": rp_mean,
            "wasserstein_nd": wdist,
            "valid_molecule_fraction": float(valid_molecule_fraction),
            "valid_random_pair_fraction": float(rp_valid_frac),
            "n_valid_molecules": n_valid_molecules,
            "n_total_molecules": n_total_molecules,
            "n_valid_random_pairs": rp_valid_n,
            "n_random_pairs": rp_total_n,
        }
    )

    if save_hist:
        hist_rows.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS.get(dataset, dataset.upper()),
                    "pair": pair,
                    "space": space,
                    "k": int(k),
                    "model": model,
                    "bit_source": bit_source,
                    "activity_protocol": activity_protocol,
                    "activity_fold": activity_fold,
                    "distance": pooled,
                }
            )
        )

    print(
        f"    {space:<20} XA_valid={str(Ar_valid.shape):>14}, XB_valid={str(Br_valid.shape):>14}, "
        f"nn_sym={sym:.4f}, random={rp_mean:.4f}, "
        f"valid_mol={valid_molecule_fraction:.3f}, "
        f"valid_random_pairs={rp_valid_frac:.3f}, time={time.time() - t0:.1f}s"
    )


def compute_dataset_distances(dataset, fps, list_a, best_model):
    rows = []
    hist_rows = []

    debug_counts = {
        "global_found": 0,
        "global_missing": 0,
        "ood_found": 0,
        "ood_missing": 0,
        "random_found": 0,
        "random_missing": 0,
        "random_bits_found": 0,
    }

    for fold_a, fold_b in PAIRS:
        pair_t0 = time.time()

        XA = fps[fold_a][0]
        XB = fps[fold_b][0]

        pair = f"{fold_a}_vs_{fold_b}"
        outer_fold = PAIR_TO_OUTER_FOLD[pair]

        n_pairwise = XA.shape[0] * XB.shape[0]

        print(
            f"\n{dataset.upper()} | {pair}: "
            f"|A|={XA.shape[0]}, |B|={XB.shape[0]}, "
            f"pairwise comparisons per direction={n_pairwise:,}, "
            f"outer_fold={outer_fold}"
        )

        add_distance_row(
            rows,
            hist_rows,
            dataset,
            pair,
            "full",
            EXPECTED_ECFP4_BITS,
            best_model,
            "full_ecfp4",
            "none",
            "none",
            XA,
            XB,
            bits_used=None,
            bit_repeat=np.nan,
        )

        for k in K_VALUES:
            global_bits = topk_bits_global(list_a, dataset, best_model, k)

            if global_bits is None or len(global_bits) == 0:
                debug_counts["global_missing"] += 1
                print(
                    f"    WARNING: no GLOBAL top-{k} bits for {dataset} | {best_model} | {pair}"
                )
            else:
                debug_counts["global_found"] += 1

                add_restricted_distance_row(
                    rows,
                    hist_rows,
                    dataset,
                    pair,
                    f"global_top{k}",
                    k,
                    best_model,
                    "activity_global_best_model",
                    "pooled",
                    "pooled",
                    XA,
                    XB,
                    global_bits,
                    bit_repeat=np.nan,
                    save_hist=True,
                )

            for protocol in ["OOD holdout", "Random shuffle"]:
                protocol_short = "ood" if protocol == "OOD holdout" else "random"

                protocol_bits = topk_bits_fold_protocol(
                    list_a,
                    dataset,
                    best_model,
                    outer_fold,
                    protocol,
                    k,
                )

                if protocol_bits is None or len(protocol_bits) == 0:
                    debug_counts[f"{protocol_short}_missing"] += 1
                    print(
                        f"    WARNING: no {protocol} top-{k} bits for "
                        f"{dataset} | {best_model} | {pair} | outer_fold={outer_fold}"
                    )
                    continue

                debug_counts[f"{protocol_short}_found"] += 1

                add_restricted_distance_row(
                    rows,
                    hist_rows,
                    dataset,
                    pair,
                    f"{protocol_short}_top{k}",
                    k,
                    best_model,
                    "activity_fold_protocol",
                    protocol,
                    outer_fold,
                    XA,
                    XB,
                    protocol_bits,
                    bit_repeat=np.nan,
                    save_hist=True,
                )

            for repeat in range(N_RANDOM_BIT_REPEATS):
                rb_bits = random_topk_bits(dataset, pair, k, repeat)
                debug_counts["random_bits_found"] += 1

                add_restricted_distance_row(
                    rows,
                    hist_rows,
                    dataset,
                    pair,
                    f"random_bits_top{k}",
                    k,
                    best_model,
                    "random_ecfp4_bits",
                    "none",
                    "none",
                    XA,
                    XB,
                    rb_bits,
                    bit_repeat=repeat,
                    save_hist=False,
                )

        print(f"  {dataset.upper()} | {pair}: done in {time.time() - pair_t0:.1f}s")

    print(f"\n{dataset.upper()} distance debug counts:")
    for key, value in debug_counts.items():
        print(f"  {key}: {value}")

    dist_df = pd.DataFrame(rows)
    hist_df = pd.concat(hist_rows, ignore_index=True) if hist_rows else pd.DataFrame()

    expected_activity_rows = len(PAIRS) * (1 + 3 * len(K_VALUES))
    expected_random_bit_rows = len(PAIRS) * len(K_VALUES) * N_RANDOM_BIT_REPEATS
    expected_total_rows = expected_activity_rows + expected_random_bit_rows

    print(
        f"\n{dataset.upper()} row-count check:"
        f"\n  expected rows if all optional spaces exist: {expected_total_rows}"
        f"\n  actual rows: {len(dist_df)}"
    )

    n_full = (dist_df["space"] == "full").sum()

    if n_full != len(PAIRS):
        raise ValueError(
            f"Expected {len(PAIRS)} full rows for {dataset}, got {n_full}."
        )

    if dist_df["nn_sym_distance"].notna().sum() == 0:
        raise ValueError(f"No valid NN distances for {dataset}.")

    return dist_df, hist_df


# %%
# Figures


def set_plot_style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "axes.linewidth": 0.8,
        }
    )


def fig_distance_histograms(hist_all, k_show=50, source_prefix="global"):
    set_plot_style()

    fig, axes = plt.subplots(
        1,
        len(DATASETS),
        figsize=(4.3 * len(DATASETS), 3.7),
        sharex=True,
        sharey=True,
    )

    if len(DATASETS) == 1:
        axes = [axes]

    bins = np.linspace(0, 1, 41)

    colors = {
        "full": "#3B4CC0",
        "restricted": "#C0392B",
    }

    if source_prefix == "global":
        restricted_space = f"global_top{k_show}"
        restricted_label = f"Top-{k_show} activity bits, global"
        file_suffix = f"global_top{k_show}"
    elif source_prefix == "ood":
        restricted_space = f"ood_top{k_show}"
        restricted_label = f"Top-{k_show} activity bits, OOD"
        file_suffix = f"ood_top{k_show}"
    elif source_prefix == "random":
        restricted_space = f"random_top{k_show}"
        restricted_label = f"Top-{k_show} activity bits, random"
        file_suffix = f"random_top{k_show}"
    else:
        raise ValueError(f"Unknown source_prefix: {source_prefix}")

    for ax, dataset in zip(axes, DATASETS):
        full = hist_all[
            (hist_all["dataset"] == dataset) & (hist_all["space"] == "full")
        ]["distance"]

        restricted = hist_all[
            (hist_all["dataset"] == dataset) & (hist_all["space"] == restricted_space)
        ]["distance"]

        for data, key, label in [
            (full, "full", "Full ECFP4 2048"),
            (restricted, "restricted", restricted_label),
        ]:
            if len(data) == 0:
                continue

            ax.hist(
                data,
                bins=bins,
                density=True,
                histtype="stepfilled",
                color=colors[key],
                alpha=0.32,
                linewidth=0,
            )

            ax.hist(
                data,
                bins=bins,
                density=True,
                histtype="step",
                color=colors[key],
                linewidth=1.6,
                label=label,
            )

            ax.axvline(
                float(np.mean(data)),
                color=colors[key],
                linestyle="--",
                linewidth=1.1,
                alpha=0.9,
            )

        ax.set_title(
            DATASET_LABELS.get(dataset, dataset.upper()), fontweight="bold", pad=6
        )
        ax.set_xlabel("Nearest-neighbour Tanimoto distance")
        ax.set_xlim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Density")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle(
        "Fold-to-fold nearest-neighbour distance\n"
        "Full structure vs activity-relevant ECFP4 bits",
        fontweight="bold",
        y=1.05,
        fontsize=12,
    )

    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(
            FIG_ROOT / f"fold_nn_distance_full_vs_{file_suffix}.{ext}",
            bbox_inches="tight",
        )

    plt.show()
    plt.close(fig)


def fig_distance_vs_k(dist_all, source_prefix="global"):
    set_plot_style()

    colors = {
        "drd2": "#4C78A8",
        "hiv": "#F58518",
        "sol": "#B279A2",
    }

    if source_prefix == "global":
        label_suffix = "global activity bits"
        space_builder = lambda k: f"global_top{k}"
        file_suffix = "global"
    elif source_prefix == "ood":
        label_suffix = "OOD activity bits"
        space_builder = lambda k: f"ood_top{k}"
        file_suffix = "ood"
    elif source_prefix == "random":
        label_suffix = "random activity bits"
        space_builder = lambda k: f"random_top{k}"
        file_suffix = "random"
    else:
        raise ValueError(f"Unknown source_prefix: {source_prefix}")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for dataset in DATASETS:
        vals = []

        for k in K_VALUES:
            space = space_builder(k)
            val = dist_all[
                (dist_all["dataset"] == dataset) & (dist_all["space"] == space)
            ]["nn_sym_distance"].mean()
            vals.append(val)

        ax.plot(
            K_VALUES,
            vals,
            marker="o",
            linewidth=1.8,
            markersize=5,
            color=colors.get(dataset, "gray"),
            label=DATASET_LABELS.get(dataset, dataset.upper()),
        )

        full = dist_all[
            (dist_all["dataset"] == dataset) & (dist_all["space"] == "full")
        ]["nn_sym_distance"].mean()

        ax.axhline(
            full,
            color=colors.get(dataset, "gray"),
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )

    ax.set_xscale("log")
    ax.set_xticks(K_VALUES)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    ax.set_xlabel("Number of top activity bits retained, k")
    ax.set_ylabel("Mean symmetric NN Tanimoto distance")
    ax.set_title(
        f"Fold distance vs retained {label_suffix}\n"
        "Dashed lines = full ECFP4 reference",
        fontweight="bold",
        pad=8,
    )

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(
            FIG_ROOT / f"fold_nn_distance_vs_k_{file_suffix}.{ext}",
            bbox_inches="tight",
        )

    plt.show()
    plt.close(fig)


def fig_protocol_vs_k(dist_all):
    set_plot_style()

    colors = {
        "ood": "#2563EB",
        "random": "#DC2626",
    }

    fig, axes = plt.subplots(
        1,
        len(DATASETS),
        figsize=(4.2 * len(DATASETS), 3.7),
        sharey=True,
    )

    if len(DATASETS) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, DATASETS):
        for protocol_short, label in [
            ("ood", "OOD activity bits"),
            ("random", "Random activity bits"),
        ]:
            vals = []

            for k in K_VALUES:
                space = f"{protocol_short}_top{k}"
                val = dist_all[
                    (dist_all["dataset"] == dataset) & (dist_all["space"] == space)
                ]["nn_sym_distance"].mean()
                vals.append(val)

            ax.plot(
                K_VALUES,
                vals,
                marker="o",
                linewidth=1.7,
                markersize=4.8,
                color=colors[protocol_short],
                label=label,
            )

        full = dist_all[
            (dist_all["dataset"] == dataset) & (dist_all["space"] == "full")
        ]["nn_sym_distance"].mean()

        ax.axhline(full, color="black", linestyle="--", linewidth=1.0, alpha=0.55)

        ax.set_xscale("log")
        ax.set_xticks(K_VALUES)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_title(DATASET_LABELS.get(dataset, dataset.upper()), fontweight="bold")
        ax.set_xlabel("Top-k activity bits")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Mean symmetric NN Tanimoto distance")
    axes[-1].legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Fold distance on protocol-specific activity bits",
        fontweight="bold",
        y=1.04,
    )

    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(
            FIG_ROOT / f"fold_nn_distance_protocol_vs_k.{ext}", bbox_inches="tight"
        )

    plt.show()
    plt.close(fig)


def fig_delta_heatmap(dist_all, k_show=50, source_prefix="global"):
    set_plot_style()

    if source_prefix == "global":
        restricted_space = f"global_top{k_show}"
        title_source = f"global top-{k_show} activity bits"
        file_suffix = f"global_top{k_show}"
    elif source_prefix == "ood":
        restricted_space = f"ood_top{k_show}"
        title_source = f"OOD top-{k_show} activity bits"
        file_suffix = f"ood_top{k_show}"
    elif source_prefix == "random":
        restricted_space = f"random_top{k_show}"
        title_source = f"random top-{k_show} activity bits"
        file_suffix = f"random_top{k_show}"
    else:
        raise ValueError(f"Unknown source_prefix: {source_prefix}")

    full = dist_all[dist_all["space"] == "full"][
        ["dataset", "pair", "nn_sym_distance"]
    ].rename(columns={"nn_sym_distance": "full_distance"})

    restricted = dist_all[dist_all["space"] == restricted_space][
        ["dataset", "pair", "nn_sym_distance"]
    ].rename(columns={"nn_sym_distance": "restricted_distance"})

    merged = full.merge(restricted, on=["dataset", "pair"], how="inner")
    merged["delta_restricted_minus_full"] = (
        merged["restricted_distance"] - merged["full_distance"]
    )

    mat = merged.pivot(
        index="dataset", columns="pair", values="delta_restricted_minus_full"
    ).reindex(DATASETS)

    ordered_cols = ["F1_vs_F2", "F1_vs_F3", "F2_vs_F3"]
    mat = mat[[c for c in ordered_cols if c in mat.columns]]

    fig, ax = plt.subplots(figsize=(6.4, 2.8))

    vmax = np.nanmax(np.abs(mat.values))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 0.05

    im = ax.imshow(
        mat.values,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )

    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels([c.replace("_vs_", "–") for c in mat.columns])

    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels([DATASET_LABELS.get(d, d.upper()) for d in mat.index])

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            if pd.notna(val):
                ax.text(
                    j,
                    i,
                    f"{val:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    ax.set_title(
        f"Restricted-minus-full fold distance\n{title_source}",
        fontweight="bold",
        pad=8,
    )

    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Δ distance", rotation=270, labelpad=12)

    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(
            FIG_ROOT / f"fold_distance_delta_{file_suffix}.{ext}", bbox_inches="tight"
        )

    plt.show()
    plt.close(fig)


def fig_activity_vs_random_bits(dist_all, k_show=50):
    """
    Compare activity top-k distances against random top-k ECFP4 bit baselines.
    """
    set_plot_style()

    activity_spaces = [
        ("global_top", "Global activity bits", "#374151"),
        ("ood_top", "OOD activity bits", "#2563EB"),
        ("random_top", "Random-shuffle activity bits", "#DC2626"),
    ]

    fig, axes = plt.subplots(
        1,
        len(DATASETS),
        figsize=(4.5 * len(DATASETS), 3.8),
        sharey=True,
    )

    if len(DATASETS) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, DATASETS):
        null_space = f"random_bits_top{k_show}"

        null = dist_all[
            (dist_all["dataset"] == dataset) & (dist_all["space"] == null_space)
        ]["nn_sym_distance"].dropna()

        if len(null) == 0:
            ax.set_title(
                f"{DATASET_LABELS.get(dataset, dataset.upper())}\n(no random baseline)"
            )
            continue

        jitter_rng = local_rng(dataset, "plot", k_show)

        x_null = jitter_rng.normal(0, 0.035, size=len(null))

        ax.scatter(
            x_null,
            null,
            s=18,
            alpha=0.35,
            color="gray",
            edgecolor="none",
            label="Random top-k bits",
        )

        null_mean = float(null.mean())
        null_q05 = float(null.quantile(0.05))
        null_q95 = float(null.quantile(0.95))

        ax.hlines(null_mean, -0.18, 0.18, color="black", linewidth=2.0)
        ax.fill_between(
            [-0.18, 0.18],
            [null_q05, null_q05],
            [null_q95, null_q95],
            color="gray",
            alpha=0.15,
            linewidth=0,
        )

        for i, (prefix, label, color) in enumerate(activity_spaces, start=1):
            space = f"{prefix}{k_show}"

            vals = dist_all[
                (dist_all["dataset"] == dataset) & (dist_all["space"] == space)
            ]["nn_sym_distance"].dropna()

            if len(vals) == 0:
                continue

            x = np.full(len(vals), i)
            x = x + jitter_rng.normal(0, 0.035, size=len(vals))

            ax.scatter(
                x,
                vals,
                s=45,
                alpha=0.85,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                label=label if dataset == DATASETS[0] else None,
            )

            ax.hlines(
                float(vals.mean()),
                i - 0.18,
                i + 0.18,
                color=color,
                linewidth=2.2,
            )

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(
            ["Random\nbits", "Global\nactivity", "OOD\nactivity", "Shuffle\nactivity"],
            rotation=0,
        )

        ax.set_title(DATASET_LABELS.get(dataset, dataset.upper()), fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Symmetric NN Tanimoto distance")

    fig.suptitle(
        f"Activity top-{k_show} bits vs random top-{k_show} ECFP4 bits",
        fontweight="bold",
        y=1.04,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, -0.08),
        )

    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(
            FIG_ROOT / f"activity_vs_random_bits_top{k_show}.{ext}",
            bbox_inches="tight",
        )

    plt.show()
    plt.close(fig)


def fig_valid_fraction_vs_k(dist_all):
    """
    Show how many molecules remain informative after restricting to top-k bits.
    """
    set_plot_style()

    spaces = [
        ("global_top", "Global activity bits", "#374151"),
        ("ood_top", "OOD activity bits", "#2563EB"),
        ("random_top", "Random-shuffle activity bits", "#DC2626"),
        ("random_bits_top", "Random ECFP4 bits", "#7F8C8D"),
    ]

    fig, axes = plt.subplots(
        1,
        len(DATASETS),
        figsize=(4.4 * len(DATASETS), 3.7),
        sharey=True,
    )

    if len(DATASETS) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, DATASETS):
        for prefix, label, color in spaces:
            vals = []

            for k in K_VALUES:
                space = f"{prefix}{k}"

                sub = dist_all[
                    (dist_all["dataset"] == dataset) & (dist_all["space"] == space)
                ]

                vals.append(sub["valid_molecule_fraction"].mean())

            ax.plot(
                K_VALUES,
                vals,
                marker="o",
                linewidth=1.7,
                markersize=4.8,
                color=color,
                label=label,
            )

        ax.set_xscale("log")
        ax.set_xticks(K_VALUES)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

        ax.set_ylim(0, 1.05)
        ax.set_title(DATASET_LABELS.get(dataset, dataset.upper()), fontweight="bold")
        ax.set_xlabel("Top-k retained bits")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Valid molecule fraction")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle(
        "Coverage of restricted ECFP4 spaces",
        fontweight="bold",
        y=1.04,
    )

    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(
            FIG_ROOT / f"valid_molecule_fraction_vs_k.{ext}", bbox_inches="tight"
        )

    plt.show()
    plt.close(fig)


# %%
# Diagnostics and paper summary


def print_output_diagnostics(dist_all, hist_all):
    print_section("Output diagnostics")

    print("\nDistance table shape:", dist_all.shape)
    print("Histogram table shape:", hist_all.shape)

    print("\nDistance rows by dataset/space:")
    print(
        dist_all.groupby(["dataset", "space"])
        .size()
        .reset_index(name="n_rows")
        .sort_values(["dataset", "space"])
        .to_string(index=False)
    )

    print("\nHistogram rows by dataset/space:")
    print(
        hist_all.groupby(["dataset", "space"])
        .size()
        .reset_index(name="n_rows")
        .sort_values(["dataset", "space"])
        .to_string(index=False)
    )

    expected_activity_rows = len(DATASETS) * len(PAIRS) * (1 + 3 * len(K_VALUES))
    expected_random_bit_rows = (
        len(DATASETS) * len(PAIRS) * len(K_VALUES) * N_RANDOM_BIT_REPEATS
    )
    expected_total_rows = expected_activity_rows + expected_random_bit_rows
    actual_rows = len(dist_all)

    print(f"\nExpected activity rows: {expected_activity_rows}")
    print(f"Expected random-bit baseline rows: {expected_random_bit_rows}")
    print(f"Expected total rows: {expected_total_rows}")
    print(f"Actual rows: {actual_rows}")

    if actual_rows != expected_total_rows:
        print(
            "\nWARNING: row count differs from the fully populated expectation. "
            "This is allowed if optional protocol-specific or activity-bit spaces "
            "are missing."
        )

    if "full" not in set(dist_all["space"]):
        raise ValueError("Missing full ECFP4 distance rows.")

    if "global_top50" not in set(dist_all["space"]):
        raise ValueError("Missing global_top50 rows.")

    if "random_bits_top50" not in set(dist_all["space"]):
        raise ValueError("Missing random_bits_top50 rows.")

    if dist_all["nn_sym_distance"].notna().sum() == 0:
        raise ValueError("All NN distances are NaN.")

    print("\nOutput diagnostics completed.")


def build_paper_summary(dist_all):
    summary_rows = []

    for dataset in DATASETS:
        for k in [50, 100]:
            full_mean = dist_all[
                (dist_all["dataset"] == dataset) & (dist_all["space"] == "full")
            ]["nn_sym_distance"].mean()

            null = dist_all[
                (dist_all["dataset"] == dataset)
                & (dist_all["space"] == f"random_bits_top{k}")
            ]["nn_sym_distance"].dropna()

            null_mean = null.mean()
            null_q05 = null.quantile(0.05)
            null_q95 = null.quantile(0.95)

            for space in [f"global_top{k}", f"ood_top{k}", f"random_top{k}"]:
                sub = dist_all[
                    (dist_all["dataset"] == dataset) & (dist_all["space"] == space)
                ]

                activity_mean = sub["nn_sym_distance"].mean()
                valid_mol = sub["valid_molecule_fraction"].mean()
                valid_pair = sub["valid_random_pair_fraction"].mean()

                summary_rows.append(
                    {
                        "dataset": dataset,
                        "k": k,
                        "space": space,
                        "full_distance_mean": full_mean,
                        "activity_distance_mean": activity_mean,
                        "delta_activity_minus_full": activity_mean - full_mean,
                        "random_bits_distance_mean": null_mean,
                        "random_bits_q05": null_q05,
                        "random_bits_q95": null_q95,
                        "delta_activity_minus_random_bits": activity_mean - null_mean,
                        "valid_molecule_fraction_mean": valid_mol,
                        "valid_random_pair_fraction_mean": valid_pair,
                    }
                )

    paper_summary = pd.DataFrame(summary_rows)

    out_path = OUT_ROOT / "fold_distance_activity_vs_random_bits_summary.csv"
    paper_summary.to_csv(out_path, index=False)

    print("\nSaved paper summary table:")
    print(out_path)
    print(paper_summary.to_string(index=False))

    return paper_summary


# %%
# Main


def main():
    total_t0 = time.time()

    print_section("Fold-to-fold Tanimoto distance analysis")
    print(f"Project root        : {PROJECT_ROOT}")
    print(f"Output root         : {OUT_ROOT}")
    print(f"Figure root         : {FIG_ROOT}")
    print(f"Cache root          : {CACHE_ROOT}")
    print(f"Wasserstein         : {'ON' if RUN_WASSERSTEIN else 'OFF'}")
    print(f"Datasets            : {DATASETS}")
    print(f"K values            : {K_VALUES}")
    print(f"Random pairs        : {N_RANDOM_PAIRS:,}")
    print(f"Random-bit repeats  : {N_RANDOM_BIT_REPEATS}")

    list_a = load_list_a()
    chosen_models = best_ecfp4_activity_models()

    print("\nBest ECFP4 activity model per dataset:")
    for dataset, model in chosen_models.items():
        print(f"  {dataset}: {model}")

    dist_parts = []
    hist_parts = []

    for dataset in DATASETS:
        dataset_t0 = time.time()

        print_section(dataset.upper())

        fps = load_subset_fps(dataset)

        best_model = chosen_models.get(dataset)
        if best_model is None:
            warnings.warn(f"No best ECFP4 model found for {dataset}; skipping.")
            continue

        print(f"\nUsing List A bits from model: {best_model}")

        dist_df, hist_df = compute_dataset_distances(dataset, fps, list_a, best_model)

        dist_parts.append(dist_df)
        hist_parts.append(hist_df)

        print(f"\n{dataset.upper()} completed in {time.time() - dataset_t0:.1f}s")

    if not dist_parts:
        raise RuntimeError("No distance results were produced.")

    dist_all = pd.concat(dist_parts, ignore_index=True)
    hist_all = pd.concat(hist_parts, ignore_index=True)

    print_output_diagnostics(dist_all, hist_all)

    summary_path = OUT_ROOT / "fold_distance_summary.csv"
    hist_path = OUT_ROOT / "fold_distance_nn_distributions.csv"

    dist_all.to_csv(summary_path, index=False)
    hist_all.to_csv(hist_path, index=False)

    print(f"\nSaved summary table: {summary_path}")
    print(f"Saved NN distributions: {hist_path}")

    paper_summary = build_paper_summary(dist_all)

    fig_distance_histograms(hist_all, k_show=50, source_prefix="global")
    fig_distance_vs_k(dist_all, source_prefix="global")
    fig_protocol_vs_k(dist_all)

    fig_delta_heatmap(dist_all, k_show=50, source_prefix="global")
    fig_delta_heatmap(dist_all, k_show=50, source_prefix="ood")
    fig_delta_heatmap(dist_all, k_show=50, source_prefix="random")

    fig_activity_vs_random_bits(dist_all, k_show=50)
    fig_activity_vs_random_bits(dist_all, k_show=100)

    fig_valid_fraction_vs_k(dist_all)

    print(f"\nDone. Figures saved to: {FIG_ROOT}")
    print(f"Total time: {time.time() - total_t0:.1f}s")

    return dist_all, hist_all, paper_summary


if __name__ == "__main__":
    dist_all, hist_all, paper_summary = main()
