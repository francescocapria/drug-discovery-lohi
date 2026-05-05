"""
Collect all experiment results from results/ and print a clean summary.

Usage:
    python collect_results.py | tee final_results.txt
    python collect_results.py --task hi
    python collect_results.py --task lo
    python collect_results.py --dataset drd2
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict

# -----------------------------------------------------------------------
# Project root: same logic as io_utils.py
# -----------------------------------------------------------------------
def get_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "README.md").exists() or (parent / "requirements.txt").exists():
            return parent
    return current

PROJECT_ROOT = get_project_root()
RESULTS_DIR  = PROJECT_ROOT / "results"

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
METRICS_HI = ["pr_auc", "roc_auc", "bedroc", "f1_at_05"]
METRICS_LO = ["mean_spearman", "mean_r2", "mean_mae"]

def infer_task(path: Path) -> str:
    parts = path.parts
    for i, p in enumerate(parts):
        if p == "results" and i + 1 < len(parts):
            return parts[i + 1]   # "hi" or "lo"
    return "unknown"

def load_fold_jsons(model_dir: Path):
    """Return list of (fold_idx, data_dict) for all params_fold_*.json found."""
    results = []
    for f in sorted(model_dir.glob("params_fold_*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            fold = data.get("fold", int(f.stem.split("_")[-1]))
            results.append((fold, data))
        except Exception as e:
            print(f"  [WARN] Could not read {f}: {e}", file=sys.stderr)
    return results

def mean(values):
    v = [x for x in values if x is not None and isinstance(x, (int, float))]
    return sum(v) / len(v) if v else None

def fmt(v):
    if v is None:
        return "   N/A  "
    return f"{v:.4f}"

def fmt_params(params: dict) -> str:
    """Short one-line summary of best_params."""
    if not params:
        return ""
    parts = []
    for k, v in params.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.2g}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",    default=None, help="Filter by task: hi or lo")
    parser.add_argument("--dataset", default=None, help="Filter by dataset: drd2, hiv, ...")
    parser.add_argument("--model",   default=None, help="Filter by model name substring")
    args = parser.parse_args()

    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    # Collect all model directories (depth: results/task/dataset/model_name/)
    entries = []  # (task, dataset, model_name, model_dir)
    for task_dir in sorted(RESULTS_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        if args.task and task != args.task:
            continue
        for dataset_dir in sorted(task_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            if args.dataset and dataset != args.dataset:
                continue
            for model_dir in sorted(dataset_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                model_name = model_dir.name
                if args.model and args.model.lower() not in model_name.lower():
                    continue
                fold_data = load_fold_jsons(model_dir)
                if fold_data:
                    entries.append((task, dataset, model_name, model_dir, fold_data))

    if not entries:
        print("No results found.")
        sys.exit(0)

    # Group by task/dataset for pretty printing
    by_task_dataset = defaultdict(list)
    for task, dataset, model_name, model_dir, fold_data in entries:
        by_task_dataset[(task, dataset)].append((model_name, fold_data))

    sep = "=" * 100

    for (task, dataset), models in sorted(by_task_dataset.items()):
        metrics_keys = METRICS_HI if task == "hi" else METRICS_LO

        print(sep)
        print(f"  TASK: {task.upper()}   DATASET: {dataset.upper()}")
        print(sep)

        # Header
        fold_cols = "  Fold1    Fold2    Fold3    Mean "
        metric_header = "".join(f"  {m:<10}" for m in metrics_keys)
        print(f"{'Model':<55}  {'Metric':<14}  {fold_cols}")
        print("-" * 100)

        for model_name, fold_data in sorted(models):
            fold_metrics = {}  # fold_idx -> test_metrics dict
            fold_params  = {}  # fold_idx -> best_params dict

            for fold_idx, data in fold_data:
                tm = data.get("test_metrics", {})
                fold_metrics[fold_idx] = tm
                fold_params[fold_idx]  = data.get("best_params", {})

            folds = sorted(fold_metrics.keys())

            # Print one row per metric
            first = True
            for metric in metrics_keys:
                values = [fold_metrics.get(f, {}).get(metric) for f in [1, 2, 3]]
                m = mean([v for v in values if v is not None])
                row_label = model_name if first else ""
                first = False
                cols = "  ".join(fmt(v) for v in values)
                print(f"  {row_label:<53}  {metric:<14}  {cols}  {fmt(m)}")

            # Print inner_cv_score if available (first fold only)
            inner_scores = [data.get("inner_cv_score") for _, data in fold_data]
            inner_mean = mean(inner_scores)
            if inner_mean is not None:
                cols = "  ".join(fmt(s) for s in [
                    next((data.get("inner_cv_score") for f, data in fold_data if f == fi), None)
                    for fi in [1, 2, 3]
                ])
                print(f"  {'':53}  {'inner_cv_score':<14}  {cols}  {fmt(inner_mean)}")

            # Best params summary (one line per fold)
            for fold_idx in folds:
                params = fold_params.get(fold_idx, {})
                if params:
                    short = fmt_params(params)
                    if len(short) > 90:
                        short = short[:87] + "..."
                    print(f"    [Fold {fold_idx} best params] {short}")

            print()

    print(sep)
    print(f"  Total experiments: {len(entries)}")
    print(sep)


if __name__ == "__main__":
    main()