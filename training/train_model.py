"""
Main training script 

Usage example:
    python training/train_model.py --config configs/hi/drd2/knn_ecfp4_drd2_hi.yaml

This script:
1. Loads the YAML config of the project
2. Construct the right model --> Maps model name --> sklearn estimator factory
3. Runs the full nested CV pipeline
4. Saves predictions + params to the folder results/
"""

import sys
import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_config
from utils.cv_pipeline import run_nested_cv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_estimator_factory(model_selected: dict, task: str):
    """
    Map the model section config to an sklearn estimator factory 
    
    Returns a callable that produces a fresh, unfitted estimator with
    the fixed params already set before.
    """
    name = model_selected["name"]
    fixed = model_selected.get("fixed", {})

    if name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        def factory():
            return KNeighborsClassifier(**fixed)
        return factory

    elif name == "svm":
        from sklearn.svm import SVC
        def factory():
            return SVC(**fixed)
        return factory

    elif name == "gb":
        from sklearn.ensemble import GradientBoostingClassifier
        def factory():
            return GradientBoostingClassifier(**fixed)
        return factory

    elif name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        def factory():
            return RandomForestClassifier(**fixed)
        return factory

    elif name == "lr":
        from sklearn.linear_model import LogisticRegression
        def factory():
            return LogisticRegression(max_iter=1000, **fixed)
        return factory
    
    elif name == "dt":
        from sklearn.tree import DecisionTreeClassifier
        def factory():
            return DecisionTreeClassifier(**fixed)
        return factory

    elif name == "dummy":
        if task == "lo":
            from sklearn.dummy import DummyRegressor
            def factory():
                return DummyRegressor(strategy="mean")
            return factory
        else:
            from sklearn.dummy import DummyClassifier
            def factory():
                return DummyClassifier(**fixed)
            return factory

    elif name == "xgb":
        from xgboost import XGBClassifier
        def factory():
            return XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                **fixed,
            )
        return factory

    elif name == "lgbm":
        from lightgbm import LGBMClassifier
        def factory():
            return LGBMClassifier(verbose=-1, **fixed)
        return factory

    else:
        raise ValueError(
            f"Unknown model name: '{name}'. "
            f"Available: knn, svm, gb, rf, mlp, dummy, xgb, lgbm"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a model using nested CV on Lo-Hi benchmark"
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Which outer folds to run (default: 1 2 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without training",
    )

    args = parser.parse_args()

    # Load config (transform yaml file)
    cfg = load_config(args.config)

    # if --dry-run is used, it doesn't train nothing, only check the config. Useful for debug
    if args.dry_run:
        import json
        print(json.dumps(cfg, indent=2))
        return

    # Build estimator factory
    factory = get_estimator_factory(cfg["model"], cfg["experiment"]["task"])

    # Run nested CV 
    results = run_nested_cv(
        task=cfg["experiment"]["task"],
        dataset=cfg["experiment"]["dataset"],
        fp_type=cfg["fingerprint"]["type"],
        model_name=cfg["model"]["name"],
        estimator_factory=factory,
        param_grid=cfg["model"]["search"],
        inner_k=cfg["cv"]["inner_k"],
        scoring=cfg["cv"]["scoring"],
        search_strategy=cfg["cv"]["search_strategy"],
        n_iter=cfg["cv"]["n_iter"],
        random_state=cfg["cv"]["random_state"],
        folds=args.folds,
    )

    # Summary
    print("\n" + "=" * 60)
    print(f"EXPERIMENT COMPLETE: {results['experiment_id']}")
    print("=" * 60)
    print(f"\nAggregated test metrics:")
    for k, v in results["aggregated"].items():
        print(f"  {k}: {v}")
    print(f"\nPer-fold best params:")
    for r in results["fold_results"]:
        print(f"  Fold {r['fold']}: {r['best_params']}")
    print()


if __name__ == "__main__":
    main()