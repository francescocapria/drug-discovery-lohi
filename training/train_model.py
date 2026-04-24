"""
Main training script 

Usage example:
    python training/train_model.py --config configs/hi/drd2/knn/knn_ecfp4_drd2_hi.yaml

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
# Tanimoto kernel (for SVM on binary fingerprints)
# ---------------------------------------------------------------------------

def tanimoto_kernel(X, Y):
    XY = X @ Y.T
    X_sq = (X ** 2).sum(axis=1).reshape(-1, 1)
    Y_sq = (Y ** 2).sum(axis=1).reshape(-1, 1)
    return XY / (X_sq + Y_sq.T - XY)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_estimator_factory(model_selected: dict, task: str, fp_type: str = "ecfp4"):
    """
    Map the model section config to an sklearn estimator factory 
    
    Returns a callable that produces a fresh, unfitted estimator with
    the fixed params already set before.
    """
    name = model_selected["name"]
    fixed = model_selected.get("fixed", {})

    if name == "knn":
        if task == "lo":
            from sklearn.neighbors import KNeighborsRegressor
            def factory():
                return KNeighborsRegressor(**fixed)
            return factory
        else:
            from sklearn.neighbors import KNeighborsClassifier
            def factory():
                return KNeighborsClassifier(**fixed)
            return factory

    elif name == "svm":
        from sklearn.svm import SVC, SVR

        # Scaling only for continuous descriptors (rdkit_desc)
        use_scaling = fp_type == "rdkit_desc"

        # Tanimoto kernel for binary fingerprints
        kernel_type = fixed.get("kernel")
        if kernel_type == "tanimoto":
            fixed = {k: v for k, v in fixed.items() if k != "kernel"}
            kernel_arg = tanimoto_kernel
        else:
            kernel_arg = kernel_type

        SvmClass = SVR if task == "lo" else SVC

        def factory():
            if kernel_type == "tanimoto":
                model = SvmClass(kernel=tanimoto_kernel, **fixed)
            else:
                model = SvmClass(**fixed)

            if use_scaling:
                return Pipeline([("scaler", StandardScaler()), ("model", model)])
            return model

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
            return LogisticRegression(**fixed)
        return factory
    
    elif name == "linreg":
        from sklearn.linear_model import LinearRegression
        def factory():
            return LinearRegression(**fixed)
        return factory
    
    elif name == "dt":
        if task == "lo":
            from sklearn.tree import DecisionTreeRegressor
            def factory():
                return DecisionTreeRegressor(**fixed)
            return factory
        else:
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
            f"Available: knn, svm, gb, rf, lr, dt, dummy, xgb, lgbm"
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
    config_name = Path(args.config).stem
    
    task = cfg["experiment"]["task"] 

    if config_name.endswith(f"_{task}"):
        model_name = config_name[:-(len(task) + 1)]
    else:
        model_name = config_name  

    fp_config = cfg["fingerprint"]
    if "types" in fp_config:
        fp_list = fp_config["types"]
    elif "type" in fp_config:
        fp_list = [fp_config["type"]]
    else:
        raise ValueError("Config must have fingerprint.type or fingerprint.types")
    
    # if --dry-run is used, it doesn't train nothing, only check the config. Useful for debug
    if args.dry_run:
        import json
        print(json.dumps(cfg, indent=2))
        return

    # Run nested CV for each fingerprint
    for fp_type in fp_list:
        factory = get_estimator_factory(cfg["model"], cfg["experiment"]["task"], fp_type)

        # If the estimator is a Pipeline, add the "model__" prefix to param_grid keys
        param_grid = cfg["model"]["search"].copy()
        if isinstance(factory(), Pipeline):
            param_grid = {f"model__{k}": v for k, v in param_grid.items()}

        results = run_nested_cv(
            task=cfg["experiment"]["task"],
            dataset=cfg["experiment"]["dataset"],
            fp_type=fp_type,
            model_name=model_name,
            estimator_factory=factory,
            param_grid=param_grid,
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