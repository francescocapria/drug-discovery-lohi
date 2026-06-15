"""
Config loader for experiment YAML files.

Each YAML file defines ONE experiment: model + fingerprint + dataset + search space.

"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional

import logging

logger = logging.getLogger(__name__)


# Default config values

DEFAULTS = {
    "cv": {
        "inner_k": 2,
        "scoring": "average_precision",
        "search_strategy": "grid",
        "n_iter": 50,
        "random_state": 42,
        "inner_split_strategy": "kfold",  # "kfold" | "holdout" | "random_shuffle"
        "holdout_val_fraction": 0.2,  # only with random_shuffle
    },
    "artifacts": {
        "save_model": False,
        "save_complexity": False,
        "save_feature_importance": False,
        "save_cv_results": False,
    },
}


# Config loader


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate a YAML experiment config.

    config_path : Path to the YAML config file.

    Return: dict with sections: experiment, fingerprint, model, cv
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Validate
    for section in ["experiment", "fingerprint", "model"]:
        if section not in cfg:
            raise ValueError(f"Config missing required section: '{section}'")

    # Fill defaults for cv section
    if "cv" not in cfg:
        cfg["cv"] = {}
    for key, default_val in DEFAULTS["cv"].items():
        if key not in cfg["cv"]:
            cfg["cv"][key] = default_val

    # Fill defaults for artifacts section
    if "artifacts" not in cfg:
        cfg["artifacts"] = {}
    for key, default_val in DEFAULTS["artifacts"].items():
        if key not in cfg["artifacts"]:
            cfg["artifacts"][key] = default_val

    # Validate experiment section
    exp = cfg["experiment"]
    for key in ["task", "dataset"]:
        if key not in exp:
            raise ValueError(f"experiment.{key} is required")
    if exp["task"] not in ("hi", "lo"):
        raise ValueError(f"experiment.task must be 'hi' or 'lo', got '{exp['task']}'")

    # Validate inner split strategy
    valid_strategies = ("kfold", "holdout", "random_shuffle")
    strategy = cfg["cv"].get("inner_split_strategy", "kfold")
    if strategy not in valid_strategies:
        raise ValueError(
            f"cv.inner_split_strategy must be one of {valid_strategies}, got '{strategy}'"
        )

    # Validate artifacts section
    valid_artifact_keys = set(DEFAULTS["artifacts"].keys())
    for key in cfg["artifacts"]:
        if key not in valid_artifact_keys:
            raise ValueError(
                f"artifacts.{key} is not supported. "
                f"Valid keys are: {sorted(valid_artifact_keys)}"
            )

    for key, value in cfg["artifacts"].items():
        if not isinstance(value, bool):
            raise ValueError(f"artifacts.{key} must be true or false, got {value}")

    # Validate fingerprint section
    fp = cfg["fingerprint"]
    if "type" not in fp and "types" not in fp:
        raise ValueError("fingerprint.type or fingerprint.types is required")

    # Validate model section
    model = cfg["model"]
    if "name" not in model:
        raise ValueError("model.name is required")
    if "search" not in model:
        model["search"] = {}
    if "fixed" not in model:
        model["fixed"] = {}

    logger.info(
        f"Loaded config: {exp.get('name', config_path.stem)} | "
        f"model={model['name']} fp={fp.get('type', fp.get('types'))} "
        f"task={exp['task']} dataset={exp['dataset']}"
    )

    return cfg


def config_to_experiment_id(cfg: Dict[str, Any]) -> str:
    """Generate a unique experiment ID from config."""
    exp = cfg["experiment"]
    return exp.get(
        "name",
        f"{cfg['model']['name']}_{cfg['fingerprint']['type']}_{exp['dataset']}_{exp['task']}",
    )
