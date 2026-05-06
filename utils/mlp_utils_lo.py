import copy
import time
import logging

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from utils.io_utils import get_feature_cache_path
from utils.fingerprints import compute_fingerprints
from utils.metrics import get_lo_metrics, aggregate_fold_metrics

from utils.mlp_utils import (
    set_seed,
    build_hidden_layers,
    get_activation,
    initialize_linear_layer,
    SimpleMLP,
    create_model,
    make_loader,
    train_one_epoch,
)


# -------------------------------------------------------------------------
# Data preparation
# -------------------------------------------------------------------------

def featurize_fold_lo(fold_idx, cfg, folds_data):
    """
    Convert one Lo fold from SMILES/dataframes to PyTorch tensors.
    """

    train_df = folds_data[fold_idx]["train"]
    test_df  = folds_data[fold_idx]["test"]

    train_cache = get_feature_cache_path(
        cfg["task"],
        cfg["dataset"],
        cfg["fp_type"],
        "train",
        fold_idx,
    )

    test_cache = get_feature_cache_path(
        cfg["task"],
        cfg["dataset"],
        cfg["fp_type"],
        "test",
        fold_idx,
    )

    X_train_np = compute_fingerprints(
        train_df["smiles"].tolist(),
        cfg["fp_type"],
        train_cache,
    )

    X_test_np = compute_fingerprints(
        test_df["smiles"].tolist(),
        cfg["fp_type"],
        test_cache,
    )

    X_train = torch.from_numpy(X_train_np).float()
    X_test  = torch.from_numpy(X_test_np).float()

    y_train_np = train_df["value"].values.astype(np.float32)
    y_test_np  = test_df["value"].values.astype(np.float32)

    y_train = torch.from_numpy(y_train_np).float()
    y_test  = torch.from_numpy(y_test_np).float()

    # kept only for logging/reference.
    y_mean = float(y_train_np.mean())
    y_std  = float(y_train_np.std())

    if y_std == 0:
        y_std = 1.0

    target_scaler = {
        "mean": y_mean,
        "std": y_std,
    }

    return X_train, y_train, X_test, y_test, target_scaler


def prepare_all_fold_tensors_lo(cfg, folds_data, logger=None):
    """
    Precompute fingerprints/descriptors and tensors for all outer Lo folds.

    No scaling is applied here.

    For ECFP4/MACCS:
    - no feature scaling is needed.

    For rdkit_desc:
    - feature scaling is needed, but it is done later inside the correct
      train/validation split to avoid leakage.
    """

    folds_tensors = {}

    scale_features = cfg.get("fp_type") == "rdkit_desc"

    for fold_idx in cfg["outer_folds"]:
        X_train, y_train, X_test, y_test, target_scaler = featurize_fold_lo(
            fold_idx,
            cfg,
            folds_data,
        )

        folds_tensors[fold_idx] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "target_scaler": target_scaler,
            "scale_features": scale_features,
        }

        if logger is not None:
            logger.info(
                f"Fold {fold_idx} | "
                f"X_train: {tuple(X_train.shape)}, "
                f"X_test: {tuple(X_test.shape)} | "
                f"scale_features={scale_features} | "
                f"y_mean={target_scaler['mean']:.3f} "
                f"y_std={target_scaler['std']:.3f}"
            )

    return folds_tensors


# -------------------------------------------------------------------------
# Small scaling utilities
# -------------------------------------------------------------------------

def scale_features_from_train_lo(X_train, X_other):
    """
    Fit StandardScaler on X_train only and transform X_train and X_other.

    Used for rdkit_desc only.
    """

    scaler = StandardScaler()

    X_train_np = X_train.numpy()
    X_other_np = X_other.numpy()

    X_train_scaled = scaler.fit_transform(X_train_np)
    X_other_scaled = scaler.transform(X_other_np)

    X_train_t = torch.from_numpy(X_train_scaled).float()
    X_other_t = torch.from_numpy(X_other_scaled).float()

    return X_train_t, X_other_t, scaler


def scale_target_from_train_lo(y_train_np, y_other_np):
    """
    Standardize target values using only y_train_np statistics.

    y_scaled = (y - mean_train) / std_train
    """

    y_mean = float(y_train_np.mean())
    y_std  = float(y_train_np.std())

    if y_std == 0:
        y_std = 1.0

    y_train_scaled = (y_train_np - y_mean) / y_std
    y_other_scaled = (y_other_np - y_mean) / y_std

    y_train_t = torch.from_numpy(y_train_scaled.astype(np.float32)).float()
    y_other_t = torch.from_numpy(y_other_scaled.astype(np.float32)).float()

    target_scaler = {
        "mean": y_mean,
        "std": y_std,
    }

    return y_train_t, y_other_t, target_scaler


def inverse_scale_predictions_lo(predictions, target_scaler):
    """
    Convert scaled predictions back to the original target scale.
    """

    return predictions * target_scaler["std"] + target_scaler["mean"]


def maybe_scale_inner_features_lo( X_inner_train, X_inner_val, scale_features,
):
    """
    Scale features inside one inner-CV split if needed.

    For ECFP4/MACCS:
    - scale_features=False, tensors are returned unchanged.

    For rdkit_desc:
    - scale_features=True, scaler is fitted on inner train only.
    """

    if not scale_features:
        return X_inner_train, X_inner_val, None

    return scale_features_from_train_lo(
        X_inner_train,
        X_inner_val,
    )


def should_drop_last_for_batchnorm_lo(X, batch_size, batchnorm):
    """
    If BatchNorm is active and the final batch has size 1, PyTorch can fail.

    In that case we use drop_last=True.
    """

    if not batchnorm:
        return False

    n = len(X)

    if n <= batch_size:
        return False

    last_batch_size = n % batch_size

    return last_batch_size == 1


def make_holdout_split_lo( n_samples, val_fraction=0.15, seed=42,
):
    """
    Create a simple random train/validation split for final early stopping.
    """

    rng = np.random.default_rng(seed)

    indices = rng.permutation(n_samples)

    val_size = int(round(val_fraction * n_samples))
    val_size = max(1, val_size)
    val_size = min(val_size, n_samples - 1)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return train_idx, val_idx


# -------------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------------

def predict_values(model, X, device):
    """
    Predict continuous values from the model.

    For Lo there is no sigmoid:
    the output is a real-valued regression score.
    """

    model.eval()

    with torch.no_grad():
        predictions = model(X.to(device))

    return predictions.cpu().numpy()


# -------------------------------------------------------------------------
# Training and validation
# -------------------------------------------------------------------------

def evaluate_model_lo(model, X_val, y_val, device):
    """
    Evaluate the model on validation data for Lo early stopping.

    Here we use MSE, not Spearman.
    Spearman is kept only for the final test evaluation.
    """

    predictions = predict_values(
        model,
        X_val,
        device,
    )

    y_true = y_val.numpy()

    mse = float(np.mean((y_true - predictions) ** 2))

    metrics = {
        "mse": mse,
        "neg_mse": -mse,
    }

    return metrics


def train_and_evaluate_lo( X_train, y_train, X_val, y_val, cluster_val, hp, device, seed=42,
):
    """
    Train one MLP for Lo regression.

    Loss:
    - MSELoss, because Lo has continuous activity values.

    Early stopping:
    - validation -MSE, not Spearman.
    - Spearman is used only for final test evaluation on the true Lo clusters.

    """

    set_seed(seed)

    model, hidden_layers = create_model(
        input_dim=X_train.shape[1],
        hp=hp,
    )

    model = model.to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp["lr"],
        weight_decay=hp["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    drop_last = should_drop_last_for_batchnorm_lo(
        X=X_train,
        batch_size=hp["batch_size"],
        batchnorm=hp["batchnorm"],
    )

    train_loader = make_loader(
        X_train,
        y_train,
        batch_size=hp["batch_size"],
        shuffle=True,
        seed=seed,
        drop_last=drop_last,
    )

    best_score = -np.inf
    best_state = None
    best_epoch = -1
    best_train_loss = None
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "val_mse": [],
        "val_neg_mse": [],
    }

    for epoch in range(hp["max_epochs"]):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            grad_clip=hp["grad_clip"],
            device=device,
        )

        val_metrics = evaluate_model_lo(
            model=model,
            X_val=X_val,
            y_val=y_val,
            device=device,
        )

        val_mse = val_metrics["mse"]
        val_score = val_metrics["neg_mse"]

        scheduler.step(val_score)

        history["train_loss"].append(train_loss)
        history["val_mse"].append(val_mse)
        history["val_neg_mse"].append(val_score)

        if val_score > best_score:
            best_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_train_loss = train_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= hp["patience"]:
            break

    result = {
        "best_score": best_score,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "hidden_layers": hidden_layers,
        "history": history,
    }

    return result


# -------------------------------------------------------------------------
# Hyperparameter search
# -------------------------------------------------------------------------

def sample_hyperparameters(search_space, rng):
    """
    Sample one random hyperparameter configuration.

    search_space must contain discrete lists/arrays.
    """

    hp = {}

    for name, possible_values in search_space.items():
        value = rng.choice(possible_values)

        if hasattr(value, "item"):
            value = value.item()

        hp[name] = value

    return hp


def evaluate_hyperparameters_inner_cv_lo( X_train, y_train, cluster_train, hp, fixed_hp, inner_k, device, seed, scale_features=False,
):
    """
    Evaluate one hyperparameter configuration with inner KFold CV.
    """

    splitter = KFold(
        n_splits=inner_k,
        shuffle=True,
        random_state=seed,
    )

    y_train_np_full = y_train.numpy()

    scores = []
    train_losses_at_best = []

    for inner_train_idx, inner_val_idx in splitter.split(X_train):
        X_inner_train = X_train[inner_train_idx]
        X_inner_val = X_train[inner_val_idx]

        y_inner_train_np = y_train_np_full[inner_train_idx]
        y_inner_val_np = y_train_np_full[inner_val_idx]

        X_inner_train, X_inner_val, _ = maybe_scale_inner_features_lo(
            X_inner_train=X_inner_train,
            X_inner_val=X_inner_val,
            scale_features=scale_features,
        )

        y_inner_train, y_inner_val, _ = scale_target_from_train_lo(
            y_inner_train_np,
            y_inner_val_np,
        )

        # It is not used for early stopping.
        cluster_inner_val = cluster_train[inner_val_idx]

        full_hp = {
            **hp,
            **fixed_hp,
        }

        training_result = train_and_evaluate_lo(
            X_train=X_inner_train,
            y_train=y_inner_train,
            X_val=X_inner_val,
            y_val=y_inner_val,
            cluster_val=cluster_inner_val,
            hp=full_hp,
            device=device,
            seed=seed,
        )

        scores.append(training_result["best_score"])
        train_losses_at_best.append(training_result["best_train_loss"])

    mean_score = float(np.mean(scores))
    mean_train_loss_at_best = float(np.mean(train_losses_at_best))

    return mean_score, mean_train_loss_at_best


def run_random_search_for_fold_lo( X_train, y_train, cluster_train, fold_idx, cfg, search_space, fixed_hp, n_iter, device, seed, logger=None, scale_features=False,
):
    """
    Run random search inside one outer Lo fold.
    """

    rng = np.random.default_rng(seed + fold_idx)

    search_results = []

    for iteration in range(n_iter):
        hp = sample_hyperparameters(
            search_space,
            rng,
        )

        start_time = time.time()

        mean_inner_score, mean_train_loss_at_best = evaluate_hyperparameters_inner_cv_lo(
            X_train=X_train,
            y_train=y_train,
            cluster_train=cluster_train,
            hp=hp,
            fixed_hp=fixed_hp,
            inner_k=cfg["inner_k"],
            device=device,
            seed=seed + fold_idx,
            scale_features=scale_features,
        )

        elapsed_time = time.time() - start_time

        result = {
            "hp": hp,
            "score": mean_inner_score,
            "mean_train_loss_at_best": mean_train_loss_at_best,
        }

        search_results.append(result)

        if logger is not None:
            logger.info(
                f"  [{iteration + 1}/{n_iter}] "
                f"inner MSE={-mean_inner_score:.4f} "
                f"(score={mean_inner_score:.4f}) "
                f"({elapsed_time:.0f}s) | "
                f"L={hp['n_layers']} "
                f"N={hp['n_nodes']} "
                f"r={hp['r']} "
                f"dropout={hp['dropout']} "
                f"lr={hp['lr']:.0e}"
            )

    search_results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    best_result = search_results[0]

    best_hp = {
        **best_result["hp"],
        **fixed_hp,
    }

    best_score = best_result["score"]

    # diagnostic only. For Lo we do not use this as a stopping threshold
    best_train_loss_diagnostic = best_result["mean_train_loss_at_best"]

    return best_hp, best_score, best_train_loss_diagnostic, search_results


# -------------------------------------------------------------------------
# Final retraining and test evaluation
# -------------------------------------------------------------------------

def predict_with_trained_state_lo( X_test, input_dim, hp, state_dict, device,
):
    """
    Rebuild the model, load saved weights, and predict continuous values.
    """

    model, _ = create_model(
        input_dim=input_dim,
        hp=hp,
    )

    model = model.to(device)
    model.load_state_dict(state_dict)

    predictions = predict_values(
        model,
        X_test,
        device,
    )

    return predictions


def retrain_ensemble_and_evaluate_test_lo( X_train, y_train, X_test, y_test, cluster_train, cluster_test, best_hp, best_train_loss_diagnostic, fold_idx, n_seeds, device, seed, logger=None, scale_features=False,
):
    """
    Retrain the best Lo configuration with multiple seeds and evaluate
    the ensemble on the outer test fold.

    Final retraining:
    - split the outer train fold into 85% retraining and 15% early-stopping
      validation;
    - early stopping is based on validation -MSE;
    - final test evaluation is still mean Spearman intra-cluster using the
      real test clusters.
    """

    all_test_predictions_real = []
    seed_results = []

    y_train_np = y_train.numpy()
    y_test_np = y_test.numpy()

    for seed_id in range(n_seeds):
        current_seed = seed + fold_idx * 100 + seed_id

        rt_idx, es_idx = make_holdout_split_lo(
            n_samples=len(y_train),
            val_fraction=0.15,
            seed=current_seed,
        )

        X_retrain = X_train[rt_idx]
        X_es = X_train[es_idx]

        y_retrain_np = y_train_np[rt_idx]
        y_es_np = y_train_np[es_idx]

        if scale_features:
            X_retrain, X_es, scaler = scale_features_from_train_lo(
                X_retrain,
                X_es,
            )

            X_test_scaled_np = scaler.transform(X_test.numpy())
            X_test_final = torch.from_numpy(X_test_scaled_np).float()

        else:
            X_test_final = X_test

        y_retrain, y_es, target_scaler = scale_target_from_train_lo(
            y_retrain_np,
            y_es_np,
        )

        # kept for compatibility. Not used for early stopping 
        cluster_es = cluster_train[es_idx]

        training_result = train_and_evaluate_lo(
            X_train=X_retrain,
            y_train=y_retrain,
            X_val=X_es,
            y_val=y_es,
            cluster_val=cluster_es,
            hp=best_hp,
            device=device,
            seed=current_seed,
        )

        if logger is not None:
            logger.info(
                f"  Seed {seed_id + 1}/{n_seeds}: "
                f"early-stop val MSE={-training_result['best_score']:.4f} "
                f"(score={training_result['best_score']:.4f}) "
                f"at epoch {training_result['best_epoch']}"
            )

        test_predictions_scaled = predict_with_trained_state_lo(
            X_test=X_test_final,
            input_dim=X_test_final.shape[1],
            hp=best_hp,
            state_dict=training_result["best_state"],
            device=device,
        )

        test_predictions_real = inverse_scale_predictions_lo(
            test_predictions_scaled,
            target_scaler,
        )

        all_test_predictions_real.append(test_predictions_real)

        seed_results.append({
            "seed": current_seed,
            "best_epoch": training_result["best_epoch"],
            "best_score_neg_mse": training_result["best_score"],
            "hidden_layers": training_result["hidden_layers"],
        })

    ensemble_predictions = np.mean(
        all_test_predictions_real,
        axis=0,
    )

    test_metrics = get_lo_metrics(
        y_test_np,
        ensemble_predictions,
        cluster_test,
    )

    result = {
        "test_metrics": test_metrics,
        "test_predictions": ensemble_predictions,
        "seed_results": seed_results,
    }

    return result


# -------------------------------------------------------------------------
# Full nested random-search pipeline
# -------------------------------------------------------------------------

def run_nested_random_search_lo( cfg, folds_tensors, folds_data, search_space, fixed_hp, n_iter, n_seeds, device, seed=42, logger=None,
):
    """
    Run nested CV with random search for the Lo task.

    For Lo:
    - inner model selection uses KFold;
    - early stopping uses validation -MSE;
    - final test evaluation uses the true Lo test clusters and mean Spearman.
    """

    fold_results = []

    for fold_idx in cfg["outer_folds"]:
        if logger is not None:
            logger.info(
                f"\n{'=' * 60}\n"
                f"OUTER FOLD {fold_idx}\n"
                f"{'=' * 60}"
            )

        X_train = folds_tensors[fold_idx]["X_train"]
        y_train = folds_tensors[fold_idx]["y_train"]
        X_test  = folds_tensors[fold_idx]["X_test"]
        y_test  = folds_tensors[fold_idx]["y_test"]

        scale_features = folds_tensors[fold_idx].get(
            "scale_features",
            False,
        )

        cluster_train = folds_data[fold_idx]["train"]["cluster"].values
        cluster_test  = folds_data[fold_idx]["test"]["cluster"].values

        best_hp, best_inner_score, best_train_loss_diagnostic, search_results = run_random_search_for_fold_lo(
            X_train=X_train,
            y_train=y_train,
            cluster_train=cluster_train,
            fold_idx=fold_idx,
            cfg=cfg,
            search_space=search_space,
            fixed_hp=fixed_hp,
            n_iter=n_iter,
            device=device,
            seed=seed,
            logger=logger,
            scale_features=scale_features,
        )

        if logger is not None:
            logger.info(
                f"[Fold {fold_idx}] "
                f"Best inner MSE: {-best_inner_score:.4f} "
                f"(score={best_inner_score:.4f}) | "
                f"inner train-loss diagnostic: {best_train_loss_diagnostic:.4f}"
            )

        final_result = retrain_ensemble_and_evaluate_test_lo(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cluster_train=cluster_train,
            cluster_test=cluster_test,
            best_hp=best_hp,
            best_train_loss_diagnostic=best_train_loss_diagnostic,
            fold_idx=fold_idx,
            n_seeds=n_seeds,
            device=device,
            seed=seed,
            logger=logger,
            scale_features=scale_features,
        )

        if logger is not None:
            logger.info(
                f"[Fold {fold_idx}] "
                f"Test metrics: {final_result['test_metrics']}"
            )

        fold_result = {
            "fold": fold_idx,
            "best_hp": best_hp,
            "inner_score": best_inner_score,
            "best_train_loss_diagnostic": best_train_loss_diagnostic,
            "test_metrics": final_result["test_metrics"],
            "test_predictions": final_result["test_predictions"],
            "seed_results": final_result["seed_results"],
            "search_results": search_results,
        }

        fold_results.append(fold_result)

    return fold_results


# -------------------------------------------------------------------------
# Reporting
# -------------------------------------------------------------------------

def print_final_results_lo(fold_results, title="MLP LO RESULTS"):
    """
    Print per-fold and aggregated test results for the Lo task.
    """

    test_metrics_per_fold = [
        result["test_metrics"]
        for result in fold_results
    ]

    aggregated_metrics = aggregate_fold_metrics(
        test_metrics_per_fold,
    )

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for result in fold_results:
        fold = result["fold"]
        metrics = result["test_metrics"]

        spearman = metrics.get("mean_spearman", float("nan"))

        print(
            f"Fold {fold}: "
            f"mean_spearman={spearman:.4f}"
        )

    print("\nAggregated metrics:")

    for metric_name, metric_value in aggregated_metrics.items():
        print(f"  {metric_name}: {metric_value}")

    print("\nBest hyperparameters per fold:")

    for result in fold_results:
        fold = result["fold"]
        hp = result["best_hp"]

        print(
            f"Fold {fold}: "
            f"L={hp['n_layers']} "
            f"N={hp['n_nodes']} "
            f"r={hp['r']} "
            f"act={hp['activation']} "
            f"dropout={hp['dropout']} "
            f"bn={hp['batchnorm']} "
            f"init={hp['init']} "
            f"lr={hp['lr']:.0e} "
            f"wd={hp['weight_decay']:.0e} "
            f"bs={hp['batch_size']}"
        )

    return aggregated_metrics