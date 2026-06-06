import copy
import time
import random
import logging

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from utils.io_utils import get_feature_cache_path
from utils.fingerprints import compute_fingerprints
from utils.metrics import get_hi_metrics, aggregate_fold_metrics

# -------------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------------


def set_seed(seed: int):
    """
    Set random seeds for Python, NumPy and PyTorch.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # cuda casuality if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------------------
# Data preparation
# -------------------------------------------------------------------------


def featurize_fold(fold_idx, cfg, folds_data):
    """
    Convert one fold from SMILES/dataframes to PyTorch tensors.

    Important:
    - For ECFP4/MACCS, the features are already binary fingerprints.
    - For RDKit descriptors, we still keep the raw descriptors here.
      Scaling is done later inside each inner split and final retraining
      step, to avoid leakage.
    """

    train_df = folds_data[fold_idx]["train"]
    test_df = folds_data[fold_idx]["test"]

    # cache paths
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

    # numpy array --> pytorch tensors
    X_train = torch.from_numpy(X_train_np).float()
    X_test = torch.from_numpy(X_test_np).float()

    # float for BCEWithLogitsLoss --> 0.0 or 1.0
    y_train = torch.from_numpy(train_df["value"].values.astype(np.float32)).float()

    y_test = torch.from_numpy(test_df["value"].values.astype(np.float32)).float()

    return X_train, y_train, X_test, y_test


def prepare_all_fold_tensors(cfg, folds_data, logger=None):
    """
    Precompute fingerprints/descriptors and tensors for all outer folds.

    No feature scaling is applied here. This is intentional: continuous descriptors,
    such as rdkit_desc, must be scaled later inside the correct train/validation split to avoid leakage.
    """

    folds_tensors = {}

    scale_features = cfg.get("fp_type") == "rdkit_desc"

    for fold_idx in cfg["outer_folds"]:
        X_train, y_train, X_test, y_test = featurize_fold(
            fold_idx,
            cfg,
            folds_data,
        )

        folds_tensors[fold_idx] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "scale_features": scale_features,
        }

        # measuring positive-negative imbalance
        pos_weight = compute_pos_weight(y_train)

        if logger is not None:
            logger.info(
                f"Fold {fold_idx} | "
                f"X_train: {tuple(X_train.shape)}, "
                f"X_test: {tuple(X_test.shape)} | "
                f"scale_features={scale_features} | "
                f"pos_weight={pos_weight.item():.3f}"
            )

    return folds_tensors


def scale_features_from_train(X_train, X_other):
    """
    Fit StandardScaler on X_train only and transform X_train and X_other.

    Used only for continuous descriptors such as rdkit_desc.
    """

    scaler = StandardScaler()

    X_train_np = X_train.numpy()
    X_other_np = X_other.numpy()

    X_train_scaled = scaler.fit_transform(X_train_np)
    X_other_scaled = scaler.transform(X_other_np)

    X_train_t = torch.from_numpy(X_train_scaled).float()
    X_other_t = torch.from_numpy(X_other_scaled).float()

    return X_train_t, X_other_t, scaler


def compute_pos_weight(y):
    """
    Compute positive-class weight for BCEWithLogitsLoss.

    pos_weight = n_negative / n_positive

    This increases the contribution of positive examples when the dataset is
    imbalanced.
    """

    n_pos = y.sum().item()
    n_total = y.numel()
    n_neg = n_total - n_pos

    if n_pos == 0:
        return torch.tensor(1.0, dtype=torch.float32)

    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def make_loader(X, y, batch_size, shuffle=True, seed=42, drop_last=False):
    """
    Build a reproducible PyTorch DataLoader from tensors.
    """

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=drop_last,
    )

    return loader


def should_drop_last_for_batchnorm(X, batch_size, batchnorm):
    """
    Decide whether to drop the final batch.
    """

    if not batchnorm:
        return False

    n = len(X)

    if n <= batch_size:
        return False

    last_batch_size = n % batch_size

    return last_batch_size == 1


# -------------------------------------------------------------------------
# Model architecture
# -------------------------------------------------------------------------


def build_hidden_layers(n_layers, n_nodes, r):
    """
    Build hidden-layer widths using a flat or pyramidal shape.

    If r = 1, all layers have the same width.
    If r < 1, the architecture is pyramidal.
    """

    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")

    # flat network
    if r == 1.0:
        width = n_nodes // n_layers
        return [max(1, width)] * n_layers  # to avoid layers with 0 neurons

    # pyramidal network formula
    # e.g. n_layers=3 , n_nodes=896 , r=0.5 --> first_width=512
    first_width = n_nodes * (1 - r) / (1 - r**n_layers)

    widths = []

    # construct every width
    # e.g n_layers = 3, first_width = 512, r = 0.5 --> [512, 256, 128]
    for layer_idx in range(n_layers):
        width = round(first_width * (r**layer_idx))
        width = max(1, int(width))
        widths.append(width)

    return widths


def get_activation(name):
    """
    Return the activation layer selected by name.
    """

    if name == "relu":
        return nn.ReLU()

    if name == "leaky_relu":
        return nn.LeakyReLU(0.01)

    if name == "elu":
        return nn.ELU()

    if name == "gelu":
        return nn.GELU()

    # SiLU(x) = x * sigmoid(x)
    if name == "silu":
        return nn.SiLU()

    raise ValueError(f"Unknown activation: {name}")


def initialize_linear_layer(layer, init_name, activation_name):
    """
    Initialize one linear layer.

    Relu and leaky_relu --> kaiming initialization
    Gelu, silu and elu --> zavier inizialization
    """

    if init_name == "kaiming":
        if activation_name == "leaky_relu":
            nn.init.kaiming_normal_(layer.weight, nonlinearity="leaky_relu")
        elif activation_name in ("gelu", "silu", "elu"):
            nn.init.xavier_uniform_(layer.weight)
        else:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    elif init_name == "xavier":
        nn.init.xavier_uniform_(layer.weight)

    else:
        raise ValueError(f"Unknown initialization: {init_name}")

    nn.init.zeros_(layer.bias)


class SimpleMLP(nn.Module):
    """
    Simple MLP for binary classification.

    Output:
    - one scalar logit per molecule;
    - sigmoid is not applied here, because BCEWithLogitsLoss expects logits.
    """

    def __init__(
        self,
        input_dim,
        hidden_layers,
        activation="relu",
        dropout=0.0,
        batchnorm=False,
        init="kaiming",
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        # Linear → BatchNorm → Activation → Dropout
        for hidden_dim in hidden_layers:
            linear = nn.Linear(current_dim, hidden_dim)
            initialize_linear_layer(linear, init, activation)

            layers.append(linear)

            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(get_activation(activation))

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            current_dim = hidden_dim  # e.g. 2048 → 512

        output_layer = nn.Linear(current_dim, 1)
        nn.init.xavier_uniform_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

        layers.append(output_layer)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.model(x)
        logits = logits.squeeze(1)
        return logits


def create_model(input_dim, hp):
    """
    Build an MLP from a hyperparameter dictionary.
    """

    hidden_layers = build_hidden_layers(
        n_layers=hp["n_layers"],
        n_nodes=hp["n_nodes"],
        r=hp["r"],
    )

    model = SimpleMLP(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        activation=hp["activation"],
        dropout=hp["dropout"],
        batchnorm=hp["batchnorm"],
        init=hp["init"],
    )

    return model, hidden_layers


# -------------------------------------------------------------------------
# Training and validation
# -------------------------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, grad_clip, device):
    """
    Train the model for one epoch and return average training loss.
    """

    model.train()

    total_loss = 0.0
    n_examples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)  # forward pass
        loss = criterion(logits, y_batch)

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip,
            )

        optimizer.step()

        batch_size = len(y_batch)
        total_loss += loss.item() * batch_size  # weighted mean on the epoch
        n_examples += batch_size

    average_loss = total_loss / max(1, n_examples)

    return average_loss


def predict_probabilities(model, X, device):
    """
    Predict positive-class probabilities from logits.
    """

    model.eval()  # always before making predictions

    with torch.no_grad():
        logits = model(X.to(device))
        probabilities = torch.sigmoid(logits)

    return probabilities.cpu().numpy()


def evaluate_model(model, X_val, y_val, device):
    """
    Evaluate the model on validation data using Hi metrics.
    """

    probabilities = predict_probabilities(
        model,
        X_val,
        device,
    )

    metrics = get_hi_metrics(
        y_val.numpy(),
        probabilities,
    )

    return metrics


def train_and_evaluate(
    X_train,
    y_train,
    X_val,
    y_val,
    hp,
    device,
    seed=42,
):
    """
    Train one MLP with early stopping on validation PR-AUC.

    This is used during inner CV and final retraining.
    """

    set_seed(seed)

    model, hidden_layers = create_model(
        input_dim=X_train.shape[1],  # e.g. ecfp4 --> 2048
        hp=hp,
    )

    model = model.to(device)

    pos_weight = compute_pos_weight(y_train).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp["lr"],
        weight_decay=hp["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # max beacuse pr-auc must increment
        factor=0.5,
        patience=5,  # reduce learning rate if PR-AUC plateaus for 5 epochs (not early stopping).
    )

    drop_last = should_drop_last_for_batchnorm(
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

    # early stopping variables
    best_score = -1.0
    best_state = None
    best_epoch = -1
    best_train_loss = None
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "val_pr_auc": [],
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

        val_metrics = evaluate_model(
            model=model,
            X_val=X_val,
            y_val=y_val,
            device=device,
        )

        val_pr_auc = val_metrics["pr_auc"]

        scheduler.step(val_pr_auc)

        history["train_loss"].append(train_loss)
        history["val_pr_auc"].append(val_pr_auc)

        if val_pr_auc > best_score:
            best_score = val_pr_auc
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

    search_space must contain discrete lists/arrays of values.

    """

    hp = {}

    for name, possible_values in search_space.items():
        value = rng.choice(possible_values)

        if hasattr(value, "item"):
            value = value.item()

        hp[name] = value

    return hp


def check_stratified_cv_is_possible(y_train, inner_k):
    """
    Each class must have at least inner_k examples.
    """

    y_np = y_train.numpy().astype(int)

    n_pos = int(y_np.sum())
    n_neg = int(len(y_np) - n_pos)

    # e.g. not valid --> n_pos = 1, n_neg = 50, inner_k = 2
    if n_pos < inner_k or n_neg < inner_k:
        raise ValueError(
            f"StratifiedKFold with inner_k={inner_k} is not possible: "
            f"n_pos={n_pos}, n_neg={n_neg}."
        )


def maybe_scale_inner_features(
    X_inner_train,
    X_inner_val,
    scale_features,
):
    """
    Scale features inside one inner-CV split if needed.

    For ECFP4/MACCS:
    - scale_features=False, tensors are returned unchanged.

    For rdkit_desc:
    - scale_features=True, scaler is fit on inner train only.
    """

    if not scale_features:
        return X_inner_train, X_inner_val, None

    return scale_features_from_train(
        X_inner_train,
        X_inner_val,
    )


def evaluate_hyperparameters_inner_cv(
    X_train,
    y_train,
    hp,
    fixed_hp,
    inner_k,
    device,
    seed,
    scale_features=False,
):
    """
    Evaluate one hyperparameter configuration with inner stratified CV.

    Returns:
    - mean validation PR-AUC;
    - mean train loss at the best validation epoch, kept only as diagnostic.

    Feature scaling is done inside each inner fold when scale_features=True.
    """

    check_stratified_cv_is_possible(y_train, inner_k)

    splitter = StratifiedKFold(
        n_splits=inner_k,
        shuffle=True,
        random_state=seed,
    )

    y_numpy = y_train.numpy().astype(int)

    scores = []
    train_losses_at_best = []

    for inner_train_idx, inner_val_idx in splitter.split(X_train, y_numpy):
        X_inner_train = X_train[inner_train_idx]
        y_inner_train = y_train[inner_train_idx]

        X_inner_val = X_train[inner_val_idx]
        y_inner_val = y_train[inner_val_idx]

        X_inner_train, X_inner_val, _ = maybe_scale_inner_features(
            X_inner_train=X_inner_train,
            X_inner_val=X_inner_val,
            scale_features=scale_features,
        )

        full_hp = {
            **hp,
            **fixed_hp,
        }

        training_result = train_and_evaluate(
            X_train=X_inner_train,
            y_train=y_inner_train,
            X_val=X_inner_val,
            y_val=y_inner_val,
            hp=full_hp,
            device=device,
            seed=seed,
        )

        scores.append(training_result["best_score"])
        train_losses_at_best.append(training_result["best_train_loss"])

    mean_score = float(np.mean(scores))
    mean_train_loss_at_best = float(np.mean(train_losses_at_best))

    return mean_score, mean_train_loss_at_best


def run_random_search_for_fold(
    X_train,
    y_train,
    fold_idx,
    cfg,
    search_space,
    fixed_hp,
    n_iter,
    device,
    seed,
    logger=None,
    scale_features=False,
):
    """
    Run random search inside one outer fold.
    """

    rng = np.random.default_rng(seed + fold_idx)

    search_results = []

    for iteration in range(n_iter):
        hp = sample_hyperparameters(
            search_space,
            rng,
        )

        start_time = time.time()

        mean_inner_score, mean_train_loss_at_best = evaluate_hyperparameters_inner_cv(
            X_train=X_train,
            y_train=y_train,
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
                f"inner PR-AUC={mean_inner_score:.4f} "
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

    # Diagnostic only. Final retraining now uses an internal stratified
    # early-stopping split instead of this threshold.
    best_train_loss_diagnostic = best_result["mean_train_loss_at_best"]

    return best_hp, best_score, best_train_loss_diagnostic, search_results


# -------------------------------------------------------------------------
# Final retraining and test evaluation
# -------------------------------------------------------------------------


def predict_with_trained_state(X_test, input_dim, hp, state_dict, device):
    """
    Rebuild the model, load saved weights, and predict test probabilities.
    """

    model, _ = create_model(
        input_dim=input_dim,
        hp=hp,
    )

    model = model.to(device)
    model.load_state_dict(state_dict)

    probabilities = predict_probabilities(
        model,
        X_test,
        device,
    )

    return probabilities


def make_stratified_holdout_split(y, val_fraction=0.15, seed=42):
    """
    Create a stratified train/validation split for final early stopping.
    """

    y_np = y.numpy().astype(int)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_fraction,
        random_state=seed,
    )

    train_idx, val_idx = next(splitter.split(np.zeros(len(y_np)), y_np))

    return train_idx, val_idx


def retrain_ensemble_and_evaluate_test(
    X_train,
    y_train,
    X_test,
    y_test,
    best_hp,
    best_train_loss_diagnostic,
    fold_idx,
    n_seeds,
    device,
    seed,
    logger=None,
    scale_features=False,
):
    """
    Retrain the best configuration with multiple seeds and evaluate the
    ensemble on the outer test fold.

    We use an internal stratified 85/15 split for early stopping on PR-AUC.
    This is more robust than stopping on a train-loss threshold, especially
    when BCEWithLogitsLoss uses pos_weight.
    """

    all_test_probabilities = []
    seed_results = []

    for seed_id in range(n_seeds):
        current_seed = seed + fold_idx * 100 + seed_id

        train_idx, val_idx = make_stratified_holdout_split(
            y=y_train,
            val_fraction=0.15,
            seed=current_seed,
        )

        X_retrain = X_train[train_idx]
        y_retrain = y_train[train_idx]

        # data for early stopping
        X_es = X_train[val_idx]
        y_es = y_train[val_idx]

        if scale_features:
            X_retrain, X_es, scaler = scale_features_from_train(
                X_retrain,
                X_es,
            )

            X_test_scaled_np = scaler.transform(X_test.numpy())
            X_test_final = torch.from_numpy(X_test_scaled_np).float()

        else:
            X_test_final = X_test

        training_result = train_and_evaluate(
            X_train=X_retrain,
            y_train=y_retrain,
            X_val=X_es,
            y_val=y_es,
            hp=best_hp,
            device=device,
            seed=current_seed,
        )

        if logger is not None:
            logger.info(
                f"  Seed {seed_id + 1}/{n_seeds}: "
                f"early-stop val PR-AUC={training_result['best_score']:.4f} "
                f"at epoch {training_result['best_epoch']}"
            )

        test_probabilities = predict_with_trained_state(
            X_test=X_test_final,
            input_dim=X_test_final.shape[1],
            hp=best_hp,
            state_dict=training_result["best_state"],
            device=device,
        )

        all_test_probabilities.append(test_probabilities)

        seed_results.append(
            {
                "seed": current_seed,
                "best_epoch": training_result["best_epoch"],
                "best_score": training_result["best_score"],
                "hidden_layers": training_result["hidden_layers"],
            }
        )

    ensemble_probabilities = np.mean(
        all_test_probabilities,
        axis=0,
    )

    test_metrics = get_hi_metrics(
        y_test.numpy(),
        ensemble_probabilities,
    )

    result = {
        "test_metrics": test_metrics,
        "test_probabilities": ensemble_probabilities,
        "seed_results": seed_results,
    }

    return result


# -------------------------------------------------------------------------
# Full nested random-search pipeline
# -------------------------------------------------------------------------


def run_nested_random_search(
    cfg,
    folds_tensors,
    search_space,
    fixed_hp,
    n_iter,
    n_seeds,
    device,
    seed=42,
    logger=None,
):
    """
    Run nested CV with random search and final ensemble evaluation.
    """

    fold_results = []

    for fold_idx in cfg["outer_folds"]:
        if logger is not None:
            logger.info(f"\n{'=' * 60}\n" f"OUTER FOLD {fold_idx}\n" f"{'=' * 60}")

        X_train = folds_tensors[fold_idx]["X_train"]
        y_train = folds_tensors[fold_idx]["y_train"]
        X_test = folds_tensors[fold_idx]["X_test"]
        y_test = folds_tensors[fold_idx]["y_test"]

        scale_features = folds_tensors[fold_idx].get(
            "scale_features",
            False,
        )

        best_hp, best_inner_score, best_train_loss_diagnostic, search_results = (
            run_random_search_for_fold(
                X_train=X_train,
                y_train=y_train,
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
        )

        if logger is not None:
            logger.info(
                f"[Fold {fold_idx}] "
                f"Best inner PR-AUC: {best_inner_score:.4f} | "
                f"inner train-loss diagnostic: {best_train_loss_diagnostic:.4f}"
            )

        final_result = retrain_ensemble_and_evaluate_test(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
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
                f"[Fold {fold_idx}] " f"Test metrics: {final_result['test_metrics']}"
            )

        fold_result = {
            "fold": fold_idx,
            "best_hp": best_hp,
            "inner_score": best_inner_score,
            "best_train_loss_diagnostic": best_train_loss_diagnostic,
            "test_metrics": final_result["test_metrics"],
            "test_probabilities": final_result["test_probabilities"],
            "seed_results": final_result["seed_results"],
            "search_results": search_results,
        }

        fold_results.append(fold_result)

    return fold_results


# -------------------------------------------------------------------------
# Grid search
# -------------------------------------------------------------------------


def run_grid_search_for_fold(
    X_train,
    y_train,
    fold_idx,
    cfg,
    grid,
    fixed_hp,
    device,
    seed,
    logger=None,
    scale_features=False,
):
    """
    Evaluate every configuration in a fixed grid with inner stratified CV.

    grid must be a list of dictionaries.
    """

    search_results = []

    for iteration, hp in enumerate(grid):
        start_time = time.time()

        mean_inner_score, mean_train_loss_at_best = evaluate_hyperparameters_inner_cv(
            X_train=X_train,
            y_train=y_train,
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
                f"  [{iteration + 1}/{len(grid)}] "
                f"inner PR-AUC={mean_inner_score:.4f} "
                f"({elapsed_time:.0f}s) | "
                f"lr={hp['lr']:.0e} "
                f"dropout={hp['dropout']} "
                f"wd={hp['weight_decay']:.0e}"
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
    best_train_loss_diagnostic = best_result["mean_train_loss_at_best"]

    return best_hp, best_score, best_train_loss_diagnostic, search_results


def run_nested_grid_search(
    cfg,
    folds_tensors,
    grid,
    fixed_hp,
    n_seeds,
    device,
    seed=42,
    logger=None,
):
    """
    Run nested CV with grid search and final ensemble evaluation.
    """

    fold_results = []

    for fold_idx in cfg["outer_folds"]:
        if logger is not None:
            logger.info(
                f"\n{'=' * 60}\n" f"GRID SEARCH — OUTER FOLD {fold_idx}\n" f"{'=' * 60}"
            )

        X_train = folds_tensors[fold_idx]["X_train"]
        y_train = folds_tensors[fold_idx]["y_train"]
        X_test = folds_tensors[fold_idx]["X_test"]
        y_test = folds_tensors[fold_idx]["y_test"]

        scale_features = folds_tensors[fold_idx].get(
            "scale_features",
            False,
        )

        best_hp, best_inner_score, best_train_loss_diagnostic, search_results = (
            run_grid_search_for_fold(
                X_train=X_train,
                y_train=y_train,
                fold_idx=fold_idx,
                cfg=cfg,
                grid=grid,
                fixed_hp=fixed_hp,
                device=device,
                seed=seed,
                logger=logger,
                scale_features=scale_features,
            )
        )

        if logger is not None:
            logger.info(
                f"[Fold {fold_idx}] "
                f"Best grid PR-AUC: {best_inner_score:.4f} | "
                f"inner train-loss diagnostic: {best_train_loss_diagnostic:.4f}"
            )

        final_result = retrain_ensemble_and_evaluate_test(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
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
                f"[Fold {fold_idx}] " f"Test metrics: {final_result['test_metrics']}"
            )

        fold_result = {
            "fold": fold_idx,
            "best_hp": best_hp,
            "inner_score": best_inner_score,
            "best_train_loss_diagnostic": best_train_loss_diagnostic,
            "test_metrics": final_result["test_metrics"],
            "test_probabilities": final_result["test_probabilities"],
            "seed_results": final_result["seed_results"],
            "search_results": search_results,
        }

        fold_results.append(fold_result)

    return fold_results


# -------------------------------------------------------------------------
# Reporting
# -------------------------------------------------------------------------


def print_final_results(fold_results, title="MLP RESULTS"):
    """
    Print per-fold and aggregated test results.
    """

    test_metrics_per_fold = [result["test_metrics"] for result in fold_results]

    aggregated_metrics = aggregate_fold_metrics(
        test_metrics_per_fold,
    )

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for result in fold_results:
        fold = result["fold"]
        metrics = result["test_metrics"]

        pr_auc = metrics.get("pr_auc", float("nan"))
        roc_auc = metrics.get("roc_auc", float("nan"))

        print(f"Fold {fold}: " f"PR-AUC={pr_auc:.4f}  " f"ROC-AUC={roc_auc:.4f}")

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
