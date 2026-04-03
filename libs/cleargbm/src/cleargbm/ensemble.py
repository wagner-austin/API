"""Gradient boosting ensemble training and prediction.

Uses numpy arrays for efficient data representation.
Supports early stopping when validation loss stops improving.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from cleargbm._hooks_ensemble import predict_proba_from_raw as _predict_proba_hook
from cleargbm._hooks_ensemble import predict_raw_ensemble as _predict_raw_hook
from cleargbm._hooks_infra import WorkerPoolProtocol, create_worker_pool
from cleargbm._hooks_native import (
    NativeModel,
)
from cleargbm._hooks_native import (
    predict_proba_native as _predict_proba_native_hook,
)
from cleargbm._hooks_native import (
    predict_raw_native as _predict_raw_native_hook,
)
from cleargbm._hooks_native import (
    train_native as _train_native_hook,
)
from cleargbm.histogram import precompute_feature_bins
from cleargbm.losses import BinaryLogLoss, raw_to_proba
from cleargbm.tree import build_tree, predict_tree
from cleargbm.types import (
    DecisionTree,
    GradientBoostingConfig,
    GradientBoostingModel,
    TrainingProgress,
)


class _EarlyStoppingState(TypedDict):
    """Internal state for early stopping tracking.

    Args:
        best_val_loss: Best validation loss seen so far.
        best_round: Zero-indexed round where best loss was achieved.
        rounds_without_improvement: Consecutive rounds without improvement.
        should_stop: Whether to stop training.
    """

    best_val_loss: float
    best_round: int
    rounds_without_improvement: int
    should_stop: bool


def _init_early_stopping_state() -> _EarlyStoppingState:
    """Initialize early stopping state.

    Returns:
        Initial state with best_val_loss set to infinity.
    """
    return _EarlyStoppingState(
        best_val_loss=float("inf"),
        best_round=0,
        rounds_without_improvement=0,
        should_stop=False,
    )


def _update_early_stopping_state(
    state: _EarlyStoppingState,
    val_loss: float,
    tree_idx: int,
    patience: int,
) -> _EarlyStoppingState:
    """Update early stopping state with new validation loss.

    Args:
        state: Current early stopping state.
        val_loss: Validation loss for current round.
        tree_idx: Zero-indexed tree/round index.
        patience: Number of rounds without improvement before stopping.

    Returns:
        Updated early stopping state (new instance).
    """
    if val_loss < state["best_val_loss"]:
        # Improvement: reset counter
        return _EarlyStoppingState(
            best_val_loss=val_loss,
            best_round=tree_idx,
            rounds_without_improvement=0,
            should_stop=False,
        )

    # No improvement
    new_rounds_without_improvement = state["rounds_without_improvement"] + 1
    should_stop = new_rounds_without_improvement >= patience

    return _EarlyStoppingState(
        best_val_loss=state["best_val_loss"],
        best_round=state["best_round"],
        rounds_without_improvement=new_rounds_without_improvement,
        should_stop=should_stop,
    )


class _ValidationState(TypedDict):
    """State for validation tracking during training.

    Args:
        x_val: Validation feature matrix.
        y_val: Validation labels.
        raw_preds: Current raw predictions on validation set.
        early_stopping_rounds: Patience for early stopping (None = disabled).
        es_state: Early stopping state tracker.
    """

    x_val: NDArray[np.float64]
    y_val: NDArray[np.int64]
    raw_preds: NDArray[np.float64]
    early_stopping_rounds: int
    es_state: _EarlyStoppingState


def _update_validation(
    val_state: _ValidationState,
    tree: DecisionTree,
    tree_idx: int,
    learning_rate: float,
) -> tuple[float, _ValidationState]:
    """Update validation predictions and early stopping state.

    Args:
        val_state: Current validation state.
        tree: Newly built tree.
        tree_idx: Zero-indexed tree index.
        learning_rate: Learning rate for prediction updates.

    Returns:
        Tuple of (validation loss, updated validation state).
    """
    tree_preds = predict_tree(tree, val_state["x_val"])
    new_raw_preds = _add_tree_predictions(val_state["raw_preds"], tree_preds, learning_rate)
    val_loss = _compute_loss(val_state["y_val"], new_raw_preds)

    new_es_state = _update_early_stopping_state(
        val_state["es_state"],
        val_loss,
        tree_idx,
        val_state["early_stopping_rounds"],
    )

    return val_loss, _ValidationState(
        x_val=val_state["x_val"],
        y_val=val_state["y_val"],
        raw_preds=new_raw_preds,
        early_stopping_rounds=val_state["early_stopping_rounds"],
        es_state=new_es_state,
    )


def _validate_training_inputs(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    feature_names: tuple[str, ...],
) -> None:
    """Validate training inputs.

    Args:
        x_train: Training feature matrix.
        y_train: Training labels.
        feature_names: Feature names.

    Raises:
        ValueError: If inputs are invalid.
    """
    n_train: int = x_train.shape[0]
    if n_train == 0:
        raise ValueError("x_train must not be empty")
    n_y: int = y_train.shape[0]
    if n_train != n_y:
        raise ValueError(f"x_train and y_train must have same length, got {n_train} and {n_y}")
    n_features: int = x_train.shape[1]
    if n_features != len(feature_names):
        raise ValueError(
            f"x_train has {n_features} features but {len(feature_names)} feature names provided"
        )


class _SimpleValState(TypedDict):
    """State for simple validation tracking (no early stopping).

    Args:
        x_val: Validation feature matrix.
        y_val: Validation labels.
        raw_preds: Current raw predictions.
    """

    x_val: NDArray[np.float64]
    y_val: NDArray[np.int64]
    raw_preds: NDArray[np.float64]


def _update_simple_validation(
    state: _SimpleValState,
    tree: DecisionTree,
    learning_rate: float,
) -> tuple[float, _SimpleValState]:
    """Update simple validation predictions (no early stopping).

    Args:
        state: Current validation state.
        tree: Newly built tree.
        learning_rate: Learning rate.

    Returns:
        Tuple of (validation loss, updated state).
    """
    tree_preds = predict_tree(tree, state["x_val"])
    new_raw_preds = _add_tree_predictions(state["raw_preds"], tree_preds, learning_rate)
    val_loss = _compute_loss(state["y_val"], new_raw_preds)
    return val_loss, _SimpleValState(
        x_val=state["x_val"],
        y_val=state["y_val"],
        raw_preds=new_raw_preds,
    )


def _compute_loss(
    y_true: NDArray[np.int64],
    raw_preds: NDArray[np.float64],
) -> float:
    """Compute binary log loss from raw predictions.

    Args:
        y_true: True labels (0 or 1).
        raw_preds: Raw predictions (log-odds).

    Returns:
        Mean loss value.
    """
    loss_fn = BinaryLogLoss()
    probas = raw_to_proba(raw_preds)
    return loss_fn.loss(y_true, probas)


def _add_tree_predictions(
    raw_preds: NDArray[np.float64],
    tree_preds: NDArray[np.float64],
    learning_rate: float,
) -> NDArray[np.float64]:
    """Add scaled tree predictions to raw predictions.

    Args:
        raw_preds: Current raw predictions.
        tree_preds: Tree predictions to add.
        learning_rate: Learning rate multiplier.

    Returns:
        Updated raw predictions.
    """
    result: NDArray[np.float64] = raw_preds + learning_rate * tree_preds
    return result


def _create_worker_pool(
    n_jobs: int,
    bin_edges_raw: tuple[tuple[float, ...], ...],
    sample_bins: NDArray[np.int64],
) -> WorkerPoolProtocol | None:
    """Create an initialized worker pool or return None.

    Args:
        n_jobs: Number of parallel workers (-1 = all cores, 1 = sequential).
        bin_edges_raw: Raw bin edges for each feature (pickle-safe tuples).
        sample_bins: Per-sample bin assignments (n_samples, n_features).

    Returns:
        WorkerPoolProtocol or None if sequential.
    """
    actual_workers = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
    if actual_workers <= 1:
        return None
    return create_worker_pool(actual_workers, bin_edges_raw, sample_bins)


def train_gradient_boosting(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    x_val: NDArray[np.float64] | None,
    y_val: NDArray[np.int64] | None,
    config: GradientBoostingConfig,
    feature_names: tuple[str, ...],
    progress_callback: Callable[[TrainingProgress], None] | None = None,
) -> GradientBoostingModel:
    """Train gradient boosting classifier.

    Supports early stopping when validation loss stops improving. When
    early_stopping_rounds is set in config and validation data is provided,
    training stops after the specified number of rounds without improvement.
    The returned model contains only trees up to and including the best round.

    Args:
        x_train: Training feature matrix (n_samples, n_features).
        y_train: Training labels (0 or 1).
        x_val: Validation feature matrix, or None.
        y_val: Validation labels, or None.
        config: Training configuration.
        feature_names: Names for each feature.
        progress_callback: Optional callback for progress updates.

    Returns:
        Trained gradient boosting model. If early stopping is enabled and
        triggered, the model contains only trees up to the best round.

    Raises:
        ValueError: If x_train is empty or dimensions don't match.
    """
    _validate_training_inputs(x_train, y_train, feature_names)
    n_train: int = x_train.shape[0]

    # Initialize loss function
    loss_fn = BinaryLogLoss()

    # Compute initial prediction (log-odds of positive class rate)
    base_prediction = loss_fn.initial_prediction(y_train)

    # Initialize raw predictions for all training samples
    raw_preds_train: NDArray[np.float64] = np.full(n_train, base_prediction, dtype=np.float64)

    # Initialize validation state based on config
    early_stopping_rounds = config["early_stopping_rounds"]
    val_state: _ValidationState | None = None
    simple_val_state: _SimpleValState | None = None

    if x_val is not None and y_val is not None:
        n_val: int = x_val.shape[0]
        initial_val_preds: NDArray[np.float64] = np.full(n_val, base_prediction, dtype=np.float64)
        if early_stopping_rounds is not None:
            val_state = _ValidationState(
                x_val=x_val,
                y_val=y_val,
                raw_preds=initial_val_preds,
                early_stopping_rounds=early_stopping_rounds,
                es_state=_init_early_stopping_state(),
            )
        else:
            simple_val_state = _SimpleValState(
                x_val=x_val,
                y_val=y_val,
                raw_preds=initial_val_preds,
            )

    # Build trees
    trees: list[DecisionTree] = []
    n_estimators = config["n_estimators"]
    learning_rate = config["learning_rate"]

    # Precompute feature bins once for all trees
    feature_bins = precompute_feature_bins(x_train, config["max_bins"])

    # Create worker pool once for all trees (if parallel is enabled)
    n_jobs = config["n_jobs"]
    bin_edges_raw = tuple(be.edges for be in feature_bins.bin_edges)
    pool: WorkerPoolProtocol | None = _create_worker_pool(
        n_jobs, bin_edges_raw, feature_bins.sample_bins
    )

    for tree_idx in range(n_estimators):
        # Compute current probabilities
        probas_train = raw_to_proba(raw_preds_train)

        # Compute gradients and hessians
        gradients = loss_fn.gradients(y_train, probas_train)
        hessians = loss_fn.hessians(y_train, probas_train)

        tree = build_tree(
            x=x_train,
            gradients=gradients,
            hessians=hessians,
            config=config,
            feature_names=feature_names,
            feature_bins=feature_bins,
            pool=pool,
        )
        trees.append(tree)

        # Update training predictions
        tree_preds_train = predict_tree(tree, x_train)
        raw_preds_train = _add_tree_predictions(raw_preds_train, tree_preds_train, learning_rate)

        # Update validation predictions and early stopping state
        val_loss: float | None = None
        if val_state is not None:
            val_loss, val_state = _update_validation(val_state, tree, tree_idx, learning_rate)
        elif simple_val_state is not None:
            val_loss, simple_val_state = _update_simple_validation(
                simple_val_state, tree, learning_rate
            )

        # Report progress
        train_loss = _compute_loss(y_train, raw_preds_train)
        if progress_callback is not None:
            progress = TrainingProgress(
                tree_index=tree_idx,
                total_trees=n_estimators,
                train_loss=train_loss,
                val_loss=val_loss,
            )
            progress_callback(progress)

        # Check early stopping
        if val_state is not None and val_state["es_state"]["should_stop"]:
            break

    # Clean up worker pool
    if pool is not None:
        pool.close()
        pool.join()

    # If early stopping was enabled, return only trees up to best round
    final_trees: tuple[DecisionTree, ...]
    if val_state is not None:
        best_round = val_state["es_state"]["best_round"]
        final_trees = tuple(trees[: best_round + 1])
    else:
        final_trees = tuple(trees)

    return GradientBoostingModel(
        trees=final_trees,
        base_prediction=base_prediction,
        learning_rate=learning_rate,
        feature_names=feature_names,
        n_classes=2,
        config=config,
    )


def predict_raw(
    model: GradientBoostingModel,
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Predict raw scores (log-odds) for samples.

    Delegates to the active backend hook after validation.

    Args:
        model: Trained model.
        x: Feature matrix (n_samples, n_features).

    Returns:
        Raw predictions (log-odds) for each sample.

    Raises:
        ValueError: If x is empty or has wrong number of features.
    """
    n_samples: int = x.shape[0]
    if n_samples == 0:
        raise ValueError("x must not be empty")
    n_features: int = x.shape[1]
    if n_features != len(model["feature_names"]):
        raise ValueError(
            f"x has {n_features} features but model expects {len(model['feature_names'])}"
        )

    return _predict_raw_hook(
        model["trees"],
        x,
        model["base_prediction"],
        model["learning_rate"],
    )


def predict_proba(
    model: GradientBoostingModel,
    x: NDArray[np.float64],
) -> tuple[tuple[float, float], ...]:
    """Predict class probabilities.

    Delegates to the active backend hooks after validation.

    Args:
        model: Trained model.
        x: Feature matrix (n_samples, n_features).

    Returns:
        Probabilities array (n_samples, 2) for binary classification.
        Each row is (prob_class_0, prob_class_1).

    Raises:
        ValueError: If x is empty or has wrong number of features.
    """
    raw_preds = predict_raw(model, x)
    return _predict_proba_hook(raw_preds)


def train_gradient_boosting_native(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    x_val: NDArray[np.float64] | None,
    y_val: NDArray[np.int64] | None,
    config: GradientBoostingConfig,
    feature_names: tuple[str, ...],
) -> NativeModel:
    """Train gradient boosting model using the Rust full training loop.

    Runs the entire training loop in a single native call. All binning,
    histogram building, split finding, and tree construction happen in
    Rust with no per-iteration FFI overhead.

    Requires ``use_rust_backend()`` to have been called first.

    Args:
        x_train: Training feature matrix (n_samples, n_features).
        y_train: Training labels (0 or 1).
        x_val: Optional validation feature matrix.
        y_val: Optional validation labels.
        config: Training configuration.
        feature_names: Names for each feature.

    Returns:
        Opaque native model handle for use with ``predict_raw_native``
        and ``predict_proba_native``.

    Raises:
        RuntimeError: If Rust backend is not active.
        ValueError: If inputs are invalid.
    """
    _validate_training_inputs(x_train, y_train, feature_names)
    return _train_native_hook(x_train, y_train, x_val, y_val, config, feature_names)


def predict_raw_native(
    model: NativeModel,
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Predict raw scores (log-odds) using a native model.

    Delegates to the Rust model-level prediction in a single native call.

    Args:
        model: Native model handle from ``train_gradient_boosting_native``.
        x: Feature matrix (n_samples, n_features).

    Returns:
        Raw predictions (log-odds) for each sample.

    Raises:
        RuntimeError: If Rust backend is not active.
        ValueError: If x is empty.
    """
    n_samples: int = x.shape[0]
    if n_samples == 0:
        raise ValueError("x must not be empty")
    return _predict_raw_native_hook(model, x)


def predict_proba_native(
    model: NativeModel,
    x: NDArray[np.float64],
) -> tuple[tuple[float, float], ...]:
    """Predict class probabilities using a native model.

    Delegates to the Rust model-level prediction in a single native call.

    Args:
        model: Native model handle from ``train_gradient_boosting_native``.
        x: Feature matrix (n_samples, n_features).

    Returns:
        Tuple of (prob_class_0, prob_class_1) per sample.

    Raises:
        RuntimeError: If Rust backend is not active.
        ValueError: If x is empty.
    """
    n_samples: int = x.shape[0]
    if n_samples == 0:
        raise ValueError("x must not be empty")
    return _predict_proba_native_hook(model, x)


__all__ = [
    "predict_proba",
    "predict_proba_native",
    "predict_raw",
    "predict_raw_native",
    "train_gradient_boosting",
    "train_gradient_boosting_native",
]
