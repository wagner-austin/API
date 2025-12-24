"""Gradient boosting ensemble training and prediction.

Built from scratch - uses only Python stdlib (no numpy).
"""

from __future__ import annotations

import os
from collections.abc import Callable

from cleargbm._test_hooks import WorkerPoolProtocol, create_worker_pool
from cleargbm.histogram import precompute_feature_bins
from cleargbm.losses import BinaryLogLoss, raw_to_proba, sigmoid
from cleargbm.tree import build_tree, predict_tree
from cleargbm.types import (
    DecisionTree,
    FloatArray,
    FloatMatrix,
    GradientBoostingConfig,
    GradientBoostingModel,
    TrainingProgress,
)


def _compute_loss(
    y_true: tuple[int, ...],
    raw_preds: FloatArray,
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
    raw_preds: FloatArray,
    tree_preds: FloatArray,
    learning_rate: float,
) -> FloatArray:
    """Add scaled tree predictions to raw predictions.

    Args:
        raw_preds: Current raw predictions.
        tree_preds: Tree predictions to add.
        learning_rate: Learning rate multiplier.

    Returns:
        Updated raw predictions.
    """
    return tuple(rp + learning_rate * tp for rp, tp in zip(raw_preds, tree_preds, strict=True))


def _create_worker_pool(
    n_jobs: int,
    bin_edges_raw: tuple[tuple[float, ...], ...],
    sample_bins: tuple[tuple[int, ...], ...],
) -> WorkerPoolProtocol | None:
    """Create an initialized worker pool or return None.

    Args:
        n_jobs: Number of parallel workers (-1 = all cores, 1 = sequential).
        bin_edges_raw: Raw bin edges for each feature (pickle-safe tuples).
        sample_bins: Per-sample bin assignments for each feature.

    Returns:
        WorkerPoolProtocol or None if sequential.
    """
    actual_workers = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
    if actual_workers <= 1:
        return None
    return create_worker_pool(actual_workers, bin_edges_raw, sample_bins)


def train_gradient_boosting(
    x_train: FloatMatrix,
    y_train: tuple[int, ...],
    x_val: FloatMatrix | None,
    y_val: tuple[int, ...] | None,
    config: GradientBoostingConfig,
    feature_names: tuple[str, ...],
    progress_callback: Callable[[TrainingProgress], None] | None = None,
) -> GradientBoostingModel:
    """Train gradient boosting classifier.

    Args:
        x_train: Training feature matrix (n_samples, n_features).
        y_train: Training labels (0 or 1).
        x_val: Validation feature matrix, or None.
        y_val: Validation labels, or None.
        config: Training configuration.
        feature_names: Names for each feature.
        progress_callback: Optional callback for progress updates.

    Returns:
        Trained gradient boosting model.

    Raises:
        ValueError: If x_train is empty or dimensions don't match.
    """
    if len(x_train) == 0:
        raise ValueError("x_train must not be empty")
    if len(x_train) != len(y_train):
        raise ValueError(
            f"x_train and y_train must have same length, got {len(x_train)} and {len(y_train)}"
        )
    if len(x_train[0]) != len(feature_names):
        raise ValueError(
            f"x_train has {len(x_train[0])} features but "
            f"{len(feature_names)} feature names provided"
        )

    # Initialize loss function
    loss_fn = BinaryLogLoss()

    # Compute initial prediction (log-odds of positive class rate)
    base_prediction = loss_fn.initial_prediction(y_train)

    # Initialize raw predictions for all training samples
    n_train = len(x_train)
    raw_preds_train: FloatArray = tuple(base_prediction for _ in range(n_train))

    # Initialize validation predictions if validation set provided
    raw_preds_val: FloatArray | None = None
    if x_val is not None and y_val is not None:
        n_val = len(x_val)
        raw_preds_val = tuple(base_prediction for _ in range(n_val))

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

        # Update validation predictions if available
        val_loss: float | None = None
        if x_val is not None and y_val is not None and raw_preds_val is not None:
            tree_preds_val = predict_tree(tree, x_val)
            raw_preds_val = _add_tree_predictions(raw_preds_val, tree_preds_val, learning_rate)
            val_loss = _compute_loss(y_val, raw_preds_val)

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

    # Clean up worker pool
    if pool is not None:
        pool.close()
        pool.join()

    return GradientBoostingModel(
        trees=tuple(trees),
        base_prediction=base_prediction,
        learning_rate=learning_rate,
        feature_names=feature_names,
        n_classes=2,
        config=config,
    )


def predict_raw(
    model: GradientBoostingModel,
    x: FloatMatrix,
) -> FloatArray:
    """Predict raw scores (log-odds) for samples.

    Args:
        model: Trained model.
        x: Feature matrix (n_samples, n_features).

    Returns:
        Raw predictions (log-odds) for each sample.

    Raises:
        ValueError: If x is empty or has wrong number of features.
    """
    if len(x) == 0:
        raise ValueError("x must not be empty")
    if len(x[0]) != len(model["feature_names"]):
        raise ValueError(
            f"x has {len(x[0])} features but model expects {len(model['feature_names'])}"
        )

    n_samples = len(x)
    base_prediction = model["base_prediction"]
    learning_rate = model["learning_rate"]

    # Start with base prediction
    raw_preds: FloatArray = tuple(base_prediction for _ in range(n_samples))

    # Add contributions from each tree
    for tree in model["trees"]:
        tree_preds = predict_tree(tree, x)
        raw_preds = _add_tree_predictions(raw_preds, tree_preds, learning_rate)

    return raw_preds


def predict_proba(
    model: GradientBoostingModel,
    x: FloatMatrix,
) -> tuple[tuple[float, float], ...]:
    """Predict class probabilities.

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

    result: list[tuple[float, float]] = []
    for raw in raw_preds:
        prob_1 = sigmoid(raw)
        prob_0 = 1.0 - prob_1
        result.append((prob_0, prob_1))

    return tuple(result)


__all__ = [
    "predict_proba",
    "predict_raw",
    "train_gradient_boosting",
]
