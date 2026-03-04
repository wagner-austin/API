"""Testing utilities for covenant_ml."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from .trainer import set_cuda_available_hook
from .types import (
    LightGBMConfig,
    LogRegConfig,
    LSTMConfig,
    LSTMPrecision,
    MLPConfig,
    MLPOptimizer,
    MLPPrecision,
    RandomForestConfig,
    RequestedDevice,
    TrainConfig,
)


def set_cuda_hook(hook: Callable[[], bool] | None) -> None:
    """Override CUDA availability detection for tests."""
    set_cuda_available_hook(hook)


def make_train_config(
    *,
    device: Literal["cpu", "cuda", "auto"] = "cpu",
    learning_rate: float = 0.1,
    max_depth: int = 3,
    n_estimators: int = 10,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    early_stopping_rounds: int = 10,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    scale_pos_weight: float | None = None,
) -> TrainConfig:
    """Build a complete TrainConfig for tests with strict typing."""
    config: TrainConfig = {
        "device": device,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "random_state": random_state,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "early_stopping_rounds": early_stopping_rounds,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
    }
    if scale_pos_weight is not None:
        config["scale_pos_weight"] = scale_pos_weight
    return config


def make_logreg_config(
    *,
    solver: Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"] = "lbfgs",
    penalty: Literal["l1", "l2", "elasticnet", "none"] = "l2",
    inverse_reg_strength: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-4,
    class_weight_balanced: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    l1_ratio: float = 0.5,
) -> LogRegConfig:
    """Build a complete LogRegConfig for tests with strict typing.

    Args:
        solver: Optimization algorithm.
        penalty: Regularization type.
        inverse_reg_strength: Inverse of regularization strength (sklearn's C parameter).
        max_iter: Maximum iterations.
        tol: Tolerance for stopping.
        class_weight_balanced: Whether to balance class weights.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        random_state: Random seed.
        l1_ratio: ElasticNet mixing parameter.

    Returns:
        LogRegConfig for testing.
    """
    return {
        "solver": solver,
        "penalty": penalty,
        "C": inverse_reg_strength,
        "max_iter": max_iter,
        "tol": tol,
        "class_weight_balanced": class_weight_balanced,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_state": random_state,
        "l1_ratio": l1_ratio,
    }


def make_random_forest_config(
    *,
    n_estimators: int = 10,
    max_depth: int | None = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: Literal["sqrt", "log2"] | float | int | None = "sqrt",
    bootstrap: bool = True,
    class_weight_balanced: bool = True,
    n_jobs: int = -1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    oob_score: bool = False,
) -> RandomForestConfig:
    """Build a complete RandomForestConfig for tests with strict typing.

    Args:
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        min_samples_split: Minimum samples to split.
        min_samples_leaf: Minimum samples in leaf.
        max_features: Features per split.
        bootstrap: Whether to use bootstrap.
        class_weight_balanced: Whether to balance class weights.
        n_jobs: Parallel workers.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        random_state: Random seed.
        oob_score: Whether to compute OOB score.

    Returns:
        RandomForestConfig for testing.
    """
    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "bootstrap": bootstrap,
        "class_weight_balanced": class_weight_balanced,
        "n_jobs": n_jobs,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_state": random_state,
        "oob_score": oob_score,
    }


def make_xgboost_regressor_config(
    *,
    device: RequestedDevice = "cpu",
    learning_rate: float = 0.1,
    max_depth: int = 3,
    n_estimators: int = 10,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    early_stopping_rounds: int = 10,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
) -> TrainConfig:
    """Build a TrainConfig for XGBoost regressor tests.

    Same as make_train_config but without scale_pos_weight (regression
    has no class imbalance concept).

    Args:
        device: Device to use for training.
        learning_rate: Learning rate.
        max_depth: Maximum tree depth.
        n_estimators: Number of boosting rounds.
        subsample: Subsample ratio.
        colsample_bytree: Feature subsample ratio.
        random_state: Random seed.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        early_stopping_rounds: Early stopping patience.
        reg_alpha: L1 regularization.
        reg_lambda: L2 regularization.

    Returns:
        TrainConfig for XGBoost regressor testing.
    """
    return {
        "device": device,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "random_state": random_state,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "early_stopping_rounds": early_stopping_rounds,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
    }


def make_lightgbm_regressor_config(
    *,
    device: RequestedDevice = "cpu",
    learning_rate: float = 0.1,
    max_depth: int = -1,
    n_estimators: int = 10,
    num_leaves: int = 31,
    min_child_samples: int = 20,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    early_stopping_rounds: int = 10,
) -> LightGBMConfig:
    """Build a LightGBMConfig for regressor tests.

    Args:
        device: Device to use for training.
        learning_rate: Learning rate.
        max_depth: Maximum tree depth (-1 for no limit).
        n_estimators: Number of boosting rounds.
        num_leaves: Maximum number of leaves per tree.
        min_child_samples: Minimum samples in a leaf.
        subsample: Subsample ratio.
        colsample_bytree: Feature subsample ratio.
        reg_alpha: L1 regularization.
        reg_lambda: L2 regularization.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        random_state: Random seed.
        early_stopping_rounds: Early stopping patience.

    Returns:
        LightGBMConfig for regressor testing.
    """
    return {
        "device": device,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_state": random_state,
        "early_stopping_rounds": early_stopping_rounds,
    }


def make_mlp_regressor_config(
    *,
    device: RequestedDevice = "cpu",
    precision: MLPPrecision = "fp32",
    optimizer: MLPOptimizer = "adamw",
    hidden_sizes: tuple[int, ...] = (32, 16),
    learning_rate: float = 0.001,
    batch_size: int = 256,
    n_epochs: int = 10,
    dropout: float = 0.1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    early_stopping_patience: int = 5,
) -> MLPConfig:
    """Build an MLPConfig for regressor tests.

    Args:
        device: Device to use for training.
        precision: Precision mode.
        optimizer: Optimizer name.
        hidden_sizes: Tuple of hidden layer widths.
        learning_rate: Learning rate.
        batch_size: Mini-batch size.
        n_epochs: Maximum training epochs.
        dropout: Dropout probability.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        random_state: Random seed.
        early_stopping_patience: Early stopping patience.

    Returns:
        MLPConfig for regressor testing.
    """
    return {
        "device": device,
        "precision": precision,
        "optimizer": optimizer,
        "hidden_sizes": hidden_sizes,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "dropout": dropout,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_state": random_state,
        "early_stopping_patience": early_stopping_patience,
    }


def make_lstm_regressor_config(
    *,
    device: RequestedDevice = "cpu",
    precision: LSTMPrecision = "fp32",
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    bidirectional: bool = False,
    sequence_length: int = 4,
    learning_rate: float = 0.001,
    batch_size: int = 256,
    n_epochs: int = 10,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    early_stopping_patience: int = 5,
) -> LSTMConfig:
    """Build an LSTMConfig for regressor tests.

    Args:
        device: Device to use for training.
        precision: Precision mode.
        hidden_size: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout between LSTM layers.
        bidirectional: Whether to use bidirectional LSTM.
        sequence_length: Number of timesteps per sequence.
        learning_rate: Learning rate.
        batch_size: Mini-batch size.
        n_epochs: Maximum training epochs.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        random_state: Random seed.
        early_stopping_patience: Early stopping patience.

    Returns:
        LSTMConfig for regressor testing.
    """
    return {
        "device": device,
        "precision": precision,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "sequence_length": sequence_length,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_state": random_state,
        "early_stopping_patience": early_stopping_patience,
    }


__all__ = [
    "make_lightgbm_regressor_config",
    "make_logreg_config",
    "make_lstm_regressor_config",
    "make_mlp_regressor_config",
    "make_random_forest_config",
    "make_train_config",
    "make_xgboost_regressor_config",
    "set_cuda_hook",
]
