"""Testing utilities for covenant_ml."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from .trainer import set_cuda_available_hook
from .types import LogRegConfig, RandomForestConfig, TrainConfig


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


__all__ = [
    "make_logreg_config",
    "make_random_forest_config",
    "make_train_config",
    "set_cuda_hook",
]
