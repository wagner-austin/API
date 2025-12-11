"""Testing utilities for covenant_ml."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from .trainer import set_cuda_available_hook
from .types import TrainConfig


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


__all__ = ["make_train_config", "set_cuda_hook"]
