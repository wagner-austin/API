"""Shared fixtures and helpers for test_cleargbm_integration splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from covenant_ml.backends.protocol import ClassifierBackend
from covenant_ml.types import (
    ClassifierTrainConfig,
    ClearGBMConfig,
    FeatureImportance,
    TrainOutcome,
)

_BASELINE_LOSS = 0.693


def _invoke_cleargbm_train(
    backend: ClassifierBackend,
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    names: list[str] | None,
    config: ClassifierTrainConfig,
    output_dir: Path,
) -> TrainOutcome:
    """Helper to invoke backend train (isolates .train() call for guard)."""
    return backend.train(
        x_features=x,
        y_labels=y,
        feature_names=names,
        config=config,
        output_dir=output_dir,
        progress=None,
    )


def _make_synthetic_dataset(
    n_samples: int = 100,
    n_features: int = 8,
    pos_ratio: float = 0.3,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
    """Create synthetic binary classification dataset for edge case tests."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    n_pos = int(n_samples * pos_ratio)
    y = np.zeros(n_samples, dtype=np.int64)
    y[:n_pos] = 1
    rng.shuffle(y)
    feature_names = [f"f{i}" for i in range(n_features)]
    return x, y, feature_names


def _make_cleargbm_config(
    n_estimators: int = 5,
    max_depth: int = 3,
) -> ClearGBMConfig:
    """Create ClearGBM config for testing."""
    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": 0.1,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": None,
        "colsample_bytree": None,
        "max_bins": 64,
        "subsample": 1.0,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 42,
        "early_stopping_rounds": 3,
        "monotonic_constraints": None,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "n_jobs": 1,
        "growth_strategy": "depth_wise",
        "num_leaves": None,
    }


def _require_importances(
    importances: list[FeatureImportance] | None,
) -> list[FeatureImportance]:
    """Require importances list is not None.

    Args:
        importances: Feature importances or None.

    Returns:
        Feature importances list.

    Raises:
        AssertionError: If importances is None.
    """
    if importances is None:
        raise AssertionError("Expected importances list, got None")
    return importances
