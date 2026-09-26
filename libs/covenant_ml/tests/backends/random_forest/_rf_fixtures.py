"""Shared helpers for the Random Forest backend integration test modules."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from covenant_ml.backends.protocol import ClassifierBackend
from covenant_ml.types import (
    ClassifierTrainConfig,
    RandomForestConfig,
    TrainOutcome,
)


def _invoke_rf_train(
    backend: ClassifierBackend,
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    names: list[str] | None,
    config: ClassifierTrainConfig,
    output_dir: Path,
) -> TrainOutcome:
    """Helper to invoke backend train (isolates .train() call for guard).

    Args:
        backend: Classifier backend to use.
        x: Feature matrix.
        y: Labels.
        names: Feature names.
        config: Training configuration.
        output_dir: Output directory for model artifacts.

    Returns:
        TrainOutcome from the training run.
    """
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
    """Create synthetic binary classification dataset for edge case tests.

    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        pos_ratio: Ratio of positive samples.
        seed: Random seed.

    Returns:
        Tuple of (features, labels, feature_names).
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    n_pos = int(n_samples * pos_ratio)
    y = np.zeros(n_samples, dtype=np.int64)
    y[:n_pos] = 1
    rng.shuffle(y)
    feature_names = [f"f{i}" for i in range(n_features)]
    return x, y, feature_names


def _make_rf_config(
    n_estimators: int = 10,
    max_depth: int | None = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: Literal["sqrt", "log2"] | float | None = "sqrt",
    bootstrap: bool = True,
    oob_score: bool = False,
) -> RandomForestConfig:
    """Create RandomForest config for testing.

    Args:
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        min_samples_split: Minimum samples to split a node.
        min_samples_leaf: Minimum samples in a leaf.
        max_features: Number of features to consider for best split.
        bootstrap: Whether to use bootstrap samples.
        oob_score: Whether to compute out-of-bag score.

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
        "class_weight_balanced": True,
        "n_jobs": 1,
        "oob_score": oob_score,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 42,
    }
