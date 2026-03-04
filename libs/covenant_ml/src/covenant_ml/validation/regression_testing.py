"""Testing utilities for regression cross-validation.

Provides factory functions and test data generators for regression CV tests.
This module is exported for consumers to use in their test suites.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .types import CVSplit, CVSplitInfo


def make_regression_targets(
    n_samples: int = 100,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Create deterministic regression targets for testing.

    Generates continuous values using a deterministic function of the
    sample index, suitable for testing cross-validation infrastructure.

    Args:
        n_samples: Total number of samples.
        seed: Seed offset for the deterministic function.

    Returns:
        Array of continuous targets with shape (n_samples,).
    """
    y: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        y[i] = float((i + seed) % 97) / 10.0 + 1.0
    return y


def make_regression_features(
    y: NDArray[np.float64],
    n_features: int = 5,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Create feature matrix with linear relationship to targets.

    Features are generated so that a simple linear model can predict y
    from the first feature. Additional features add noise.

    Args:
        y: Target values to create predictive features for.
        n_features: Number of features to generate.
        seed: Random seed for noise generation.

    Returns:
        Feature matrix of shape (n_samples, n_features).
    """
    n_samples = len(y)
    x: NDArray[np.float64] = np.zeros((n_samples, n_features), dtype=np.float64)

    for i in range(n_samples):
        # First feature is linearly correlated with target
        target_val = float(y.item(i))
        x[i, 0] = target_val * 0.5 + float((i + seed) % 13) / 100.0
        # Remaining features are deterministic noise
        for j in range(1, n_features):
            x[i, j] = float((i * 7 + j * 3 + seed) % 100) / 100.0

    return x


def make_test_regression_cv_split_info(
    n_samples: int = 100,
    n_folds: int = 3,
) -> CVSplitInfo:
    """Create a CVSplitInfo for regression testing.

    Generates simple non-overlapping splits for test verification.
    No stratification (regression targets are continuous).

    Args:
        n_samples: Total number of samples.
        n_folds: Number of folds to create.

    Returns:
        CVSplitInfo with simple non-overlapping splits.
    """
    fold_size = n_samples // n_folds
    indices: NDArray[np.intp] = np.arange(n_samples, dtype=np.intp)
    folds: list[CVSplit] = []

    for fold_num in range(n_folds):
        val_start = fold_num * fold_size
        val_end = val_start + fold_size if fold_num < n_folds - 1 else n_samples

        val_indices = indices[val_start:val_end]
        train_mask: NDArray[np.bool_] = np.ones(n_samples, dtype=np.bool_)
        train_mask[val_start:val_end] = False
        train_indices: NDArray[np.intp] = indices[train_mask]

        folds.append(
            CVSplit(
                fold_number=fold_num,
                train_indices=train_indices,
                val_indices=val_indices,
            )
        )

    return CVSplitInfo(
        n_folds=n_folds,
        n_samples=n_samples,
        folds=tuple(folds),
    )


__all__ = [
    "make_regression_features",
    "make_regression_targets",
    "make_test_regression_cv_split_info",
]
