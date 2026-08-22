"""Shared fixtures and helpers for test_runner splits."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def _make_labels(n_pos: int, n_neg: int, seed: int = 42) -> NDArray[np.int64]:
    """Create shuffled binary label array.

    Args:
        n_pos: Number of positive samples.
        n_neg: Number of negative samples.
        seed: Random seed for shuffling.

    Returns:
        Shuffled label array.
    """
    pos: NDArray[np.int64] = np.ones(n_pos, dtype=np.int64)
    neg: NDArray[np.int64] = np.zeros(n_neg, dtype=np.int64)
    result: NDArray[np.int64] = np.concatenate([pos, neg])
    rng = np.random.default_rng(seed)
    rng.shuffle(result)
    return result


def _get_label(y: NDArray[np.int64], idx: int) -> int:
    """Get label at index with proper typing."""
    return int(y.item(idx))


def _get_feature(x: NDArray[np.float64], row: int, col: int) -> float:
    """Get feature value at position with proper typing."""
    return float(x.item((row, col)))


def _set_feature(x: NDArray[np.float64], row: int, col: int, value: float) -> None:
    """Set feature value at position."""
    x[row, col] = value


def _compute_mean_1d(arr: NDArray[np.float64]) -> float:
    """Compute mean of 1D array using iteration."""
    n = len(arr)
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        total += float(arr.item(i))
    return total / n


def _make_separable_features(
    y: NDArray[np.int64],
    n_features: int,
    separation: float = 2.0,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Create feature matrix where classes are linearly separable.

    This allows a simple model to achieve high AUC for testing.

    Args:
        y: Labels to create separable features for.
        n_features: Number of features.
        separation: How far apart class centers are.
        seed: Random seed.

    Returns:
        Feature matrix where positives have higher mean values.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(y)
    x: NDArray[np.float64] = rng.standard_normal((n_samples, n_features)).astype(np.float64)

    # Shift positive samples to have higher values
    for i in range(n_samples):
        if _get_label(y, i) == 1:
            current = _get_feature(x, i, 0)
            _set_feature(x, i, 0, current + separation)

    return x


def _sigmoid(z: float) -> float:
    """Compute sigmoid function."""
    return 1.0 / (1.0 + math.exp(-z))


def _check_probabilities_valid(oof: NDArray[np.float64]) -> bool:
    """Check if all values are valid probabilities."""
    for i in range(len(oof)):
        p = float(oof.item(i))
        if p < 0.0 or p > 1.0:
            return False
    return True


class SimpleLogisticModel:
    """Simple logistic-like model for testing.

    Uses mean of first feature as decision boundary.
    """

    def __init__(self, threshold: float, scale: float) -> None:
        """Initialize model.

        Args:
            threshold: Decision threshold on first feature.
            scale: Scaling factor for probability computation.
        """
        self._threshold = threshold
        self._scale = scale

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities.

        Uses sigmoid of (first_feature - threshold) * scale.

        Args:
            x: Feature matrix.

        Returns:
            Probabilities for class 1.
        """
        n_samples = int(x.shape[0])
        proba: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):
            val = _get_feature(x, i, 0)
            z = (val - self._threshold) * self._scale
            proba[i] = _sigmoid(z)

        return proba


def simple_trainer(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    x_val: NDArray[np.float64],
    y_val: NDArray[np.int64],
    fold_number: int,
) -> SimpleLogisticModel:
    """Train simple model for testing.

    Computes optimal threshold from training data.

    Args:
        x_train: Training features.
        y_train: Training labels.
        x_val: Validation features (unused).
        y_val: Validation labels (unused).
        fold_number: Fold number (unused).

    Returns:
        Trained SimpleLogisticModel.
    """
    _ = x_val, y_val, fold_number  # Unused

    # Compute class means on first feature
    pos_mask: NDArray[np.bool_] = y_train == 1
    neg_mask: NDArray[np.bool_] = y_train == 0

    # Extract first column for each class
    pos_vals: NDArray[np.float64] = x_train[pos_mask, 0]
    neg_vals: NDArray[np.float64] = x_train[neg_mask, 0]

    pos_mean = _compute_mean_1d(pos_vals) if len(pos_vals) > 0 else 0.0
    neg_mean = _compute_mean_1d(neg_vals) if len(neg_vals) > 0 else 0.0

    threshold = (pos_mean + neg_mean) / 2.0
    scale = 2.0  # Fixed scale

    return SimpleLogisticModel(threshold, scale)


def _make_groups(samples_per_group: tuple[int, ...]) -> NDArray[np.int64]:
    """Create group ID array with specified samples per group.

    Args:
        samples_per_group: Tuple of sample counts per group.

    Returns:
        Array of group IDs.
    """
    groups: list[int] = []
    for group_id, count in enumerate(samples_per_group):
        groups.extend([group_id] * count)
    result: NDArray[np.int64] = np.array(groups, dtype=np.int64)
    return result


def _make_labels_for_groups(
    samples_per_group: tuple[int, ...],
    positive_groups: set[int],
    seed: int = 42,
) -> NDArray[np.int64]:
    """Create label array where specified groups have positive samples.

    Args:
        samples_per_group: Tuple of sample counts per group.
        positive_groups: Group IDs that should have positive samples.
        seed: Random seed for shuffling within groups.

    Returns:
        Binary label array.
    """
    n_samples = sum(samples_per_group)
    labels: NDArray[np.int64] = np.zeros(n_samples, dtype=np.int64)

    idx = 0
    for group_id, count in enumerate(samples_per_group):
        if group_id in positive_groups:
            labels[idx] = 1
        idx += count

    return labels


def _make_separable_features_for_groups(
    y: NDArray[np.int64],
    groups: NDArray[np.int64],
    n_features: int,
    separation: float = 2.0,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Create features separable by label with group consistency.

    Args:
        y: Labels.
        groups: Group IDs.
        n_features: Number of features.
        separation: Class separation.
        seed: Random seed.

    Returns:
        Feature matrix.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(y)
    x: NDArray[np.float64] = rng.standard_normal((n_samples, n_features)).astype(np.float64)

    # Shift positive samples
    for i in range(n_samples):
        if _get_label(y, i) == 1:
            current = _get_feature(x, i, 0)
            _set_feature(x, i, 0, current + separation)

    return x


def _get_groups_for_indices(
    groups: NDArray[np.int64],
    indices: NDArray[np.intp],
) -> set[int]:
    """Get unique group IDs for given indices."""
    result: set[int] = set()
    for i in range(len(indices)):
        idx = int(indices.item(i))
        group_id = int(groups.item(idx))
        result.add(group_id)
    return result
