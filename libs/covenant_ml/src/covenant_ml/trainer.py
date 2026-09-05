"""XGBoost model training for covenant breach prediction.

Enhanced trainer with:
- Train/validation/test splits
- Early stopping based on validation AUC
- Comprehensive metrics (loss, AUC, accuracy, precision, recall, F1)
- Progress callbacks for monitoring
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from .preprocessing import AutoPreprocessor, PreprocessingState

_log = get_logger(__name__)


class SplitData(Protocol):
    """Protocol for split dataset."""

    x: NDArray[np.float64]
    y: NDArray[np.int64]


class FeatureScaler:
    """Stores mean/std per feature column for standardization.

    Computes statistics from training data only (to avoid data leakage).
    Applies: x_normalized = (x - mean) / std

    Attributes:
        mean: Per-column mean values, shape (n_features,)
        std: Per-column standard deviation values, shape (n_features,)
        n_features: Number of features
    """

    def __init__(
        self,
        mean: NDArray[np.float64],
        std: NDArray[np.float64],
    ) -> None:
        if mean.shape != std.shape:
            raise ValueError(f"mean and std must have same shape: {mean.shape} vs {std.shape}")
        if mean.ndim != 1:
            raise ValueError(f"mean must be 1D array, got {mean.ndim}D")
        self._mean = mean
        self._std = std
        self._n_features = len(mean)

    @property
    def mean(self) -> NDArray[np.float64]:
        """Per-column mean values."""
        return self._mean

    @property
    def std(self) -> NDArray[np.float64]:
        """Per-column standard deviation values."""
        return self._std

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self._n_features

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply standardization: (x - mean) / std.

        Args:
            x: Feature matrix, shape (n_samples, n_features)

        Returns:
            Normalized feature matrix with same shape

        Raises:
            ValueError: If x has wrong number of features
        """
        n_cols: int = int(x.shape[1])
        if n_cols != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {n_cols}")
        return (x - self._mean) / self._std


def compute_feature_scaler(x_train: NDArray[np.float64]) -> FeatureScaler:
    """Compute feature scaling statistics from training data.

    Uses StandardScaler approach: mean=0, std=1 per column.
    Handles zero-variance columns by setting std=1.0 (no scaling).

    Args:
        x_train: Training feature matrix, shape (n_samples, n_features)

    Returns:
        FeatureScaler with mean/std computed from x_train
    """
    mean: NDArray[np.float64] = np.mean(x_train, axis=0)
    std: NDArray[np.float64] = np.std(x_train, axis=0)

    # Replace zero std with 1.0 to avoid division by zero
    # (columns with zero variance are effectively constant)
    zero_std_mask: NDArray[np.bool_] = std == 0.0
    std_safe: NDArray[np.float64] = np.where(zero_std_mask, 1.0, std)

    n_zero_std = int(np.count_nonzero(zero_std_mask))
    if n_zero_std > 0:
        _log.info(
            "Feature scaler found zero-variance columns",
            extra={"n_zero_variance_cols": n_zero_std},
        )

    return FeatureScaler(mean=mean, std=std_safe)


class DataSplits:
    """Container for train/val/test data splits."""

    def __init__(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.int64],
        x_test: NDArray[np.float64],
        y_test: NDArray[np.int64],
    ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

    @property
    def n_train(self) -> int:
        return len(self.y_train)

    @property
    def n_val(self) -> int:
        return len(self.y_val)

    @property
    def n_test(self) -> int:
        return len(self.y_test)

    @property
    def n_total(self) -> int:
        return self.n_train + self.n_val + self.n_test


class RegressionDataSplits:
    """Container for regression train/val/test data splits.

    Parallel to DataSplits for classification. Uses float64 targets
    instead of int64 labels.
    """

    def __init__(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.float64],
        x_test: NDArray[np.float64],
        y_test: NDArray[np.float64],
    ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return len(self.y_train)

    @property
    def n_val(self) -> int:
        """Number of validation samples."""
        return len(self.y_val)

    @property
    def n_test(self) -> int:
        """Number of test samples."""
        return len(self.y_test)

    @property
    def n_total(self) -> int:
        """Total number of samples across all splits."""
        return self.n_train + self.n_val + self.n_test


class PreprocessedDataSplits:
    """Container for preprocessed train/val/test data splits.

    Full preprocessing is applied using statistics computed from
    training data only (to avoid data leakage). Includes:
    - Outlier capping (percentile-based)
    - Special code replacement (96, 98, 999 → NaN)
    - Missing value imputation (median)
    - Z-score normalization
    """

    def __init__(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.int64],
        x_test: NDArray[np.float64],
        y_test: NDArray[np.int64],
        state: PreprocessingState,
    ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.state = state

    @property
    def n_train(self) -> int:
        return len(self.y_train)

    @property
    def n_val(self) -> int:
        return len(self.y_val)

    @property
    def n_test(self) -> int:
        return len(self.y_test)

    @property
    def n_total(self) -> int:
        return self.n_train + self.n_val + self.n_test


def preprocess_data_splits(splits: DataSplits) -> PreprocessedDataSplits:
    """Preprocess data splits using statistics from training data only.

    Applies full preprocessing pipeline:
    1. Outlier capping (1st/99th percentile bounds)
    2. Special code replacement (96, 98, 999, etc. → NaN)
    3. Missing value imputation (median)
    4. Z-score normalization (mean=0, std=1)

    All statistics are computed from training data only to prevent leakage.

    Args:
        splits: Raw data splits.

    Returns:
        PreprocessedDataSplits with cleaned and normalized features.
    """
    preprocessor = AutoPreprocessor()
    state = preprocessor.fit(splits.x_train, splits.y_train)

    _log.info(
        "Preprocessing data splits",
        extra={
            "n_features": state["n_features"],
            "n_outlier_bounds": len(state["outlier_bounds"]),
            "n_special_codes": len(state["special_codes"]),
            "n_imputation_values": len(state["imputation_values"]),
            "n_train": splits.n_train,
            "n_val": splits.n_val,
            "n_test": splits.n_test,
        },
    )

    return PreprocessedDataSplits(
        x_train=preprocessor.transform(splits.x_train, state),
        y_train=splits.y_train,
        x_val=preprocessor.transform(splits.x_val, state),
        y_val=splits.y_val,
        x_test=preprocessor.transform(splits.x_test, state),
        y_test=splits.y_test,
        state=state,
    )


def stratified_split(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
    groups: NDArray[np.int64] | None = None,
) -> DataSplits:
    """Split data into train/val/test with stratification.

    Maintains class proportions across all splits. When ``groups`` is given,
    the units being split are the groups, not the rows: every row of a group
    lands in the same split, and stratification is by each group's label.
    Rows within one group are correlated (e.g., 1,500 snapshots of one
    match), so a row-level split would place near-duplicates of training
    rows in the test set and score memorization as skill.

    Args:
        x_features: Feature matrix (n_samples, n_features)
        y_labels: Binary labels (n_samples,)
        train_ratio: Fraction for training (e.g., 0.7)
        val_ratio: Fraction for validation (e.g., 0.15)
        test_ratio: Fraction for test holdout (e.g., 0.15)
        random_state: Random seed for reproducibility
        groups: Optional group codes (n_samples,); rows sharing a code are
            one entity and are never separated across splits. A group's
            stratification label is its first row's label (constant per
            group for any honest grouped dataset).

    Returns:
        DataSplits container with train/val/test arrays

    Raises:
        ValueError: If ratios don't sum to 1.0 (within tolerance)
    """
    # Validate ratios sum to 1.0
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.3f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

    rng = np.random.default_rng(random_state)

    if groups is not None:
        return _grouped_split(x_features, y_labels, groups, train_ratio, val_ratio, test_ratio, rng)

    # Get indices for each class (np.where returns tuple, take first element)
    pos_mask: NDArray[np.bool_] = y_labels == 1
    neg_mask: NDArray[np.bool_] = y_labels == 0
    pos_indices: NDArray[np.intp] = np.flatnonzero(pos_mask)
    neg_indices: NDArray[np.intp] = np.flatnonzero(neg_mask)

    # Shuffle indices
    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)

    # Calculate split points for each class
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)

    pos_train_end = int(n_pos * train_ratio)
    pos_val_end = int(n_pos * (train_ratio + val_ratio))

    neg_train_end = int(n_neg * train_ratio)
    neg_val_end = int(n_neg * (train_ratio + val_ratio))

    # Split indices (test is remainder after train+val)
    train_idx: NDArray[np.intp] = np.concatenate(
        [
            pos_indices[:pos_train_end],
            neg_indices[:neg_train_end],
        ]
    )
    val_idx: NDArray[np.intp] = np.concatenate(
        [
            pos_indices[pos_train_end:pos_val_end],
            neg_indices[neg_train_end:neg_val_end],
        ]
    )
    test_idx: NDArray[np.intp] = np.concatenate(
        [
            pos_indices[pos_val_end:],
            neg_indices[neg_val_end:],
        ]
    )

    # Shuffle final indices
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    _log.info(
        "Data split complete",
        extra={
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
        },
    )

    return DataSplits(
        x_train=x_features[train_idx],
        y_train=y_labels[train_idx],
        x_val=x_features[val_idx],
        y_val=y_labels[val_idx],
        x_test=x_features[test_idx],
        y_test=y_labels[test_idx],
    )


def _grouped_split(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    groups: NDArray[np.int64],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
) -> DataSplits:
    """Split whole groups into train/val/test, stratified by group label.

    The same ratio arithmetic as the row split, applied to unique groups: a
    group's label is its first row's, positive and negative groups are
    shuffled and cut by the ratios, and every row follows its group. Row
    order within each final split is shuffled so downstream batch-wise
    consumers do not see one group as one contiguous block.

    Args:
        x_features: Feature matrix (n_samples, n_features)
        y_labels: Binary labels (n_samples,)
        groups: Group codes (n_samples,)
        train_ratio: Fraction of groups for training
        val_ratio: Fraction of groups for validation
        test_ratio: Fraction of groups for test (remainder)
        rng: Seeded generator shared with the caller

    Returns:
        DataSplits container with train/val/test arrays
    """
    unique_groups: NDArray[np.int64]
    first_row: NDArray[np.intp]
    unique_groups, first_row = np.unique(groups, return_index=True)
    group_labels: NDArray[np.int64] = y_labels[first_row]

    positive_mask: NDArray[np.bool_] = group_labels == 1
    pos_groups: NDArray[np.int64] = unique_groups[positive_mask]
    neg_groups: NDArray[np.int64] = unique_groups[~positive_mask]
    rng.shuffle(pos_groups)
    rng.shuffle(neg_groups)

    def cut(pool: NDArray[np.int64]) -> tuple[set[int], set[int], set[int]]:
        train_end = int(len(pool) * train_ratio)
        val_end = int(len(pool) * (train_ratio + val_ratio))
        as_ints: list[int] = []
        for i in range(len(pool)):
            code: np.int64 = pool[i]
            as_ints.append(int(code))
        return set(as_ints[:train_end]), set(as_ints[train_end:val_end]), set(as_ints[val_end:])

    pos_train, pos_val, pos_test = cut(pos_groups)
    neg_train, neg_val, neg_test = cut(neg_groups)

    def rows_of(members: set[int]) -> NDArray[np.intp]:
        keep: NDArray[np.bool_] = np.zeros(len(groups), dtype=np.bool_)
        for row_idx in range(len(groups)):
            row_code: np.int64 = groups[row_idx]
            keep[row_idx] = int(row_code) in members
        rows: NDArray[np.intp] = np.flatnonzero(keep)
        rng.shuffle(rows)
        return rows

    train_idx = rows_of(pos_train | neg_train)
    val_idx = rows_of(pos_val | neg_val)
    test_idx = rows_of(pos_test | neg_test)

    _log.info(
        "Grouped data split complete",
        extra={
            "n_groups": len(unique_groups),
            "n_train_groups": len(pos_train) + len(neg_train),
            "n_val_groups": len(pos_val) + len(neg_val),
            "n_test_groups": len(pos_test) + len(neg_test),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
        },
    )

    return DataSplits(
        x_train=x_features[train_idx],
        y_train=y_labels[train_idx],
        x_val=x_features[val_idx],
        y_val=y_labels[val_idx],
        x_test=x_features[test_idx],
        y_test=y_labels[test_idx],
    )


def regression_split(
    x_features: NDArray[np.float64],
    y_targets: NDArray[np.float64],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> RegressionDataSplits:
    """Split regression data into train/val/test sets.

    Uses random shuffling (not stratified — regression has no classes).

    Args:
        x_features: Feature matrix (n_samples, n_features).
        y_targets: Continuous target values (n_samples,).
        train_ratio: Fraction for training (e.g., 0.7).
        val_ratio: Fraction for validation (e.g., 0.15).
        test_ratio: Fraction for test holdout (e.g., 0.15).
        random_state: Random seed for reproducibility.

    Returns:
        RegressionDataSplits container with train/val/test arrays.

    Raises:
        ValueError: If ratios don't sum to 1.0 (within tolerance).
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.3f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

    n = len(y_targets)
    rng = np.random.default_rng(random_state)
    indices: NDArray[np.intp] = np.arange(n, dtype=np.intp)
    rng.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx: NDArray[np.intp] = indices[:train_end]
    val_idx: NDArray[np.intp] = indices[train_end:val_end]
    test_idx: NDArray[np.intp] = indices[val_end:]

    _log.info(
        "Regression data split complete",
        extra={
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
        },
    )

    return RegressionDataSplits(
        x_train=x_features[train_idx],
        y_train=y_targets[train_idx],
        x_val=x_features[val_idx],
        y_val=y_targets[val_idx],
        x_test=x_features[test_idx],
        y_test=y_targets[test_idx],
    )


__all__ = [
    "DataSplits",
    "FeatureScaler",
    "PreprocessedDataSplits",
    "RegressionDataSplits",
    "compute_feature_scaler",
    "preprocess_data_splits",
    "regression_split",
    "stratified_split",
]
