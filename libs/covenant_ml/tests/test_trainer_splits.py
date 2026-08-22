"""Tests for covenant_ml trainer module."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.trainer import (
    DataSplits,
    stratified_split,
)
from tests._trainer_fixtures import (
    _make_larger_data,
)


def test_stratified_split_creates_correct_sizes() -> None:
    """stratified_split creates splits with correct proportions."""
    x_features, y_labels = _make_larger_data(100)

    splits = stratified_split(
        x_features,
        y_labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    # Due to stratified per-class splitting, exact sizes may vary slightly
    assert 68 <= splits.n_train <= 72
    assert 13 <= splits.n_val <= 17
    assert 13 <= splits.n_test <= 17
    assert splits.n_total == 100


def test_stratified_split_maintains_class_proportions() -> None:
    """Stratified split maintains class balance in each split."""
    x_features, y_labels = _make_larger_data(100)

    splits = stratified_split(
        x_features,
        y_labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    # Original ratio is 50% positive
    original_ratio = float(np.sum(y_labels)) / len(y_labels)

    # Each split should have similar ratio (within tolerance)
    train_ratio = float(np.sum(splits.y_train)) / len(splits.y_train)
    val_ratio = float(np.sum(splits.y_val)) / len(splits.y_val)
    test_ratio = float(np.sum(splits.y_test)) / len(splits.y_test)

    assert abs(train_ratio - original_ratio) < 0.1
    assert abs(val_ratio - original_ratio) < 0.15
    assert abs(test_ratio - original_ratio) < 0.15


def test_stratified_split_raises_on_invalid_ratios() -> None:
    """stratified_split raises ValueError if ratios don't sum to 1.0."""
    x_features, y_labels = _make_larger_data(100)

    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        stratified_split(
            x_features,
            y_labels,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.2,  # Sum = 1.1
            random_state=42,
        )


def test_stratified_split_deterministic() -> None:
    """Same random_state produces same splits."""
    x_features, y_labels = _make_larger_data(100)

    splits1 = stratified_split(x_features, y_labels, 0.7, 0.15, 0.15, random_state=123)
    splits2 = stratified_split(x_features, y_labels, 0.7, 0.15, 0.15, random_state=123)

    assert np.array_equal(splits1.y_train, splits2.y_train)
    assert np.array_equal(splits1.y_val, splits2.y_val)
    assert np.array_equal(splits1.y_test, splits2.y_test)


def test_data_splits_properties() -> None:
    """DataSplits has correct property values."""
    x_train = np.zeros((70, 8), dtype=np.float64)
    y_train = np.zeros(70, dtype=np.int64)
    x_val = np.zeros((15, 8), dtype=np.float64)
    y_val = np.zeros(15, dtype=np.int64)
    x_test = np.zeros((15, 8), dtype=np.float64)
    y_test = np.zeros(15, dtype=np.int64)

    splits = DataSplits(x_train, y_train, x_val, y_val, x_test, y_test)

    assert splits.n_train == 70
    assert splits.n_val == 15
    assert splits.n_test == 15
    assert splits.n_total == 100


def _make_1d_array(values: tuple[float, ...]) -> NDArray[np.float64]:
    """Create a 1D float64 array from a tuple of floats."""
    arr: NDArray[np.float64] = np.zeros(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        arr[i] = v
    return arr


def _make_2d_array(rows: tuple[tuple[float, ...], ...]) -> NDArray[np.float64]:
    """Create a 2D float64 array from nested tuples."""
    n_rows = len(rows)
    n_cols = len(rows[0]) if n_rows > 0 else 0
    arr: NDArray[np.float64] = np.zeros((n_rows, n_cols), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            arr[i, j] = v
    return arr


def test_feature_scaler_constructor_validates_shape_mismatch() -> None:
    """FeatureScaler raises ValueError when mean and std shapes differ."""
    from covenant_ml.trainer import FeatureScaler

    mean = _make_1d_array((0.0, 1.0, 2.0))
    std = _make_1d_array((1.0, 1.0))  # Different shape

    with pytest.raises(ValueError, match=r"must have same shape"):
        FeatureScaler(mean=mean, std=std)


def test_feature_scaler_constructor_validates_1d_array() -> None:
    """FeatureScaler raises ValueError when mean is not 1D."""
    from covenant_ml.trainer import FeatureScaler

    mean = _make_2d_array(((0.0, 1.0), (2.0, 3.0)))  # 2D
    std = _make_2d_array(((1.0, 1.0), (1.0, 1.0)))

    with pytest.raises(ValueError, match=r"must be 1D array"):
        FeatureScaler(mean=mean, std=std)


def test_feature_scaler_properties() -> None:
    """FeatureScaler properties return correct values."""
    from covenant_ml.trainer import FeatureScaler

    mean = _make_1d_array((0.0, 1.0, 2.0))
    std = _make_1d_array((1.0, 2.0, 0.5))

    scaler = FeatureScaler(mean=mean, std=std)

    assert scaler.n_features == 3
    np.testing.assert_array_equal(scaler.mean, mean)
    np.testing.assert_array_equal(scaler.std, std)


def test_feature_scaler_transform_normalizes_correctly() -> None:
    """FeatureScaler.transform applies standardization correctly."""
    from covenant_ml.trainer import FeatureScaler

    mean = _make_1d_array((10.0, 20.0))
    std = _make_1d_array((2.0, 5.0))
    scaler = FeatureScaler(mean=mean, std=std)

    x = _make_2d_array(((10.0, 20.0), (12.0, 25.0), (8.0, 15.0)))
    result = scaler.transform(x)

    # Expected: (x - mean) / std
    expected = _make_2d_array(((0.0, 0.0), (1.0, 1.0), (-1.0, -1.0)))
    np.testing.assert_array_almost_equal(result, expected)


def test_feature_scaler_transform_raises_on_wrong_features() -> None:
    """FeatureScaler.transform raises ValueError on wrong feature count."""
    from covenant_ml.trainer import FeatureScaler

    mean = _make_1d_array((0.0, 1.0, 2.0))
    std = _make_1d_array((1.0, 1.0, 1.0))
    scaler = FeatureScaler(mean=mean, std=std)

    x = _make_2d_array(((1.0, 2.0),))  # 2 features, expected 3

    with pytest.raises(ValueError, match=r"Expected 3 features, got 2"):
        scaler.transform(x)


def _compute_mean(arr: NDArray[np.float64]) -> float:
    """Compute mean with explicit typing (avoids numpy Any returns)."""
    total = 0.0
    n = 0
    for elem in arr.flat:
        val: float = float(elem.item())
        total += val
        n += 1
    return total / n


def _compute_std(arr: NDArray[np.float64]) -> float:
    """Compute standard deviation with explicit typing."""
    mean_val = _compute_mean(arr)
    variance_sum = 0.0
    n = 0
    for elem in arr.flat:
        val: float = float(elem.item())
        variance_sum += (val - mean_val) ** 2
        n += 1
    std_val: float = (variance_sum / n) ** 0.5
    return std_val


def test_compute_feature_scaler_computes_correct_stats() -> None:
    """compute_feature_scaler computes mean and std from training data."""
    from covenant_ml.trainer import compute_feature_scaler

    x_train = _make_2d_array(((10.0, 100.0), (20.0, 200.0), (30.0, 300.0)))

    scaler = compute_feature_scaler(x_train)

    assert scaler.n_features == 2
    expected_mean = _make_1d_array((20.0, 200.0))
    np.testing.assert_array_almost_equal(scaler.mean, expected_mean)
    # std for [10,20,30] = sqrt(((10-20)^2+(20-20)^2+(30-20)^2)/3) = sqrt(200/3)
    col0: NDArray[np.float64] = x_train[:, 0]
    col1: NDArray[np.float64] = x_train[:, 1]
    expected_std_0: float = _compute_std(col0)
    expected_std_1: float = _compute_std(col1)
    expected_std = _make_1d_array((expected_std_0, expected_std_1))
    np.testing.assert_array_almost_equal(scaler.std, expected_std)


def test_compute_feature_scaler_handles_zero_variance() -> None:
    """compute_feature_scaler replaces zero std with 1.0 to avoid division by zero."""
    from covenant_ml.trainer import compute_feature_scaler

    # Column 1 has zero variance (all same value)
    x_train = _make_2d_array(((10.0, 5.0), (20.0, 5.0), (30.0, 5.0)))

    scaler = compute_feature_scaler(x_train)

    # Second column should have std=1.0 (not 0.0)
    std_arr: NDArray[np.float64] = scaler.std
    std_col1: float = float(std_arr.flat[1])
    assert std_col1 == 1.0
    # Transform should work without division by zero
    result = scaler.transform(x_train)
    # Second column: (5.0 - 5.0) / 1.0 = 0.0
    expected_col1 = _make_1d_array((0.0, 0.0, 0.0))
    result_col1: NDArray[np.float64] = result[:, 1]
    np.testing.assert_array_almost_equal(result_col1, expected_col1)
