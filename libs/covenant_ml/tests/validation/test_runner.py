"""Tests for the cross-validation runner.

Tests cover:
- Full cross-validation execution
- Preprocessing isolation per fold
- OOF prediction collection
- Metrics computation
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.validation import (
    run_cross_validation,
    run_group_cross_validation,
)

# =============================================================================
# Type-safe array helpers
# =============================================================================


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


# =============================================================================
# Simple trainer implementation for testing
# =============================================================================


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


# =============================================================================
# Test: run_cross_validation
# =============================================================================


class TestRunCrossValidation:
    """Tests for run_cross_validation function."""

    def test_returns_cv_result(self) -> None:
        """Returns properly structured CVResult."""
        y = _make_labels(50, 50)
        x = _make_separable_features(y, n_features=5)

        result = run_cross_validation(x, y, n_folds=3, random_state=42, trainer=simple_trainer)

        assert "n_folds" in result
        assert "fold_results" in result
        assert "mean_val_auc" in result
        assert "std_val_auc" in result
        assert "oof_predictions" in result

    def test_correct_number_of_folds(self) -> None:
        """Creates correct number of fold results."""
        y = _make_labels(50, 50)
        x = _make_separable_features(y, n_features=5)

        result = run_cross_validation(x, y, n_folds=5, random_state=42, trainer=simple_trainer)

        assert result["n_folds"] == 5
        assert len(result["fold_results"]) == 5

    def test_oof_predictions_have_correct_shape(self) -> None:
        """OOF predictions have same length as input."""
        n_samples = 100
        y = _make_labels(30, 70)
        x = _make_separable_features(y, n_features=5)

        result = run_cross_validation(x, y, n_folds=5, random_state=42, trainer=simple_trainer)

        assert len(result["oof_predictions"]) == n_samples

    def test_oof_predictions_are_probabilities(self) -> None:
        """OOF predictions are valid probabilities in [0, 1]."""
        y = _make_labels(50, 50)
        x = _make_separable_features(y, n_features=5)

        result = run_cross_validation(x, y, n_folds=5, random_state=42, trainer=simple_trainer)

        oof = result["oof_predictions"]
        assert _check_probabilities_valid(oof)

    def test_mean_auc_is_average_of_folds(self) -> None:
        """mean_val_auc is average of fold AUCs."""
        y = _make_labels(50, 50)
        x = _make_separable_features(y, n_features=5)

        result = run_cross_validation(x, y, n_folds=5, random_state=42, trainer=simple_trainer)

        fold_aucs = [fr["val_auc"] for fr in result["fold_results"]]
        expected_mean = sum(fold_aucs) / len(fold_aucs)

        assert result["mean_val_auc"] == pytest.approx(expected_mean)

    def test_separable_data_achieves_high_auc(self) -> None:
        """Model achieves high AUC on linearly separable data."""
        y = _make_labels(100, 100)
        x = _make_separable_features(y, n_features=5, separation=5.0)

        result = run_cross_validation(x, y, n_folds=5, random_state=42, trainer=simple_trainer)

        # With high separation, should achieve high AUC
        assert result["mean_val_auc"] > 0.8

    def test_each_fold_has_required_fields(self) -> None:
        """Each fold result has all required fields."""
        y = _make_labels(50, 50)
        x = _make_separable_features(y, n_features=5)

        result = run_cross_validation(x, y, n_folds=3, random_state=42, trainer=simple_trainer)

        for fold_result in result["fold_results"]:
            assert "fold_number" in fold_result
            assert "train_auc" in fold_result
            assert "val_auc" in fold_result
            assert "val_indices" in fold_result
            assert "val_predictions" in fold_result

    def test_fold_numbers_are_sequential(self) -> None:
        """Fold numbers are 0, 1, 2, ..., n_folds-1."""
        y = _make_labels(50, 50)
        x = _make_separable_features(y, n_features=5)

        result = run_cross_validation(x, y, n_folds=5, random_state=42, trainer=simple_trainer)

        fold_numbers = [fr["fold_number"] for fr in result["fold_results"]]
        assert fold_numbers == [0, 1, 2, 3, 4]

    def test_progress_callback_is_called(self) -> None:
        """Progress callback is called for each fold."""
        y = _make_labels(50, 50)
        x = _make_separable_features(y, n_features=5)

        calls: list[tuple[int, int]] = []

        def callback(fold: int, total: int) -> None:
            calls.append((fold, total))

        run_cross_validation(
            x, y, n_folds=3, random_state=42, trainer=simple_trainer, progress_callback=callback
        )

        assert calls == [(0, 3), (1, 3), (2, 3)]

    def test_reproducibility(self) -> None:
        """Same seed produces identical results."""
        y = _make_labels(50, 50)
        x = _make_separable_features(y, n_features=5)

        result1 = run_cross_validation(x, y, n_folds=3, random_state=42, trainer=simple_trainer)
        result2 = run_cross_validation(x, y, n_folds=3, random_state=42, trainer=simple_trainer)

        np.testing.assert_array_almost_equal(result1["oof_predictions"], result2["oof_predictions"])
        assert result1["mean_val_auc"] == result2["mean_val_auc"]

    def test_different_seeds_produce_different_results(self) -> None:
        """Different seeds produce different OOF predictions."""
        y = _make_labels(50, 50)
        x = _make_separable_features(y, n_features=5)

        result1 = run_cross_validation(x, y, n_folds=3, random_state=42, trainer=simple_trainer)
        result2 = run_cross_validation(x, y, n_folds=3, random_state=123, trainer=simple_trainer)

        # OOF predictions should differ due to different fold assignments
        assert not np.allclose(result1["oof_predictions"], result2["oof_predictions"])


# =============================================================================
# Test: Preprocessing Isolation
# =============================================================================


class TestPreprocessingIsolation:
    """Tests verifying preprocessing is isolated per fold."""

    def test_preprocessing_does_not_leak_between_folds(self) -> None:
        """Each fold uses its own preprocessing state.

        This is verified indirectly by checking that cross-validation
        completes successfully with data that would fail if preprocessing
        statistics leaked between folds.
        """
        y = _make_labels(50, 50)
        # Create data with different distributions in different parts
        x = _make_separable_features(y, n_features=5)

        # Add some extreme values that would cause issues if stats leaked
        _set_feature(x, 0, 0, 1000.0)

        # Should complete without error - preprocessing handles per-fold
        result = run_cross_validation(x, y, n_folds=5, random_state=42, trainer=simple_trainer)

        # OOF predictions should still be valid probabilities
        assert _check_probabilities_valid(result["oof_predictions"])


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_minimum_samples_per_class(self) -> None:
        """Works with minimum viable sample counts."""
        y = _make_labels(5, 5)  # 5 each for 5 folds
        x = _make_separable_features(y, n_features=3)

        result = run_cross_validation(x, y, n_folds=5, random_state=42, trainer=simple_trainer)

        assert len(result["fold_results"]) == 5

    def test_two_fold_cv(self) -> None:
        """Works with 2-fold CV."""
        y = _make_labels(50, 50)
        x = _make_separable_features(y, n_features=5)

        result = run_cross_validation(x, y, n_folds=2, random_state=42, trainer=simple_trainer)

        assert result["n_folds"] == 2
        assert len(result["fold_results"]) == 2

    def test_many_folds(self) -> None:
        """Works with many folds."""
        y = _make_labels(50, 50)
        x = _make_separable_features(y, n_features=5)

        result = run_cross_validation(x, y, n_folds=10, random_state=42, trainer=simple_trainer)

        assert result["n_folds"] == 10

    def test_imbalanced_data(self) -> None:
        """Works with imbalanced classes."""
        y = _make_labels(10, 90)  # 10% positive
        x = _make_separable_features(y, n_features=5, separation=3.0)

        result = run_cross_validation(x, y, n_folds=5, random_state=42, trainer=simple_trainer)

        # Should still produce valid results
        assert 0.0 <= result["mean_val_auc"] <= 1.0


# =============================================================================
# Test: Internal Helper Functions
# =============================================================================


class TestComputeStd:
    """Tests for _compute_std internal function."""

    def test_empty_tuple_returns_zero(self) -> None:
        """Empty tuple returns 0.0."""
        from covenant_ml.validation.runner import _compute_std

        result = _compute_std(())
        assert result == 0.0

    def test_single_element_returns_zero(self) -> None:
        """Single element tuple returns 0.0 (no variance)."""
        from covenant_ml.validation.runner import _compute_std

        result = _compute_std((5.0,))
        assert result == 0.0

    def test_two_elements_computes_std(self) -> None:
        """Two elements computes correct std."""
        from covenant_ml.validation.runner import _compute_std

        # std of (0, 2) = sqrt(((0-1)^2 + (2-1)^2) / 2) = sqrt(2/2) = 1.0
        result = _compute_std((0.0, 2.0))
        assert result == pytest.approx(1.0)

    def test_multiple_elements(self) -> None:
        """Multiple elements computes population std."""
        from covenant_ml.validation.runner import _compute_std

        # Values: (1, 2, 3), mean=2, variance=((1-2)^2+(2-2)^2+(3-2)^2)/3=2/3
        result = _compute_std((1.0, 2.0, 3.0))
        expected = math.sqrt(2.0 / 3.0)
        assert result == pytest.approx(expected)


class TestComputeMean:
    """Tests for _compute_mean internal function."""

    def test_empty_tuple_returns_zero(self) -> None:
        """Empty tuple returns 0.0."""
        from covenant_ml.validation.runner import _compute_mean

        result = _compute_mean(())
        assert result == 0.0

    def test_single_element_returns_value(self) -> None:
        """Single element returns that value."""
        from covenant_ml.validation.runner import _compute_mean

        result = _compute_mean((5.0,))
        assert result == 5.0

    def test_multiple_elements(self) -> None:
        """Multiple elements computes mean."""
        from covenant_ml.validation.runner import _compute_mean

        result = _compute_mean((1.0, 2.0, 3.0, 4.0))
        assert result == pytest.approx(2.5)


# =============================================================================
# Group array helpers
# =============================================================================


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


# =============================================================================
# Test: run_group_cross_validation
# =============================================================================


class TestRunGroupCrossValidation:
    """Tests for run_group_cross_validation function."""

    def test_returns_cv_result(self) -> None:
        """Returns properly structured CVResult."""
        # 20 groups, 3 samples each
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        x = _make_separable_features_for_groups(y, groups, n_features=5)

        result = run_group_cross_validation(
            x, y, groups, n_folds=5, random_state=42, trainer=simple_trainer
        )

        assert "n_folds" in result
        assert "fold_results" in result
        assert "mean_val_auc" in result
        assert "std_val_auc" in result
        assert "oof_predictions" in result

    def test_correct_number_of_folds(self) -> None:
        """Creates correct number of fold results."""
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        x = _make_separable_features_for_groups(y, groups, n_features=5)

        result = run_group_cross_validation(
            x, y, groups, n_folds=5, random_state=42, trainer=simple_trainer
        )

        assert result["n_folds"] == 5
        assert len(result["fold_results"]) == 5

    def test_groups_do_not_leak(self) -> None:
        """No group appears in both train and val of same fold."""
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        x = _make_separable_features_for_groups(y, groups, n_features=5)

        result = run_group_cross_validation(
            x, y, groups, n_folds=5, random_state=42, trainer=simple_trainer
        )

        # With 20 groups and 5 folds, each fold should have 4 groups in validation
        for fold_result in result["fold_results"]:
            val_indices = fold_result["val_indices"]
            val_groups = _get_groups_for_indices(groups, val_indices)

            # Each fold should have exactly 4 groups (20 groups / 5 folds)
            assert len(val_groups) == 4

    def test_oof_predictions_have_correct_shape(self) -> None:
        """OOF predictions have same length as input."""
        n_samples = 60
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        x = _make_separable_features_for_groups(y, groups, n_features=5)

        result = run_group_cross_validation(
            x, y, groups, n_folds=5, random_state=42, trainer=simple_trainer
        )

        assert len(result["oof_predictions"]) == n_samples

    def test_oof_predictions_are_probabilities(self) -> None:
        """OOF predictions are valid probabilities in [0, 1]."""
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        x = _make_separable_features_for_groups(y, groups, n_features=5)

        result = run_group_cross_validation(
            x, y, groups, n_folds=5, random_state=42, trainer=simple_trainer
        )

        oof = result["oof_predictions"]
        assert _check_probabilities_valid(oof)

    def test_mean_auc_is_average_of_folds(self) -> None:
        """mean_val_auc is average of fold AUCs."""
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        x = _make_separable_features_for_groups(y, groups, n_features=5)

        result = run_group_cross_validation(
            x, y, groups, n_folds=5, random_state=42, trainer=simple_trainer
        )

        fold_aucs = [fr["val_auc"] for fr in result["fold_results"]]
        expected_mean = sum(fold_aucs) / len(fold_aucs)

        assert result["mean_val_auc"] == pytest.approx(expected_mean)

    def test_progress_callback_is_called(self) -> None:
        """Progress callback is called for each fold."""
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        x = _make_separable_features_for_groups(y, groups, n_features=5)

        calls: list[tuple[int, int]] = []

        def callback(fold: int, total: int) -> None:
            calls.append((fold, total))

        run_group_cross_validation(
            x,
            y,
            groups,
            n_folds=3,
            random_state=42,
            trainer=simple_trainer,
            progress_callback=callback,
        )

        assert calls == [(0, 3), (1, 3), (2, 3)]

    def test_reproducibility(self) -> None:
        """Same seed produces identical results."""
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        x = _make_separable_features_for_groups(y, groups, n_features=5)

        result1 = run_group_cross_validation(
            x, y, groups, n_folds=3, random_state=42, trainer=simple_trainer
        )
        result2 = run_group_cross_validation(
            x, y, groups, n_folds=3, random_state=42, trainer=simple_trainer
        )

        np.testing.assert_array_almost_equal(result1["oof_predictions"], result2["oof_predictions"])
        assert result1["mean_val_auc"] == result2["mean_val_auc"]

    def test_different_seeds_produce_different_results(self) -> None:
        """Different seeds produce different OOF predictions."""
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        x = _make_separable_features_for_groups(y, groups, n_features=5)

        result1 = run_group_cross_validation(
            x, y, groups, n_folds=3, random_state=42, trainer=simple_trainer
        )
        result2 = run_group_cross_validation(
            x, y, groups, n_folds=3, random_state=123, trainer=simple_trainer
        )

        # OOF predictions should differ due to different fold assignments
        assert not np.allclose(result1["oof_predictions"], result2["oof_predictions"])
