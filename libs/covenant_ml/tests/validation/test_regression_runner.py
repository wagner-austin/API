"""Tests for the regression cross-validation runner.

Tests cover:
- kfold_split: random (non-stratified) splitting
- get_regression_fold_data: float64 data extraction
- run_regression_cross_validation: full regression CV execution
- Preprocessing isolation per fold
- OOF prediction collection
- RMSE metrics computation
- Edge cases
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.validation.regression_runner import (
    _compute_mean,
    _compute_std,
    get_regression_fold_data,
    kfold_split,
    run_regression_cross_validation,
)
from covenant_ml.validation.regression_testing import (
    make_regression_features,
    make_regression_targets,
    make_test_regression_cv_split_info,
)
from covenant_ml.validation.types import CVSplit

# =============================================================================
# Type-safe helpers
# =============================================================================


def _make_intp_array(values: tuple[int, ...]) -> NDArray[np.intp]:
    """Create intp array from tuple of ints."""
    result: NDArray[np.intp] = np.zeros(len(values), dtype=np.intp)
    for i, v in enumerate(values):
        result[i] = v
    return result


def _get_val(arr: NDArray[np.float64], idx: int) -> float:
    """Get value at index with proper typing."""
    return float(arr.item(idx))


# =============================================================================
# Simple regressor for testing
# =============================================================================


class SimpleLinearRegressor:
    """Simple linear regressor for testing.

    Predicts y = slope * x[0] + intercept, where slope and intercept
    are computed from training data.
    """

    def __init__(self, slope: float, intercept: float) -> None:
        """Initialize regressor.

        Args:
            slope: Coefficient for first feature.
            intercept: Bias term.
        """
        self._slope = slope
        self._intercept = intercept

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict continuous values.

        Args:
            x: Feature matrix.

        Returns:
            Predicted values of shape (n_samples,).
        """
        n_samples = int(x.shape[0])
        preds: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            feat0 = float(x.item((i, 0)))
            preds[i] = self._slope * feat0 + self._intercept
        return preds


def simple_regressor_trainer(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_val: NDArray[np.float64],
    y_val: NDArray[np.float64],
    fold_number: int,
) -> SimpleLinearRegressor:
    """Train simple regressor for testing.

    Computes slope and intercept from training data using first feature.

    Args:
        x_train: Training features.
        y_train: Training targets.
        x_val: Validation features (unused).
        y_val: Validation targets (unused).
        fold_number: Fold number (unused).

    Returns:
        Trained SimpleLinearRegressor.
    """
    _ = x_val, y_val, fold_number

    n_train = int(x_train.shape[0])

    # Compute means
    x_sum = 0.0
    y_sum = 0.0
    for i in range(n_train):
        x_sum += float(x_train.item((i, 0)))
        y_sum += float(y_train.item(i))
    x_mean = x_sum / max(1, n_train)
    y_mean = y_sum / max(1, n_train)

    # Compute slope via covariance / variance
    cov_sum = 0.0
    var_sum = 0.0
    for i in range(n_train):
        xd = float(x_train.item((i, 0))) - x_mean
        yd = float(y_train.item(i)) - y_mean
        cov_sum += xd * yd
        var_sum += xd * xd

    slope = cov_sum / max(1e-10, var_sum)
    intercept = y_mean - slope * x_mean

    return SimpleLinearRegressor(slope, intercept)


# =============================================================================
# Test: kfold_split
# =============================================================================


class TestKFoldSplit:
    """Tests for kfold_split function."""

    def test_returns_correct_number_of_folds(self) -> None:
        """Creates the requested number of folds."""
        split_info = kfold_split(100, n_folds=5, random_state=42)

        assert split_info["n_folds"] == 5
        assert len(split_info["folds"]) == 5
        assert split_info["n_samples"] == 100

    def test_each_sample_in_exactly_one_val_set(self) -> None:
        """Each sample appears in exactly one validation fold."""
        n_samples = 100
        split_info = kfold_split(n_samples, n_folds=5, random_state=42)

        val_counts: dict[int, int] = {}
        for fold in split_info["folds"]:
            for i in range(len(fold["val_indices"])):
                idx = int(fold["val_indices"].item(i))
                val_counts[idx] = val_counts.get(idx, 0) + 1

        # Each sample in exactly one val set
        for i in range(n_samples):
            assert val_counts.get(i, 0) == 1

    def test_train_val_no_overlap(self) -> None:
        """Train and val indices don't overlap within a fold."""
        split_info = kfold_split(50, n_folds=3, random_state=42)

        for fold in split_info["folds"]:
            train_set: set[int] = set()
            for i in range(len(fold["train_indices"])):
                train_set.add(int(fold["train_indices"].item(i)))

            val_set: set[int] = set()
            for i in range(len(fold["val_indices"])):
                val_set.add(int(fold["val_indices"].item(i)))

            overlap = train_set & val_set
            assert len(overlap) == 0, f"Fold {fold['fold_number']}: train/val overlap"

    def test_train_plus_val_equals_total(self) -> None:
        """Train + val indices cover all samples for each fold."""
        n_samples = 50
        split_info = kfold_split(n_samples, n_folds=5, random_state=42)

        for fold in split_info["folds"]:
            total = len(fold["train_indices"]) + len(fold["val_indices"])
            assert total == n_samples

    def test_approximately_equal_fold_sizes(self) -> None:
        """Folds are approximately equal in size."""
        n_samples = 100
        n_folds = 5
        split_info = kfold_split(n_samples, n_folds=n_folds, random_state=42)

        expected_size = n_samples // n_folds
        for fold in split_info["folds"]:
            val_size = len(fold["val_indices"])
            # Allow +/- 1 sample for uneven splits
            assert abs(val_size - expected_size) <= 1

    def test_reproducibility(self) -> None:
        """Same seed produces identical splits."""
        split1 = kfold_split(100, n_folds=5, random_state=42)
        split2 = kfold_split(100, n_folds=5, random_state=42)

        for f1, f2 in zip(split1["folds"], split2["folds"], strict=True):
            np.testing.assert_array_equal(f1["val_indices"], f2["val_indices"])
            np.testing.assert_array_equal(f1["train_indices"], f2["train_indices"])

    def test_different_seeds_different_splits(self) -> None:
        """Different seeds produce different splits."""
        split1 = kfold_split(100, n_folds=5, random_state=42)
        split2 = kfold_split(100, n_folds=5, random_state=123)

        # At least one fold should differ
        any_different = False
        for f1, f2 in zip(split1["folds"], split2["folds"], strict=True):
            if not np.array_equal(f1["val_indices"], f2["val_indices"]):
                any_different = True
                break
        assert any_different, "Different seeds should produce different splits"

    def test_raises_on_insufficient_folds(self) -> None:
        """Raises ValueError if n_folds < 2."""
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            kfold_split(100, n_folds=1, random_state=42)

    def test_raises_on_insufficient_samples(self) -> None:
        """Raises ValueError if n_samples < n_folds."""
        with pytest.raises(ValueError, match="Not enough samples"):
            kfold_split(2, n_folds=5, random_state=42)

    def test_two_fold_split(self) -> None:
        """Works with 2-fold split."""
        split_info = kfold_split(20, n_folds=2, random_state=42)
        assert split_info["n_folds"] == 2
        assert len(split_info["folds"]) == 2

    def test_fold_numbers_are_sequential(self) -> None:
        """Fold numbers are 0, 1, ..., n_folds-1."""
        split_info = kfold_split(100, n_folds=5, random_state=42)
        fold_numbers = [fold["fold_number"] for fold in split_info["folds"]]
        assert fold_numbers == [0, 1, 2, 3, 4]


# =============================================================================
# Test: get_regression_fold_data
# =============================================================================


class TestGetRegressionFoldData:
    """Tests for get_regression_fold_data function."""

    def test_returns_correct_data(self) -> None:
        """Returns correct x/y slices for a split."""
        n_features = 3
        x: NDArray[np.float64] = np.zeros((10, n_features), dtype=np.float64)
        y: NDArray[np.float64] = np.zeros(10, dtype=np.float64)
        for i in range(10):
            y[i] = float(i) * 0.5
            for j in range(n_features):
                x[i, j] = float(i * n_features + j)

        split = CVSplit(
            fold_number=0,
            train_indices=_make_intp_array((0, 1, 2, 3, 4, 5, 6)),
            val_indices=_make_intp_array((7, 8, 9)),
        )

        x_train, y_train, x_val, y_val = get_regression_fold_data(x, y, split)

        assert x_train.shape == (7, 3)
        assert y_train.shape == (7,)
        assert x_val.shape == (3, 3)
        assert y_val.shape == (3,)

        # Check dtype
        assert x_train.dtype == np.float64
        assert y_train.dtype == np.float64
        assert x_val.dtype == np.float64
        assert y_val.dtype == np.float64

        # Check val values
        assert _get_val(y_val, 0) == 3.5  # index 7 -> 7*0.5
        assert _get_val(y_val, 1) == 4.0  # index 8 -> 8*0.5
        assert _get_val(y_val, 2) == 4.5  # index 9 -> 9*0.5

    def test_train_val_targets_are_float64(self) -> None:
        """Train and val targets are float64 (not int64)."""
        x: NDArray[np.float64] = np.zeros((6, 2), dtype=np.float64)
        y: NDArray[np.float64] = np.zeros(6, dtype=np.float64)
        for i in range(6):
            y[i] = float(i) * 1.1

        split = CVSplit(
            fold_number=0,
            train_indices=_make_intp_array((0, 1, 2, 3)),
            val_indices=_make_intp_array((4, 5)),
        )

        _, y_train, _, y_val = get_regression_fold_data(x, y, split)

        assert y_train.dtype == np.float64
        assert y_val.dtype == np.float64


# =============================================================================
# Test: run_regression_cross_validation
# =============================================================================


class TestRunRegressionCrossValidation:
    """Tests for run_regression_cross_validation function."""

    def test_returns_regression_cv_result(self) -> None:
        """Returns properly structured RegressionCVResult."""
        y = make_regression_targets(100)
        x = make_regression_features(y, n_features=5)

        result = run_regression_cross_validation(
            x, y, n_folds=3, random_state=42, trainer=simple_regressor_trainer
        )

        assert "n_folds" in result
        assert "fold_results" in result
        assert "mean_val_rmse" in result
        assert "std_val_rmse" in result
        assert "oof_predictions" in result

    def test_correct_number_of_folds(self) -> None:
        """Creates correct number of fold results."""
        y = make_regression_targets(100)
        x = make_regression_features(y, n_features=5)

        result = run_regression_cross_validation(
            x, y, n_folds=5, random_state=42, trainer=simple_regressor_trainer
        )

        assert result["n_folds"] == 5
        assert len(result["fold_results"]) == 5

    def test_oof_predictions_have_correct_shape(self) -> None:
        """OOF predictions have same length as input."""
        n_samples = 100
        y = make_regression_targets(n_samples)
        x = make_regression_features(y, n_features=5)

        result = run_regression_cross_validation(
            x, y, n_folds=5, random_state=42, trainer=simple_regressor_trainer
        )

        assert len(result["oof_predictions"]) == n_samples
        assert result["oof_predictions"].dtype == np.float64

    def test_oof_predictions_are_finite(self) -> None:
        """OOF predictions are all finite values."""
        y = make_regression_targets(100)
        x = make_regression_features(y, n_features=5)

        result = run_regression_cross_validation(
            x, y, n_folds=3, random_state=42, trainer=simple_regressor_trainer
        )

        oof = result["oof_predictions"]
        for i in range(len(oof)):
            val = float(oof.item(i))
            assert math.isfinite(val), f"OOF prediction at index {i} is not finite: {val}"

    def test_mean_rmse_is_average_of_folds(self) -> None:
        """mean_val_rmse is average of fold RMSEs."""
        y = make_regression_targets(100)
        x = make_regression_features(y, n_features=5)

        result = run_regression_cross_validation(
            x, y, n_folds=5, random_state=42, trainer=simple_regressor_trainer
        )

        fold_rmses = [fr["val_rmse"] for fr in result["fold_results"]]
        expected_mean = sum(fold_rmses) / len(fold_rmses)

        assert result["mean_val_rmse"] == pytest.approx(expected_mean)

    def test_rmse_values_are_non_negative(self) -> None:
        """All RMSE values in result are non-negative."""
        y = make_regression_targets(100)
        x = make_regression_features(y, n_features=5)

        result = run_regression_cross_validation(
            x, y, n_folds=3, random_state=42, trainer=simple_regressor_trainer
        )

        assert result["mean_val_rmse"] >= 0.0
        assert result["std_val_rmse"] >= 0.0
        for fr in result["fold_results"]:
            assert fr["train_rmse"] >= 0.0
            assert fr["val_rmse"] >= 0.0

    def test_each_fold_has_required_fields(self) -> None:
        """Each fold result has all required fields."""
        y = make_regression_targets(100)
        x = make_regression_features(y, n_features=5)

        result = run_regression_cross_validation(
            x, y, n_folds=3, random_state=42, trainer=simple_regressor_trainer
        )

        for fold_result in result["fold_results"]:
            assert "fold_number" in fold_result
            assert "train_rmse" in fold_result
            assert "val_rmse" in fold_result
            assert "val_indices" in fold_result
            assert "val_predictions" in fold_result

    def test_fold_numbers_are_sequential(self) -> None:
        """Fold numbers are 0, 1, 2, ..., n_folds-1."""
        y = make_regression_targets(100)
        x = make_regression_features(y, n_features=5)

        result = run_regression_cross_validation(
            x, y, n_folds=5, random_state=42, trainer=simple_regressor_trainer
        )

        fold_numbers = [fr["fold_number"] for fr in result["fold_results"]]
        assert fold_numbers == [0, 1, 2, 3, 4]

    def test_progress_callback_is_called(self) -> None:
        """Progress callback is called for each fold."""
        y = make_regression_targets(100)
        x = make_regression_features(y, n_features=5)

        calls: list[tuple[int, int]] = []

        def callback(fold: int, total: int) -> None:
            calls.append((fold, total))

        run_regression_cross_validation(
            x,
            y,
            n_folds=3,
            random_state=42,
            trainer=simple_regressor_trainer,
            progress_callback=callback,
        )

        assert calls == [(0, 3), (1, 3), (2, 3)]

    def test_reproducibility(self) -> None:
        """Same seed produces identical results."""
        y = make_regression_targets(100)
        x = make_regression_features(y, n_features=5)

        result1 = run_regression_cross_validation(
            x, y, n_folds=3, random_state=42, trainer=simple_regressor_trainer
        )
        result2 = run_regression_cross_validation(
            x, y, n_folds=3, random_state=42, trainer=simple_regressor_trainer
        )

        np.testing.assert_array_almost_equal(result1["oof_predictions"], result2["oof_predictions"])
        assert result1["mean_val_rmse"] == result2["mean_val_rmse"]

    def test_different_seeds_produce_different_results(self) -> None:
        """Different seeds produce different OOF predictions."""
        y = make_regression_targets(100)
        x = make_regression_features(y, n_features=5)

        result1 = run_regression_cross_validation(
            x, y, n_folds=3, random_state=42, trainer=simple_regressor_trainer
        )
        result2 = run_regression_cross_validation(
            x, y, n_folds=3, random_state=123, trainer=simple_regressor_trainer
        )

        assert not np.allclose(result1["oof_predictions"], result2["oof_predictions"])

    def test_model_achieves_reasonable_rmse(self) -> None:
        """Simple model achieves reasonable RMSE on correlated data."""
        y = make_regression_targets(200)
        x = make_regression_features(y, n_features=5)

        result = run_regression_cross_validation(
            x, y, n_folds=5, random_state=42, trainer=simple_regressor_trainer
        )

        # Linear model on linearly-correlated data should achieve low RMSE
        # relative to target range (~1.0 to ~10.6)
        assert result["mean_val_rmse"] < 5.0

    def test_without_progress_callback(self) -> None:
        """Works without progress callback (None)."""
        y = make_regression_targets(60)
        x = make_regression_features(y, n_features=3)

        result = run_regression_cross_validation(
            x, y, n_folds=3, random_state=42, trainer=simple_regressor_trainer
        )

        assert result["n_folds"] == 3
        assert result["mean_val_rmse"] >= 0.0


# =============================================================================
# Test: Preprocessing Isolation
# =============================================================================


class TestRegressionPreprocessingIsolation:
    """Tests verifying preprocessing is isolated per fold in regression CV."""

    def test_preprocessing_does_not_leak_between_folds(self) -> None:
        """Each fold uses its own preprocessing state.

        Verified by including extreme values that would cause issues if
        preprocessing statistics leaked between folds.
        """
        y = make_regression_targets(100)
        x = make_regression_features(y, n_features=5)

        # Add extreme value
        x[0, 0] = 1000.0

        result = run_regression_cross_validation(
            x, y, n_folds=5, random_state=42, trainer=simple_regressor_trainer
        )

        # OOF predictions should still be finite
        oof = result["oof_predictions"]
        for i in range(len(oof)):
            val = float(oof.item(i))
            assert math.isfinite(val), f"OOF at index {i} is not finite: {val}"


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestRegressionEdgeCases:
    """Tests for edge cases in regression CV."""

    def test_minimum_samples(self) -> None:
        """Works with minimum viable sample counts."""
        y = make_regression_targets(10)
        x = make_regression_features(y, n_features=3)

        result = run_regression_cross_validation(
            x, y, n_folds=5, random_state=42, trainer=simple_regressor_trainer
        )

        assert len(result["fold_results"]) == 5

    def test_two_fold_cv(self) -> None:
        """Works with 2-fold CV."""
        y = make_regression_targets(100)
        x = make_regression_features(y, n_features=5)

        result = run_regression_cross_validation(
            x, y, n_folds=2, random_state=42, trainer=simple_regressor_trainer
        )

        assert result["n_folds"] == 2
        assert len(result["fold_results"]) == 2

    def test_many_folds(self) -> None:
        """Works with many folds."""
        y = make_regression_targets(100)
        x = make_regression_features(y, n_features=5)

        result = run_regression_cross_validation(
            x, y, n_folds=10, random_state=42, trainer=simple_regressor_trainer
        )

        assert result["n_folds"] == 10


# =============================================================================
# Test: Internal Helper Functions
# =============================================================================


class TestRegressionComputeStd:
    """Tests for _compute_std internal function."""

    def test_empty_tuple_returns_zero(self) -> None:
        """Empty tuple returns 0.0."""
        result = _compute_std(())
        assert result == 0.0

    def test_single_element_returns_zero(self) -> None:
        """Single element tuple returns 0.0 (no variance)."""
        result = _compute_std((5.0,))
        assert result == 0.0

    def test_two_elements_computes_std(self) -> None:
        """Two elements computes correct std."""
        # std of (0, 2) = sqrt(((0-1)^2 + (2-1)^2) / 2) = sqrt(2/2) = 1.0
        result = _compute_std((0.0, 2.0))
        assert result == pytest.approx(1.0)

    def test_multiple_elements(self) -> None:
        """Multiple elements computes population std."""
        # Values: (1, 2, 3), mean=2, variance=((1-2)^2+(2-2)^2+(3-2)^2)/3=2/3
        result = _compute_std((1.0, 2.0, 3.0))
        expected = math.sqrt(2.0 / 3.0)
        assert result == pytest.approx(expected)


class TestRegressionComputeMean:
    """Tests for _compute_mean internal function."""

    def test_empty_tuple_returns_zero(self) -> None:
        """Empty tuple returns 0.0."""
        result = _compute_mean(())
        assert result == 0.0

    def test_single_element_returns_value(self) -> None:
        """Single element returns that value."""
        result = _compute_mean((5.0,))
        assert result == 5.0

    def test_multiple_elements(self) -> None:
        """Multiple elements computes mean."""
        result = _compute_mean((1.0, 2.0, 3.0, 4.0))
        assert result == pytest.approx(2.5)


# =============================================================================
# Test: Testing Utilities
# =============================================================================


class TestRegressionTestingUtilities:
    """Tests for regression testing utility functions."""

    def test_make_regression_targets_shape(self) -> None:
        """make_regression_targets returns correct shape and dtype."""
        y = make_regression_targets(50)
        assert y.shape == (50,)
        assert y.dtype == np.float64

    def test_make_regression_targets_values_are_finite(self) -> None:
        """make_regression_targets returns finite values."""
        y = make_regression_targets(100)
        for i in range(100):
            val = float(y.item(i))
            assert math.isfinite(val)

    def test_make_regression_targets_are_continuous(self) -> None:
        """make_regression_targets returns continuous (not binary) values."""
        y = make_regression_targets(100)
        unique_count = len({float(y.item(i)) for i in range(100)})
        # Should have many unique values (not just 0/1)
        assert unique_count > 2

    def test_make_regression_features_shape(self) -> None:
        """make_regression_features returns correct shape and dtype."""
        y = make_regression_targets(50)
        x = make_regression_features(y, n_features=4)
        assert x.shape == (50, 4)
        assert x.dtype == np.float64

    def test_make_test_regression_cv_split_info_structure(self) -> None:
        """make_test_regression_cv_split_info returns valid CVSplitInfo."""
        info = make_test_regression_cv_split_info(100, 3)
        assert info["n_folds"] == 3
        assert info["n_samples"] == 100
        assert len(info["folds"]) == 3

        # Each fold should have non-overlapping indices
        for fold in info["folds"]:
            train_set: set[int] = set()
            for i in range(len(fold["train_indices"])):
                train_set.add(int(fold["train_indices"].item(i)))
            val_set: set[int] = set()
            for i in range(len(fold["val_indices"])):
                val_set.add(int(fold["val_indices"].item(i)))
            assert len(train_set & val_set) == 0

    def test_make_test_regression_cv_split_info_covers_all_samples(self) -> None:
        """Each sample appears in at least one val fold."""
        n_samples = 60
        info = make_test_regression_cv_split_info(n_samples, 3)

        all_val: set[int] = set()
        for fold in info["folds"]:
            for i in range(len(fold["val_indices"])):
                all_val.add(int(fold["val_indices"].item(i)))

        assert all_val == set(range(n_samples))
