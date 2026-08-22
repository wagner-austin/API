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

from covenant_ml.validation.regression_runner import (
    _compute_mean,
    _compute_std,
    run_regression_cross_validation,
)
from covenant_ml.validation.regression_testing import (
    make_regression_features,
    make_regression_targets,
    make_test_regression_cv_split_info,
)
from tests.validation._regression_runner_fixtures import (
    simple_regressor_trainer,
)


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
