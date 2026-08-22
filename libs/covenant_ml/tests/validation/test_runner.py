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

from covenant_ml.validation import (
    run_cross_validation,
)
from tests.validation._runner_fixtures import (
    _check_probabilities_valid,
    _make_labels,
    _make_separable_features,
    _set_feature,
    simple_trainer,
)


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
