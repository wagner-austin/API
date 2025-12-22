"""Tests for the OOF (out-of-fold) utilities.

Tests cover:
- OOF AUC computation
- OOF metrics computation
- Coverage validation
- Stacking utilities
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.validation import (
    CVResult,
    FoldResult,
)
from covenant_ml.validation.oof import (
    combine_oof_predictions,
    compute_oof_auc,
    compute_oof_metrics,
    get_oof_for_stacking,
    validate_oof_coverage,
)

# =============================================================================
# Type-safe array helpers
# =============================================================================


def _make_labels(n_pos: int, n_neg: int) -> NDArray[np.int64]:
    """Create binary label array.

    Args:
        n_pos: Number of positive samples.
        n_neg: Number of negative samples.

    Returns:
        Label array with positives first, then negatives.
    """
    pos: NDArray[np.int64] = np.ones(n_pos, dtype=np.int64)
    neg: NDArray[np.int64] = np.zeros(n_neg, dtype=np.int64)
    result: NDArray[np.int64] = np.concatenate([pos, neg])
    return result


def _make_perfect_predictions(y: NDArray[np.int64]) -> NDArray[np.float64]:
    """Create predictions that perfectly match labels.

    Args:
        y: Labels.

    Returns:
        Predictions: 1.0 for positive, 0.0 for negative.
    """
    result: NDArray[np.float64] = y.astype(np.float64)
    return result


def _make_random_predictions(n_samples: int, seed: int = 42) -> NDArray[np.float64]:
    """Create random predictions.

    Args:
        n_samples: Number of predictions.
        seed: Random seed.

    Returns:
        Random probabilities in [0, 1].
    """
    rng = np.random.default_rng(seed)
    result: NDArray[np.float64] = rng.random(n_samples).astype(np.float64)
    return result


def _compute_std_from_values(values: tuple[float, ...]) -> float:
    """Compute standard deviation from tuple of values.

    Args:
        values: Tuple of float values.

    Returns:
        Population standard deviation.
    """
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    var_sum = 0.0
    for v in values:
        diff = v - mean
        var_sum += diff * diff
    if n > 0:
        return math.sqrt(var_sum / n)
    return 0.0


def _make_cv_result(
    n_samples: int,
    n_folds: int,
    oof_predictions: NDArray[np.float64],
    val_aucs: tuple[float, ...],
) -> CVResult:
    """Create a CVResult for testing.

    Args:
        n_samples: Total number of samples.
        n_folds: Number of folds.
        oof_predictions: OOF predictions array.
        val_aucs: Tuple of validation AUCs per fold.

    Returns:
        CVResult with proper structure.
    """
    samples_per_fold = n_samples // n_folds

    fold_results: list[FoldResult] = []
    for fold_num in range(n_folds):
        start = fold_num * samples_per_fold
        end = start + samples_per_fold if fold_num < n_folds - 1 else n_samples

        val_indices: NDArray[np.intp] = np.arange(start, end, dtype=np.intp)
        val_predictions: NDArray[np.float64] = oof_predictions[start:end].copy()

        fold_results.append(
            FoldResult(
                fold_number=fold_num,
                train_auc=0.9,  # Fixed train AUC for testing
                val_auc=val_aucs[fold_num],
                val_indices=val_indices,
                val_predictions=val_predictions,
            )
        )

    mean_auc = sum(val_aucs) / len(val_aucs)
    std_auc = _compute_std_from_values(val_aucs)

    return CVResult(
        n_folds=n_folds,
        fold_results=tuple(fold_results),
        mean_val_auc=mean_auc,
        std_val_auc=std_auc,
        oof_predictions=oof_predictions,
    )


def _make_intp_array(values: tuple[int, ...]) -> NDArray[np.intp]:
    """Create intp array from tuple of ints."""
    result: NDArray[np.intp] = np.zeros(len(values), dtype=np.intp)
    for i, v in enumerate(values):
        result[i] = v
    return result


def _make_float64_array(values: tuple[float, ...]) -> NDArray[np.float64]:
    """Create float64 array from tuple of floats."""
    result: NDArray[np.float64] = np.zeros(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        result[i] = v
    return result


def _get_2d_value(arr: NDArray[np.float64], row: int, col: int) -> float:
    """Get value from 2D array with proper typing."""
    return float(arr.item((row, col)))


# =============================================================================
# Test: compute_oof_auc
# =============================================================================


class TestComputeOofAuc:
    """Tests for compute_oof_auc function."""

    def test_perfect_predictions_gives_auc_1(self) -> None:
        """Perfect predictions yield AUC of 1.0."""
        y = _make_labels(50, 50)
        oof = _make_perfect_predictions(y)

        auc = compute_oof_auc(y, oof)

        assert auc == pytest.approx(1.0)

    def test_inverse_predictions_gives_auc_0(self) -> None:
        """Inverted predictions yield AUC of 0.0."""
        y = _make_labels(50, 50)
        oof = 1.0 - _make_perfect_predictions(y)  # Invert

        auc = compute_oof_auc(y, oof)

        assert auc == pytest.approx(0.0)

    def test_random_predictions_gives_auc_around_05(self) -> None:
        """Random predictions yield AUC around 0.5."""
        y = _make_labels(100, 100)
        oof = _make_random_predictions(200)

        auc = compute_oof_auc(y, oof)

        assert 0.3 <= auc <= 0.7

    def test_returns_valid_auc_range(self) -> None:
        """Returns AUC value in valid range [0, 1]."""
        y = _make_labels(50, 50)
        oof = _make_random_predictions(100)

        auc = compute_oof_auc(y, oof)

        assert 0.0 <= auc <= 1.0


# =============================================================================
# Test: compute_oof_metrics
# =============================================================================


class TestComputeOofMetrics:
    """Tests for compute_oof_metrics function."""

    def test_returns_oof_metrics(self) -> None:
        """Returns properly structured OOFMetrics."""
        y = _make_labels(50, 50)
        oof = _make_perfect_predictions(y)
        cv_result = _make_cv_result(100, 5, oof, (0.95, 0.96, 0.94, 0.97, 0.95))

        metrics = compute_oof_metrics(y, cv_result)

        assert "oof_auc" in metrics
        assert "mean_fold_auc" in metrics
        assert "std_fold_auc" in metrics
        assert "eval_metrics" in metrics

    def test_oof_auc_matches_direct_computation(self) -> None:
        """oof_auc matches direct AUC computation."""
        y = _make_labels(50, 50)
        oof = _make_random_predictions(100, seed=123)
        cv_result = _make_cv_result(100, 5, oof, (0.6, 0.7, 0.65, 0.68, 0.72))

        metrics = compute_oof_metrics(y, cv_result)
        direct_auc = compute_oof_auc(y, oof)

        assert metrics["oof_auc"] == pytest.approx(direct_auc)

    def test_includes_full_eval_metrics(self) -> None:
        """eval_metrics contains all standard evaluation metrics."""
        y = _make_labels(50, 50)
        oof = _make_random_predictions(100)
        cv_result = _make_cv_result(100, 5, oof, (0.6, 0.7, 0.65, 0.68, 0.72))

        metrics = compute_oof_metrics(y, cv_result)
        eval_metrics = metrics["eval_metrics"]

        assert "loss" in eval_metrics
        assert "auc" in eval_metrics
        assert "accuracy" in eval_metrics
        assert "precision" in eval_metrics
        assert "recall" in eval_metrics
        assert "f1_score" in eval_metrics


# =============================================================================
# Test: validate_oof_coverage
# =============================================================================


class TestValidateOofCoverage:
    """Tests for validate_oof_coverage function."""

    def test_valid_coverage_returns_true(self) -> None:
        """Returns True when all samples covered exactly once."""
        oof = _make_random_predictions(100)
        cv_result = _make_cv_result(100, 5, oof, (0.6, 0.7, 0.65, 0.68, 0.72))

        is_valid = validate_oof_coverage(100, cv_result)

        assert is_valid is True

    def test_wrong_length_returns_false(self) -> None:
        """Returns False when OOF length doesn't match n_samples."""
        oof = _make_random_predictions(90)  # Wrong length
        cv_result = _make_cv_result(90, 5, oof, (0.6, 0.7, 0.65, 0.68, 0.72))

        is_valid = validate_oof_coverage(100, cv_result)  # Expect 100

        assert is_valid is False

    def test_duplicate_indices_returns_false(self) -> None:
        """Returns False when sample appears in multiple validation folds."""
        oof = _make_random_predictions(100)
        cv_result = _make_cv_result(100, 5, oof, (0.6, 0.7, 0.65, 0.68, 0.72))

        # Manually corrupt to have duplicate in val_indices
        fold_results = list(cv_result["fold_results"])
        # Add index 0 to fold 1's val_indices (already in fold 0)
        corrupted_fold1 = FoldResult(
            fold_number=1,
            train_auc=0.9,
            val_auc=0.7,
            val_indices=_make_intp_array((0, 20, 21, 22, 23)),  # 0 is duplicate
            val_predictions=_make_float64_array((0.5, 0.5, 0.5, 0.5, 0.5)),
        )
        fold_results[1] = corrupted_fold1

        corrupted_result = CVResult(
            n_folds=5,
            fold_results=tuple(fold_results),
            mean_val_auc=0.67,
            std_val_auc=0.03,
            oof_predictions=oof,
        )

        is_valid = validate_oof_coverage(100, corrupted_result)

        assert is_valid is False


# =============================================================================
# Test: get_oof_for_stacking
# =============================================================================


class TestGetOofForStacking:
    """Tests for get_oof_for_stacking function."""

    def test_returns_oof_predictions(self) -> None:
        """Returns the OOF predictions array."""
        oof = _make_random_predictions(100)
        cv_result = _make_cv_result(100, 5, oof, (0.6, 0.7, 0.65, 0.68, 0.72))

        result = get_oof_for_stacking(cv_result)

        np.testing.assert_array_equal(result, oof)

    def test_has_correct_shape(self) -> None:
        """Returned array has correct shape."""
        oof = _make_random_predictions(100)
        cv_result = _make_cv_result(100, 5, oof, (0.6, 0.7, 0.65, 0.68, 0.72))

        result = get_oof_for_stacking(cv_result)

        assert result.shape == (100,)


# =============================================================================
# Test: combine_oof_predictions
# =============================================================================


class TestCombineOofPredictions:
    """Tests for combine_oof_predictions function."""

    def test_combines_into_columns(self) -> None:
        """Stacks arrays as columns."""
        oof1 = _make_random_predictions(100, seed=1)
        oof2 = _make_random_predictions(100, seed=2)
        oof3 = _make_random_predictions(100, seed=3)

        combined = combine_oof_predictions((oof1, oof2, oof3))

        assert combined.shape == (100, 3)
        np.testing.assert_array_equal(combined[:, 0], oof1)
        np.testing.assert_array_equal(combined[:, 1], oof2)
        np.testing.assert_array_equal(combined[:, 2], oof3)

    def test_single_array(self) -> None:
        """Works with single array."""
        oof = _make_random_predictions(100)

        combined = combine_oof_predictions((oof,))

        assert combined.shape == (100, 1)

    def test_raises_on_empty_tuple(self) -> None:
        """Raises ValueError on empty input."""
        with pytest.raises(ValueError, match="At least one OOF array required"):
            combine_oof_predictions(())

    def test_raises_on_length_mismatch(self) -> None:
        """Raises ValueError when arrays have different lengths."""
        oof1 = _make_random_predictions(100)
        oof2 = _make_random_predictions(90)  # Wrong length

        with pytest.raises(ValueError, match="OOF array 1 has length 90"):
            combine_oof_predictions((oof1, oof2))

    def test_preserves_values(self) -> None:
        """Combined array contains exact original values."""
        oof1 = _make_float64_array((0.1, 0.2, 0.3))
        oof2 = _make_float64_array((0.4, 0.5, 0.6))

        combined = combine_oof_predictions((oof1, oof2))

        assert _get_2d_value(combined, 0, 0) == 0.1
        assert _get_2d_value(combined, 1, 0) == 0.2
        assert _get_2d_value(combined, 2, 1) == 0.6
