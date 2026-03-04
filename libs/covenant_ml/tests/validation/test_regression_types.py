"""Tests for regression cross-validation types.

Tests cover:
- RegressionFoldResult TypedDict structure
- RegressionCVResult TypedDict structure
- Type compatibility checks
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.validation.regression_types import (
    RegressionCVResult,
    RegressionFoldResult,
)

# =============================================================================
# Type-safe array helpers
# =============================================================================


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


# =============================================================================
# Test: RegressionFoldResult
# =============================================================================


class TestRegressionFoldResult:
    """Tests for RegressionFoldResult TypedDict."""

    def test_can_create_fold_result(self) -> None:
        """Can create a RegressionFoldResult with all required fields."""
        val_indices = _make_intp_array((0, 1, 2))
        val_predictions = _make_float64_array((1.5, 2.3, 0.8))

        result = RegressionFoldResult(
            fold_number=0,
            train_rmse=0.15,
            val_rmse=0.22,
            val_indices=val_indices,
            val_predictions=val_predictions,
        )

        assert result["fold_number"] == 0
        assert result["train_rmse"] == 0.15
        assert result["val_rmse"] == 0.22
        np.testing.assert_array_equal(result["val_indices"], val_indices)
        np.testing.assert_array_equal(result["val_predictions"], val_predictions)

    def test_fold_result_supports_dict_access(self) -> None:
        """RegressionFoldResult supports dict-style key access."""
        val_indices = _make_intp_array((0,))
        val_predictions = _make_float64_array((3.14,))

        result = RegressionFoldResult(
            fold_number=0,
            train_rmse=0.1,
            val_rmse=0.2,
            val_indices=val_indices,
            val_predictions=val_predictions,
        )

        keys = list(result.keys())
        assert "fold_number" in keys
        assert "train_rmse" in keys
        assert "val_rmse" in keys
        assert "val_indices" in keys
        assert "val_predictions" in keys

    def test_fold_result_rmse_values_are_non_negative(self) -> None:
        """RMSE values should be non-negative in a well-formed result."""
        val_indices = _make_intp_array((0, 1))
        val_predictions = _make_float64_array((1.0, 2.0))

        result = RegressionFoldResult(
            fold_number=0,
            train_rmse=0.0,
            val_rmse=0.5,
            val_indices=val_indices,
            val_predictions=val_predictions,
        )

        assert result["train_rmse"] >= 0.0
        assert result["val_rmse"] >= 0.0


# =============================================================================
# Test: RegressionCVResult
# =============================================================================


class TestRegressionCVResult:
    """Tests for RegressionCVResult TypedDict."""

    def test_can_create_cv_result(self) -> None:
        """Can create a RegressionCVResult with all required fields."""
        val_indices = _make_intp_array((0, 1))
        val_predictions = _make_float64_array((1.5, 2.5))

        fold_result = RegressionFoldResult(
            fold_number=0,
            train_rmse=0.15,
            val_rmse=0.25,
            val_indices=val_indices,
            val_predictions=val_predictions,
        )

        oof = _make_float64_array((1.5, 2.5))

        cv_result = RegressionCVResult(
            n_folds=1,
            fold_results=(fold_result,),
            mean_val_rmse=0.25,
            std_val_rmse=0.0,
            oof_predictions=oof,
        )

        assert cv_result["n_folds"] == 1
        assert len(cv_result["fold_results"]) == 1
        assert cv_result["mean_val_rmse"] == 0.25
        assert cv_result["std_val_rmse"] == 0.0

    def test_cv_result_with_multiple_folds(self) -> None:
        """Can create RegressionCVResult with multiple folds."""
        folds: list[RegressionFoldResult] = []
        for i in range(5):
            val_indices_list = tuple(i * 10 + j for j in range(10))
            val_indices = _make_intp_array(val_indices_list)
            pred_vals = tuple(float(j) * 0.1 + 1.0 for j in range(10))
            val_predictions = _make_float64_array(pred_vals)
            folds.append(
                RegressionFoldResult(
                    fold_number=i,
                    train_rmse=0.1 + float(i) * 0.01,
                    val_rmse=0.2 + float(i) * 0.02,
                    val_indices=val_indices,
                    val_predictions=val_predictions,
                )
            )

        oof = _make_float64_array(tuple(float(i) * 0.1 for i in range(50)))

        cv_result = RegressionCVResult(
            n_folds=5,
            fold_results=tuple(folds),
            mean_val_rmse=0.24,
            std_val_rmse=0.028,
            oof_predictions=oof,
        )

        assert cv_result["n_folds"] == 5
        assert len(cv_result["fold_results"]) == 5

    def test_cv_result_oof_predictions_shape(self) -> None:
        """OOF predictions have expected shape."""
        val_indices = _make_intp_array((0, 1, 2, 3, 4))
        val_predictions = _make_float64_array((1.0, 2.0, 3.0, 4.0, 5.0))

        fold_result = RegressionFoldResult(
            fold_number=0,
            train_rmse=0.1,
            val_rmse=0.2,
            val_indices=val_indices,
            val_predictions=val_predictions,
        )

        oof = _make_float64_array((1.0, 2.0, 3.0, 4.0, 5.0))

        cv_result = RegressionCVResult(
            n_folds=1,
            fold_results=(fold_result,),
            mean_val_rmse=0.2,
            std_val_rmse=0.0,
            oof_predictions=oof,
        )

        assert len(cv_result["oof_predictions"]) == 5
        assert cv_result["oof_predictions"].dtype == np.float64
