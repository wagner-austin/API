"""Tests for the validation types.

Tests cover:
- TypedDict structure verification
- Type compatibility checks
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.validation import (
    CVResult,
    CVSplit,
    CVSplitInfo,
    FoldResult,
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
# Test: FoldResult
# =============================================================================


class TestFoldResult:
    """Tests for FoldResult TypedDict."""

    def test_can_create_fold_result(self) -> None:
        """Can create a FoldResult with all required fields."""
        val_indices = _make_intp_array((0, 1, 2))
        val_predictions = _make_float64_array((0.1, 0.5, 0.9))

        result = FoldResult(
            fold_number=0,
            train_auc=0.85,
            val_auc=0.82,
            val_indices=val_indices,
            val_predictions=val_predictions,
        )

        assert result["fold_number"] == 0
        assert result["train_auc"] == 0.85
        assert result["val_auc"] == 0.82
        np.testing.assert_array_equal(result["val_indices"], val_indices)
        np.testing.assert_array_equal(result["val_predictions"], val_predictions)

    def test_fold_result_supports_dict_access(self) -> None:
        """FoldResult supports dict-style key access."""
        val_indices = _make_intp_array((0,))
        val_predictions = _make_float64_array((0.5,))

        result = FoldResult(
            fold_number=0,
            train_auc=0.8,
            val_auc=0.75,
            val_indices=val_indices,
            val_predictions=val_predictions,
        )

        # Verify dict-style access works for all keys
        keys = list(result.keys())
        assert "fold_number" in keys
        assert "train_auc" in keys
        assert "val_auc" in keys
        assert "val_indices" in keys
        assert "val_predictions" in keys


# =============================================================================
# Test: CVResult
# =============================================================================


class TestCVResult:
    """Tests for CVResult TypedDict."""

    def test_can_create_cv_result(self) -> None:
        """Can create a CVResult with all required fields."""
        val_indices = _make_intp_array((0, 1))
        val_predictions = _make_float64_array((0.3, 0.7))

        fold_result = FoldResult(
            fold_number=0,
            train_auc=0.9,
            val_auc=0.85,
            val_indices=val_indices,
            val_predictions=val_predictions,
        )

        oof = _make_float64_array((0.3, 0.7))

        cv_result = CVResult(
            n_folds=1,
            fold_results=(fold_result,),
            mean_val_auc=0.85,
            std_val_auc=0.0,
            oof_predictions=oof,
        )

        assert cv_result["n_folds"] == 1
        assert len(cv_result["fold_results"]) == 1
        assert cv_result["mean_val_auc"] == 0.85
        assert cv_result["std_val_auc"] == 0.0

    def test_cv_result_with_multiple_folds(self) -> None:
        """Can create CVResult with multiple folds."""
        folds: list[FoldResult] = []
        rng = np.random.default_rng(42)
        for i in range(5):
            val_indices_list = tuple(i * 10 + j for j in range(10))
            val_indices = _make_intp_array(val_indices_list)
            rand_vals: NDArray[np.float64] = rng.random(10).astype(np.float64)
            folds.append(
                FoldResult(
                    fold_number=i,
                    train_auc=0.9,
                    val_auc=0.8 + i * 0.01,
                    val_indices=val_indices,
                    val_predictions=rand_vals,
                )
            )

        oof: NDArray[np.float64] = rng.random(50).astype(np.float64)

        cv_result = CVResult(
            n_folds=5,
            fold_results=tuple(folds),
            mean_val_auc=0.82,
            std_val_auc=0.014,
            oof_predictions=oof,
        )

        assert cv_result["n_folds"] == 5
        assert len(cv_result["fold_results"]) == 5


# =============================================================================
# Test: CVSplit
# =============================================================================


class TestCVSplit:
    """Tests for CVSplit TypedDict."""

    def test_can_create_cv_split(self) -> None:
        """Can create a CVSplit with all required fields."""
        train_indices = _make_intp_array((0, 1, 2, 3))
        val_indices = _make_intp_array((4, 5))

        split = CVSplit(
            fold_number=0,
            train_indices=train_indices,
            val_indices=val_indices,
        )

        assert split["fold_number"] == 0
        np.testing.assert_array_equal(split["train_indices"], train_indices)
        np.testing.assert_array_equal(split["val_indices"], val_indices)


# =============================================================================
# Test: CVSplitInfo
# =============================================================================


class TestCVSplitInfo:
    """Tests for CVSplitInfo TypedDict."""

    def test_can_create_cv_split_info(self) -> None:
        """Can create a CVSplitInfo with all required fields."""
        split = CVSplit(
            fold_number=0,
            train_indices=_make_intp_array((0, 1, 2)),
            val_indices=_make_intp_array((3, 4)),
        )

        info = CVSplitInfo(
            n_folds=1,
            n_samples=5,
            folds=(split,),
        )

        assert info["n_folds"] == 1
        assert info["n_samples"] == 5
        assert len(info["folds"]) == 1

    def test_cv_split_info_with_multiple_folds(self) -> None:
        """Can create CVSplitInfo with multiple folds."""
        splits: list[CVSplit] = []
        for i in range(3):
            train_list = tuple(j for j in range(10) if j % 3 != i)
            val_list = tuple(j for j in range(10) if j % 3 == i)
            splits.append(
                CVSplit(
                    fold_number=i,
                    train_indices=_make_intp_array(train_list),
                    val_indices=_make_intp_array(val_list),
                )
            )

        info = CVSplitInfo(
            n_folds=3,
            n_samples=10,
            folds=tuple(splits),
        )

        assert info["n_folds"] == 3
        assert len(info["folds"]) == 3
