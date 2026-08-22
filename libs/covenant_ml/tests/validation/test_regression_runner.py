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

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.validation.regression_runner import (
    get_regression_fold_data,
    kfold_split,
)
from covenant_ml.validation.types import CVSplit
from tests.validation._regression_runner_fixtures import (
    _get_val,
    _make_intp_array,
)


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
