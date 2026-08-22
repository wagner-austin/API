"""Tests for the cross-validation splitter.

Tests cover:
- Stratified k-fold split creation
- Group-stratified k-fold split creation
- Class proportion preservation
- Fold data extraction
- Edge cases and error conditions
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.validation import (
    get_fold_data,
    stratified_kfold_split,
)
from tests.validation._splitter_fixtures import (
    _check_all_ones,
    _count_class,
    _get_unique_indices,
    _indices_to_set,
    _make_features,
    _make_intp_array,
    _make_labels,
)


class TestStratifiedKfoldSplit:
    """Tests for stratified_kfold_split function."""

    def test_creates_correct_number_of_folds(self) -> None:
        """Creates exactly n_folds splits."""
        y = _make_labels(50, 50)
        split_info = stratified_kfold_split(y, n_folds=5, random_state=42)

        assert split_info["n_folds"] == 5
        assert len(split_info["folds"]) == 5

    def test_folds_have_correct_structure(self) -> None:
        """Each fold has required fields."""
        y = _make_labels(50, 50)
        split_info = stratified_kfold_split(y, n_folds=3, random_state=42)

        for fold in split_info["folds"]:
            assert "fold_number" in fold
            assert "train_indices" in fold
            assert "val_indices" in fold

    def test_all_samples_appear_in_exactly_one_val_fold(self) -> None:
        """Each sample appears in exactly one validation fold."""
        n_samples = 100
        y = _make_labels(30, 70)
        split_info = stratified_kfold_split(y, n_folds=5, random_state=42)

        # Collect all validation indices
        val_counts: NDArray[np.int64] = np.zeros(n_samples, dtype=np.int64)
        for fold in split_info["folds"]:
            val_indices = fold["val_indices"]
            for i in range(len(val_indices)):
                idx = int(val_indices.item(i))
                current = int(val_counts.item(idx))
                val_counts[idx] = current + 1

        # Each sample should appear exactly once
        assert _check_all_ones(val_counts)

    def test_train_and_val_are_disjoint(self) -> None:
        """Train and validation sets within each fold do not overlap."""
        y = _make_labels(50, 50)
        split_info = stratified_kfold_split(y, n_folds=5, random_state=42)

        for fold in split_info["folds"]:
            train_set = _indices_to_set(fold["train_indices"])
            val_set = _indices_to_set(fold["val_indices"])
            intersection = train_set & val_set
            assert len(intersection) == 0

    def test_train_and_val_cover_all_samples(self) -> None:
        """Train + validation covers all samples in each fold."""
        n_samples = 100
        y = _make_labels(30, 70)
        split_info = stratified_kfold_split(y, n_folds=5, random_state=42)

        for fold in split_info["folds"]:
            train_set = _indices_to_set(fold["train_indices"])
            val_set = _indices_to_set(fold["val_indices"])
            combined = train_set | val_set
            assert len(combined) == n_samples

    def test_maintains_stratification(self) -> None:
        """Class proportions are approximately maintained in each fold."""
        y = _make_labels(100, 400)  # 20% positive
        split_info = stratified_kfold_split(y, n_folds=5, random_state=42)

        overall_ratio = 100 / 500  # 0.2

        for fold in split_info["folds"]:
            val_indices = fold["val_indices"]
            y_val = y[val_indices]
            n_pos = _count_class(y_val, 1)
            fold_ratio = n_pos / len(val_indices)

            # Should be within 10% relative of overall ratio
            assert abs(fold_ratio - overall_ratio) < 0.1 * overall_ratio + 0.05

    def test_reproducibility_with_same_seed(self) -> None:
        """Same random_state produces identical splits."""
        y = _make_labels(50, 50)
        split1 = stratified_kfold_split(y, n_folds=5, random_state=42)
        split2 = stratified_kfold_split(y, n_folds=5, random_state=42)

        for fold1, fold2 in zip(split1["folds"], split2["folds"], strict=True):
            np.testing.assert_array_equal(fold1["train_indices"], fold2["train_indices"])
            np.testing.assert_array_equal(fold1["val_indices"], fold2["val_indices"])

    def test_different_seeds_produce_different_splits(self) -> None:
        """Different random_state produces different splits."""
        y = _make_labels(50, 50)
        split1 = stratified_kfold_split(y, n_folds=5, random_state=42)
        split2 = stratified_kfold_split(y, n_folds=5, random_state=123)

        # At least one fold should be different
        any_different = False
        for fold1, fold2 in zip(split1["folds"], split2["folds"], strict=True):
            if not np.array_equal(fold1["train_indices"], fold2["train_indices"]):
                any_different = True
                break

        assert any_different

    def test_raises_on_less_than_2_folds(self) -> None:
        """Raises ValueError if n_folds < 2."""
        y = _make_labels(50, 50)

        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            stratified_kfold_split(y, n_folds=1, random_state=42)

    def test_raises_on_insufficient_positive_samples(self) -> None:
        """Raises ValueError if fewer positives than folds."""
        y = _make_labels(2, 100)  # Only 2 positives for 5 folds

        with pytest.raises(ValueError, match="Not enough positive samples"):
            stratified_kfold_split(y, n_folds=5, random_state=42)

    def test_raises_on_insufficient_negative_samples(self) -> None:
        """Raises ValueError if fewer negatives than folds."""
        y = _make_labels(100, 2)  # Only 2 negatives for 5 folds

        with pytest.raises(ValueError, match="Not enough negative samples"):
            stratified_kfold_split(y, n_folds=5, random_state=42)

    def test_minimum_viable_split(self) -> None:
        """Works with minimum viable sample counts."""
        y = _make_labels(5, 5)  # 5 of each class for 5 folds
        split_info = stratified_kfold_split(y, n_folds=5, random_state=42)

        # Each fold should have exactly 2 validation samples (1 pos, 1 neg)
        for fold in split_info["folds"]:
            assert len(fold["val_indices"]) == 2

    def test_records_n_samples(self) -> None:
        """Split info records total sample count."""
        y = _make_labels(30, 70)
        split_info = stratified_kfold_split(y, n_folds=5, random_state=42)

        assert split_info["n_samples"] == 100


class TestGetFoldData:
    """Tests for get_fold_data function."""

    def test_extracts_correct_data(self) -> None:
        """Extracts train and val data matching indices."""
        y = _make_labels(10, 10)
        x = _make_features(20, 5)
        split_info = stratified_kfold_split(y, n_folds=2, random_state=42)

        fold = split_info["folds"][0]
        x_train, y_train, x_val, y_val = get_fold_data(x, y, fold)

        # Check shapes using shape tuple elements
        n_train = int(x_train.shape[0])
        n_val = int(x_val.shape[0])
        n_train_expected = len(fold["train_indices"])
        n_val_expected = len(fold["val_indices"])

        assert len(y_train) == n_train_expected
        assert len(y_val) == n_val_expected
        assert n_train == len(y_train)
        assert n_val == len(y_val)

        # Check feature dimensions preserved
        assert int(x_train.shape[1]) == 5
        assert int(x_val.shape[1]) == 5

    def test_preserves_label_correspondence(self) -> None:
        """Labels correspond to correct samples."""
        y = _make_labels(10, 10)
        x = _make_features(20, 5)
        split_info = stratified_kfold_split(y, n_folds=2, random_state=42)

        fold = split_info["folds"][0]
        _x_train, y_train, _x_val, y_val = get_fold_data(x, y, fold)

        # Verify against original labels
        train_indices = fold["train_indices"]
        for i in range(len(train_indices)):
            idx = int(train_indices.item(i))
            assert int(y_train.item(i)) == int(y.item(idx))

        val_indices = fold["val_indices"]
        for i in range(len(val_indices)):
            idx = int(val_indices.item(i))
            assert int(y_val.item(i)) == int(y.item(idx))

    def test_preserves_feature_correspondence(self) -> None:
        """Features correspond to correct samples."""
        y = _make_labels(10, 10)
        x = _make_features(20, 5, seed=123)
        split_info = stratified_kfold_split(y, n_folds=2, random_state=42)

        fold = split_info["folds"][0]
        x_train, _y_train, x_val, _y_val = get_fold_data(x, y, fold)

        # Verify against original features using row-by-row comparison
        train_indices = fold["train_indices"]
        for i in range(len(train_indices)):
            idx = int(train_indices.item(i))
            np.testing.assert_array_equal(x_train[i, :], x[idx, :])

        val_indices = fold["val_indices"]
        for i in range(len(val_indices)):
            idx = int(val_indices.item(i))
            np.testing.assert_array_equal(x_val[i, :], x[idx, :])


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_two_fold_split(self) -> None:
        """Two-fold split works correctly (50/50 split)."""
        y = _make_labels(50, 50)
        split_info = stratified_kfold_split(y, n_folds=2, random_state=42)

        # Each fold should have ~50 validation samples
        for fold in split_info["folds"]:
            assert 45 <= len(fold["val_indices"]) <= 55

    def test_many_folds(self) -> None:
        """Works with many folds (leave-one-out like)."""
        y = _make_labels(50, 50)
        split_info = stratified_kfold_split(y, n_folds=10, random_state=42)

        assert len(split_info["folds"]) == 10

        # Each fold should have ~10 validation samples
        for fold in split_info["folds"]:
            assert 5 <= len(fold["val_indices"]) <= 15

    def test_imbalanced_classes(self) -> None:
        """Works with highly imbalanced classes."""
        y = _make_labels(10, 190)  # 5% positive
        split_info = stratified_kfold_split(y, n_folds=5, random_state=42)

        # Should still stratify correctly
        all_val_indices = _get_unique_indices(split_info)
        assert len(all_val_indices) == 200


class TestConcatIndices:
    """Tests for _concat_indices internal function."""

    def test_empty_args_returns_empty_array(self) -> None:
        """No arrays returns empty array."""
        from covenant_ml.validation.splitter import _concat_indices

        result = _concat_indices()
        assert len(result) == 0
        assert result.dtype == np.intp

    def test_single_array_returns_copy(self) -> None:
        """Single array returns that array's contents."""
        from covenant_ml.validation.splitter import _concat_indices

        arr = _make_intp_array((1, 2, 3))
        result = _concat_indices(arr)
        np.testing.assert_array_equal(result, arr)

    def test_multiple_arrays_concatenates(self) -> None:
        """Multiple arrays are concatenated."""
        from covenant_ml.validation.splitter import _concat_indices

        arr1 = _make_intp_array((1, 2))
        arr2 = _make_intp_array((3, 4))
        result = _concat_indices(arr1, arr2)
        expected = _make_intp_array((1, 2, 3, 4))
        np.testing.assert_array_equal(result, expected)
