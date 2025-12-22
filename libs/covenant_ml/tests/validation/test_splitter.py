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
    CVSplitInfo,
    get_fold_data,
    group_stratified_kfold_split,
    stratified_kfold_split,
)

# =============================================================================
# Type-safe array helpers
# =============================================================================


def _make_intp_array(values: tuple[int, ...]) -> NDArray[np.intp]:
    """Create intp array from tuple of ints.

    Args:
        values: Tuple of integer values.

    Returns:
        Array of intp dtype.
    """
    result: NDArray[np.intp] = np.zeros(len(values), dtype=np.intp)
    for i, v in enumerate(values):
        result[i] = v
    return result


def _make_labels(n_pos: int, n_neg: int) -> NDArray[np.int64]:
    """Create binary label array with specified class counts.

    Args:
        n_pos: Number of positive samples (label=1).
        n_neg: Number of negative samples (label=0).

    Returns:
        Label array with n_pos ones followed by n_neg zeros.
    """
    pos: NDArray[np.int64] = np.ones(n_pos, dtype=np.int64)
    neg: NDArray[np.int64] = np.zeros(n_neg, dtype=np.int64)
    result: NDArray[np.int64] = np.concatenate([pos, neg])
    return result


def _make_features(n_samples: int, n_features: int, seed: int = 42) -> NDArray[np.float64]:
    """Create feature matrix with reproducible random values.

    Args:
        n_samples: Number of samples (rows).
        n_features: Number of features (columns).
        seed: Random seed for reproducibility.

    Returns:
        Feature matrix of shape (n_samples, n_features).
    """
    rng = np.random.default_rng(seed)
    result: NDArray[np.float64] = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    return result


def _count_class(y: NDArray[np.int64], class_value: int) -> int:
    """Count occurrences of a class value.

    Args:
        y: Label array.
        class_value: Class to count.

    Returns:
        Number of samples with given class value.
    """
    mask: NDArray[np.bool_] = y == class_value
    return int(np.sum(mask))


def _get_unique_indices(split_info: CVSplitInfo) -> set[int]:
    """Get all unique indices across all validation folds.

    Args:
        split_info: Complete split information.

    Returns:
        Set of all unique indices appearing in validation folds.
    """
    all_indices: set[int] = set()
    for fold in split_info["folds"]:
        val_indices = fold["val_indices"]
        for i in range(len(val_indices)):
            all_indices.add(int(val_indices.item(i)))
    return all_indices


def _indices_to_set(indices: NDArray[np.intp]) -> set[int]:
    """Convert index array to set of ints."""
    result: set[int] = set()
    for i in range(len(indices)):
        result.add(int(indices.item(i)))
    return result


def _check_all_ones(counts: NDArray[np.int64]) -> bool:
    """Check if all counts equal 1."""
    return all(int(counts.item(i)) == 1 for i in range(len(counts)))


# =============================================================================
# Test: stratified_kfold_split
# =============================================================================


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


# =============================================================================
# Test: get_fold_data
# =============================================================================


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


# =============================================================================
# Test: Edge Cases
# =============================================================================


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


# =============================================================================
# Test: Internal Helper Functions
# =============================================================================


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


# =============================================================================
# Group array helpers
# =============================================================================


def _make_groups(samples_per_group: tuple[int, ...]) -> NDArray[np.int64]:
    """Create group ID array with specified samples per group.

    Args:
        samples_per_group: Tuple of sample counts per group.
            E.g., (3, 2, 4) creates groups [0,0,0,1,1,2,2,2,2].

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
) -> NDArray[np.int64]:
    """Create label array where specified groups have at least one positive.

    Args:
        samples_per_group: Tuple of sample counts per group.
        positive_groups: Set of group IDs that should have positive samples.
            First sample in each positive group is set to 1.

    Returns:
        Binary label array.
    """
    n_samples = sum(samples_per_group)
    labels: NDArray[np.int64] = np.zeros(n_samples, dtype=np.int64)

    idx = 0
    for group_id, count in enumerate(samples_per_group):
        if group_id in positive_groups:
            # Set first sample in group to positive
            labels[idx] = 1
        idx += count

    return labels


def _get_groups_for_indices(
    groups: NDArray[np.int64],
    indices: NDArray[np.intp],
) -> set[int]:
    """Get unique group IDs for the given sample indices.

    Args:
        groups: Full group ID array.
        indices: Sample indices to look up.

    Returns:
        Set of unique group IDs.
    """
    result: set[int] = set()
    for i in range(len(indices)):
        idx = int(indices.item(i))
        group_id = int(groups.item(idx))
        result.add(group_id)
    return result


# =============================================================================
# Test: group_stratified_kfold_split
# =============================================================================


class TestGroupStratifiedKfoldSplit:
    """Tests for group_stratified_kfold_split function."""

    def test_creates_correct_number_of_folds(self) -> None:
        """Creates exactly n_folds splits."""
        # 20 groups, 3 samples each = 60 samples
        groups = _make_groups((3,) * 20)
        # 10 positive groups, 10 negative groups
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

        split_info = group_stratified_kfold_split(y, groups, n_folds=5, random_state=42)

        assert split_info["n_folds"] == 5
        assert len(split_info["folds"]) == 5

    def test_groups_stay_together(self) -> None:
        """All samples from a group appear in the same fold (train or val)."""
        # 20 groups with varying sizes
        groups = _make_groups((2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3))
        # 10 positive groups (first 10), 10 negative groups
        y = _make_labels_for_groups(
            (2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3),
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        )

        split_info = group_stratified_kfold_split(y, groups, n_folds=5, random_state=42)

        for fold in split_info["folds"]:
            train_groups = _get_groups_for_indices(groups, fold["train_indices"])
            val_groups = _get_groups_for_indices(groups, fold["val_indices"])

            # No group should appear in both train and val
            intersection = train_groups & val_groups
            assert len(intersection) == 0, f"Groups {intersection} appear in both train and val"

    def test_no_group_leakage_between_folds(self) -> None:
        """Each group appears in exactly one validation fold."""
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

        split_info = group_stratified_kfold_split(y, groups, n_folds=5, random_state=42)

        # Track which val fold each group appears in
        group_val_folds: dict[int, list[int]] = {}

        for fold_num, fold in enumerate(split_info["folds"]):
            val_groups = _get_groups_for_indices(groups, fold["val_indices"])
            for group_id in val_groups:
                if group_id not in group_val_folds:
                    group_val_folds[group_id] = []
                group_val_folds[group_id].append(fold_num)

        # Each group should appear in exactly one validation fold
        for group_id, fold_nums in group_val_folds.items():
            assert len(fold_nums) == 1, f"Group {group_id} appears in folds {fold_nums}"

    def test_all_samples_covered(self) -> None:
        """All samples appear in exactly one validation fold."""
        n_samples = 60
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

        split_info = group_stratified_kfold_split(y, groups, n_folds=5, random_state=42)

        # Count how many times each sample appears in validation
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
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

        split_info = group_stratified_kfold_split(y, groups, n_folds=5, random_state=42)

        for fold in split_info["folds"]:
            train_set = _indices_to_set(fold["train_indices"])
            val_set = _indices_to_set(fold["val_indices"])
            intersection = train_set & val_set
            assert len(intersection) == 0

    def test_train_and_val_cover_all_samples(self) -> None:
        """Train + validation covers all samples in each fold."""
        n_samples = 60
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

        split_info = group_stratified_kfold_split(y, groups, n_folds=5, random_state=42)

        for fold in split_info["folds"]:
            train_set = _indices_to_set(fold["train_indices"])
            val_set = _indices_to_set(fold["val_indices"])
            combined = train_set | val_set
            assert len(combined) == n_samples

    def test_stratification_by_group(self) -> None:
        """Positive and negative groups are distributed across folds."""
        # 10 positive groups, 40 negative groups
        groups = _make_groups((2,) * 50)
        y = _make_labels_for_groups((2,) * 50, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

        split_info = group_stratified_kfold_split(y, groups, n_folds=5, random_state=42)

        # Each fold should have some positive groups
        for fold in split_info["folds"]:
            val_indices = fold["val_indices"]
            y_val = y[val_indices]
            n_pos = _count_class(y_val, 1)
            # With 10 positive groups across 5 folds, expect ~2 positive groups per fold
            assert n_pos > 0, "Fold should have at least one positive sample"

    def test_reproducibility_with_same_seed(self) -> None:
        """Same random_state produces identical splits."""
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

        split1 = group_stratified_kfold_split(y, groups, n_folds=5, random_state=42)
        split2 = group_stratified_kfold_split(y, groups, n_folds=5, random_state=42)

        for fold1, fold2 in zip(split1["folds"], split2["folds"], strict=True):
            np.testing.assert_array_equal(fold1["train_indices"], fold2["train_indices"])
            np.testing.assert_array_equal(fold1["val_indices"], fold2["val_indices"])

    def test_different_seeds_produce_different_splits(self) -> None:
        """Different random_state produces different splits."""
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

        split1 = group_stratified_kfold_split(y, groups, n_folds=5, random_state=42)
        split2 = group_stratified_kfold_split(y, groups, n_folds=5, random_state=123)

        # At least one fold should be different
        any_different = False
        for fold1, fold2 in zip(split1["folds"], split2["folds"], strict=True):
            if not np.array_equal(fold1["train_indices"], fold2["train_indices"]):
                any_different = True
                break

        assert any_different

    def test_raises_on_less_than_2_folds(self) -> None:
        """Raises ValueError if n_folds < 2."""
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            group_stratified_kfold_split(y, groups, n_folds=1, random_state=42)

    def test_raises_on_groups_y_length_mismatch(self) -> None:
        """Raises ValueError if groups and y have different lengths."""
        groups = _make_groups((3,) * 10)  # 30 samples
        y = _make_labels(20, 20)  # 40 samples

        with pytest.raises(ValueError, match=r"groups length.*must match y length"):
            group_stratified_kfold_split(y, groups, n_folds=5, random_state=42)

    def test_raises_on_insufficient_groups(self) -> None:
        """Raises ValueError if fewer groups than folds."""
        groups = _make_groups((10, 10, 10))  # Only 3 groups
        y = _make_labels_for_groups((10, 10, 10), {0})

        with pytest.raises(ValueError, match="Not enough groups"):
            group_stratified_kfold_split(y, groups, n_folds=5, random_state=42)

    def test_raises_on_insufficient_positive_groups(self) -> None:
        """Raises ValueError if fewer positive groups than folds."""
        groups = _make_groups((3,) * 20)
        # Only 2 positive groups for 5 folds
        y = _make_labels_for_groups((3,) * 20, {0, 1})

        with pytest.raises(ValueError, match="Not enough positive groups"):
            group_stratified_kfold_split(y, groups, n_folds=5, random_state=42)

    def test_raises_on_insufficient_negative_groups(self) -> None:
        """Raises ValueError if fewer negative groups than folds."""
        groups = _make_groups((3,) * 20)
        # 18 positive groups, only 2 negative groups for 5 folds
        y = _make_labels_for_groups(
            (3,) * 20,
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
        )

        with pytest.raises(ValueError, match="Not enough negative groups"):
            group_stratified_kfold_split(y, groups, n_folds=5, random_state=42)

    def test_minimum_viable_split(self) -> None:
        """Works with minimum viable group counts."""
        # 5 positive groups + 5 negative groups = 10 groups for 5 folds
        groups = _make_groups((2,) * 10)
        y = _make_labels_for_groups((2,) * 10, {0, 1, 2, 3, 4})

        split_info = group_stratified_kfold_split(y, groups, n_folds=5, random_state=42)

        assert split_info["n_folds"] == 5
        # Each fold should have 2 groups (1 pos + 1 neg) = 4 samples
        for fold in split_info["folds"]:
            assert len(fold["val_indices"]) == 4

    def test_records_n_samples(self) -> None:
        """Split info records total sample count."""
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

        split_info = group_stratified_kfold_split(y, groups, n_folds=5, random_state=42)

        assert split_info["n_samples"] == 60

    def test_uneven_group_sizes(self) -> None:
        """Works with groups of varying sizes."""
        # Groups with sizes 1, 5, 10, 2, 8, 3, 7, 4, 6, 9 = 55 samples
        groups = _make_groups((1, 5, 10, 2, 8, 3, 7, 4, 6, 9))
        y = _make_labels_for_groups((1, 5, 10, 2, 8, 3, 7, 4, 6, 9), {0, 1, 2, 3, 4})

        split_info = group_stratified_kfold_split(y, groups, n_folds=5, random_state=42)

        # All samples should be covered
        all_val_indices = _get_unique_indices(split_info)
        assert len(all_val_indices) == 55

    def test_all_samples_in_group_go_to_same_fold_multisize(self) -> None:
        """Verify larger groups stay intact across train/val split."""
        # Create groups with different sizes - some quite large
        groups = _make_groups((10, 10, 10, 10, 10, 10, 10, 10, 10, 10))
        y = _make_labels_for_groups(
            (10, 10, 10, 10, 10, 10, 10, 10, 10, 10),
            {0, 1, 2, 3, 4},
        )

        split_info = group_stratified_kfold_split(y, groups, n_folds=5, random_state=42)

        # For each group, all its samples should be in the same split (train OR val)
        unique_groups: NDArray[np.int64] = np.unique(groups)
        n_unique = len(unique_groups)
        for idx in range(n_unique):
            group_id = int(unique_groups.item(idx))
            group_mask: NDArray[np.bool_] = groups == group_id
            flat_indices: NDArray[np.intp] = np.flatnonzero(group_mask)
            group_indices: set[int] = set()
            for j in range(len(flat_indices)):
                group_indices.add(int(flat_indices.item(j)))

            for fold in split_info["folds"]:
                train_set = _indices_to_set(fold["train_indices"])
                val_set = _indices_to_set(fold["val_indices"])

                in_train = group_indices & train_set
                in_val = group_indices & val_set

                # Group should be entirely in train OR entirely in val
                if len(in_train) > 0:
                    assert len(in_val) == 0
                    assert in_train == group_indices
                elif len(in_val) > 0:
                    assert len(in_train) == 0
                    assert in_val == group_indices


# =============================================================================
# Test: Internal Group Helper Functions
# =============================================================================


def _make_int64_array(values: tuple[int, ...]) -> NDArray[np.int64]:
    """Create int64 array from tuple of ints.

    Args:
        values: Tuple of integer values.

    Returns:
        Array of int64 dtype.
    """
    result: NDArray[np.int64] = np.zeros(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        result[i] = v
    return result


class TestGroupHelpers:
    """Tests for internal group helper functions."""

    def test_get_group_labels_all_negative(self) -> None:
        """All-negative groups get label 0."""
        from covenant_ml.validation.splitter import _get_group_labels

        groups = _make_int64_array((0, 0, 1, 1, 2, 2))
        y = _make_int64_array((0, 0, 0, 0, 0, 0))

        unique_groups, group_labels = _get_group_labels(groups, y)

        expected_groups = _make_int64_array((0, 1, 2))
        expected_labels = _make_int64_array((0, 0, 0))
        np.testing.assert_array_equal(unique_groups, expected_groups)
        np.testing.assert_array_equal(group_labels, expected_labels)

    def test_get_group_labels_mixed(self) -> None:
        """Groups with any positive sample get label 1."""
        from covenant_ml.validation.splitter import _get_group_labels

        groups = _make_int64_array((0, 0, 0, 1, 1, 1, 2, 2, 2))
        # Group 0: has positive, Group 1: all negative, Group 2: has positive
        y = _make_int64_array((1, 0, 0, 0, 0, 0, 0, 1, 0))

        unique_groups, group_labels = _get_group_labels(groups, y)

        expected_groups = _make_int64_array((0, 1, 2))
        expected_labels = _make_int64_array((1, 0, 1))
        np.testing.assert_array_equal(unique_groups, expected_groups)
        np.testing.assert_array_equal(group_labels, expected_labels)

    def test_get_sample_indices_for_groups(self) -> None:
        """Correctly retrieves all sample indices for selected groups."""
        from covenant_ml.validation.splitter import _get_sample_indices_for_groups

        groups = _make_int64_array((0, 0, 1, 1, 1, 2, 2))
        selected = _make_int64_array((0, 2))

        indices = _get_sample_indices_for_groups(groups, selected)

        # Group 0 has indices [0, 1], Group 2 has indices [5, 6]
        expected_set = {0, 1, 5, 6}
        result_set: set[int] = set()
        for i in range(len(indices)):
            result_set.add(int(indices.item(i)))
        assert result_set == expected_set

    def test_get_sample_indices_empty_selection(self) -> None:
        """Empty selection returns empty indices."""
        from covenant_ml.validation.splitter import _get_sample_indices_for_groups

        groups = _make_int64_array((0, 0, 1, 1))
        selected: NDArray[np.int64] = np.zeros(0, dtype=np.int64)

        indices = _get_sample_indices_for_groups(groups, selected)

        assert len(indices) == 0
