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
    group_kfold_split,
    group_stratified_kfold_split,
)
from tests.validation._splitter_fixtures import (
    _check_all_ones,
    _count_class,
    _get_groups_for_indices,
    _get_unique_indices,
    _indices_to_set,
    _make_groups,
    _make_int64_array,
    _make_labels,
    _make_labels_for_groups,
)


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


class TestGroupKfoldSplit:
    """Tests for group_kfold_split — the mixed-label-group instrument."""

    def test_creates_correct_number_of_folds(self) -> None:
        """Creates exactly n_folds splits over mixed-label groups."""
        groups = _make_groups((3,) * 10)
        # Every group mixed: labels 1,0,1 within each 3-sample group.
        y = _make_int64_array(tuple([1, 0, 1] * 10))

        split_info = group_kfold_split(y, groups, n_folds=5, random_state=42)

        assert split_info["n_folds"] == 5
        assert len(split_info["folds"]) == 5
        assert split_info["n_samples"] == 30

    def test_groups_stay_together_and_partition_the_samples(self) -> None:
        """Whole groups per fold; every sample validates exactly once."""
        sizes = (2, 3, 4, 2, 3, 4, 2, 3, 4, 2)
        groups = _make_groups(sizes)
        y = _make_int64_array(tuple(i % 2 for i in range(sum(sizes))))

        split_info = group_kfold_split(y, groups, n_folds=5, random_state=42)

        seen_val: set[int] = set()
        for fold in split_info["folds"]:
            train_groups = _get_groups_for_indices(groups, fold["train_indices"])
            val_groups = _get_groups_for_indices(groups, fold["val_indices"])
            assert train_groups & val_groups == set()
            fold_val = _indices_to_set(fold["val_indices"])
            assert seen_val & fold_val == set()
            seen_val |= fold_val
        assert seen_val == set(range(sum(sizes)))

    def test_reproducibility_with_same_seed(self) -> None:
        """The same seed reproduces identical folds."""
        groups = _make_groups((3,) * 10)
        y = _make_int64_array(tuple([1, 0, 1] * 10))

        first = group_kfold_split(y, groups, n_folds=5, random_state=7)
        second = group_kfold_split(y, groups, n_folds=5, random_state=7)

        for fold_a, fold_b in zip(first["folds"], second["folds"], strict=True):
            np.testing.assert_array_equal(fold_a["train_indices"], fold_b["train_indices"])
            np.testing.assert_array_equal(fold_a["val_indices"], fold_b["val_indices"])

    def test_requires_matching_lengths(self) -> None:
        """A groups array of the wrong length is refused."""
        y = _make_int64_array((0, 1, 0, 1))
        groups = _make_int64_array((0, 0, 1))
        with pytest.raises(ValueError, match="groups length"):
            group_kfold_split(y, groups, n_folds=2, random_state=42)

    def test_requires_at_least_two_folds(self) -> None:
        """A single fold is not cross-validation."""
        y = _make_int64_array((0, 1, 0, 1))
        groups = _make_int64_array((0, 0, 1, 1))
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            group_kfold_split(y, groups, n_folds=1, random_state=42)

    def test_requires_enough_groups(self) -> None:
        """Fewer groups than folds is refused."""
        y = _make_int64_array((0, 1, 0, 1))
        groups = _make_int64_array((0, 0, 1, 1))
        with pytest.raises(ValueError, match="Not enough groups"):
            group_kfold_split(y, groups, n_folds=3, random_state=42)
