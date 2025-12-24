"""Tests for CV strategy implementations.

Tests cover:
- StratifiedKFoldSplitter
- GroupStratifiedKFoldSplitter
- ShuffleSplitSplitter
- TimeSeriesSplitter
- Strategy protocol compliance
- Capabilities reporting
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.validation.strategies import (
    GroupStratifiedKFoldSplitter,
    ShuffleSplitSplitter,
    StratifiedKFoldSplitter,
    TimeSeriesSplitter,
    create_group_stratified_kfold_splitter,
    create_shuffle_split_splitter,
    create_stratified_kfold_splitter,
    create_time_series_splitter,
)

# =============================================================================
# Test Helpers
# =============================================================================


def _make_labels(n_pos: int, n_neg: int) -> NDArray[np.int64]:
    """Create binary label array with specified class counts."""
    pos: NDArray[np.int64] = np.ones(n_pos, dtype=np.int64)
    neg: NDArray[np.int64] = np.zeros(n_neg, dtype=np.int64)
    result: NDArray[np.int64] = np.concatenate([pos, neg])
    return result


def _make_groups(n_samples: int, n_groups: int) -> NDArray[np.int64]:
    """Create group array with roughly equal group sizes."""
    groups: NDArray[np.int64] = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        groups[i] = i % n_groups
    return groups


# =============================================================================
# StratifiedKFoldSplitter Tests
# =============================================================================


class TestStratifiedKFoldSplitter:
    """Tests for StratifiedKFoldSplitter."""

    def test_strategy_name(self) -> None:
        """Strategy name is correct."""
        splitter = StratifiedKFoldSplitter()
        assert splitter.strategy_name() == "stratified_kfold"

    def test_capabilities(self) -> None:
        """Capabilities are correctly reported."""
        splitter = StratifiedKFoldSplitter()
        caps = splitter.capabilities()

        assert caps["preserves_class_ratio"] is True
        assert caps["supports_groups"] is False
        assert caps["supports_temporal"] is False
        assert caps["supports_shuffle"] is True

    def test_split_creates_correct_number_of_folds(self) -> None:
        """Split creates the requested number of folds."""
        splitter = StratifiedKFoldSplitter()
        y = _make_labels(50, 50)

        for n_folds in [3, 5, 10]:
            split_info = splitter.split(y, n_folds=n_folds, random_state=42)
            assert split_info["n_folds"] == n_folds
            assert len(split_info["folds"]) == n_folds

    def test_split_preserves_class_ratio(self) -> None:
        """Each fold roughly preserves the class ratio."""
        splitter = StratifiedKFoldSplitter()
        y = _make_labels(30, 70)
        split_info = splitter.split(y, n_folds=5, random_state=42)

        original_ratio = 30 / 100

        for fold in split_info["folds"]:
            val_indices: NDArray[np.intp] = fold["val_indices"]
            val_labels: NDArray[np.int64] = y[val_indices]
            mask: NDArray[np.bool_] = val_labels == 1
            pos_count: int = int(np.count_nonzero(mask))
            fold_ratio = pos_count / len(val_labels)
            assert abs(fold_ratio - original_ratio) < 0.1

    def test_factory_function(self) -> None:
        """Factory function creates correct splitter."""
        splitter = create_stratified_kfold_splitter()
        assert splitter.strategy_name() == "stratified_kfold"
        caps = splitter.capabilities()
        assert caps["preserves_class_ratio"] is True


# =============================================================================
# GroupStratifiedKFoldSplitter Tests
# =============================================================================


class TestGroupStratifiedKFoldSplitter:
    """Tests for GroupStratifiedKFoldSplitter."""

    def test_strategy_name(self) -> None:
        """Strategy name is correct."""
        splitter = GroupStratifiedKFoldSplitter()
        assert splitter.strategy_name() == "group_stratified_kfold"

    def test_capabilities(self) -> None:
        """Capabilities are correctly reported."""
        splitter = GroupStratifiedKFoldSplitter()
        caps = splitter.capabilities()

        assert caps["preserves_class_ratio"] is True
        assert caps["supports_groups"] is True
        assert caps["supports_temporal"] is False
        assert caps["supports_shuffle"] is True

    def test_split_with_groups(self) -> None:
        """Split with groups keeps groups together."""
        splitter = GroupStratifiedKFoldSplitter()
        # Create 20 groups: 10 positive groups and 10 negative groups
        # Each group has 5 samples with the same label
        n_groups = 20
        samples_per_group = 5

        # First 10 groups are positive, last 10 are negative
        y_list: list[int] = []
        groups_list: list[int] = []
        for g in range(n_groups):
            label = 1 if g < n_groups // 2 else 0
            for _ in range(samples_per_group):
                y_list.append(label)
                groups_list.append(g)

        y: NDArray[np.int64] = np.array(y_list, dtype=np.int64)
        groups: NDArray[np.int64] = np.array(groups_list, dtype=np.int64)

        split_info = splitter.split(y, n_folds=5, random_state=42, groups=groups)

        assert split_info["n_folds"] == 5

        # Verify groups are kept together
        for fold in split_info["folds"]:
            val_idx: NDArray[np.intp] = fold["val_indices"]
            train_idx: NDArray[np.intp] = fold["train_indices"]
            val_group_arr: NDArray[np.int64] = groups[val_idx]
            train_group_arr: NDArray[np.int64] = groups[train_idx]
            # Use np.unique to get unique group values then convert to Python set
            unique_val: NDArray[np.int64] = np.unique(val_group_arr)
            unique_train: NDArray[np.int64] = np.unique(train_group_arr)
            # Use flat iterator and .item() for type safety
            val_list: list[int] = [np.int64(v).item() for v in unique_val.flat]
            train_list: list[int] = [np.int64(t).item() for t in unique_train.flat]
            val_groups: set[int] = set(val_list)
            train_groups: set[int] = set(train_list)
            # No group should appear in both train and val
            assert len(val_groups & train_groups) == 0

    def test_factory_function(self) -> None:
        """Factory function creates correct splitter."""
        splitter = create_group_stratified_kfold_splitter()
        assert splitter.strategy_name() == "group_stratified_kfold"
        caps = splitter.capabilities()
        assert caps["supports_groups"] is True

    def test_error_when_groups_not_provided(self) -> None:
        """Raises error when groups parameter is None."""
        splitter = GroupStratifiedKFoldSplitter()
        y = _make_labels(50, 50)

        with pytest.raises(ValueError, match="groups parameter is required"):
            splitter.split(y, n_folds=5, random_state=42)


# =============================================================================
# ShuffleSplitSplitter Tests
# =============================================================================


class TestShuffleSplitSplitter:
    """Tests for ShuffleSplitSplitter."""

    def test_strategy_name(self) -> None:
        """Strategy name is correct."""
        splitter = ShuffleSplitSplitter()
        assert splitter.strategy_name() == "shuffle_split"

    def test_capabilities(self) -> None:
        """Capabilities are correctly reported."""
        splitter = ShuffleSplitSplitter()
        caps = splitter.capabilities()

        assert caps["preserves_class_ratio"] is True
        assert caps["supports_groups"] is False
        assert caps["supports_temporal"] is False
        assert caps["supports_shuffle"] is True

    def test_test_fraction_property(self) -> None:
        """Test fraction property returns correct value."""
        splitter = ShuffleSplitSplitter(test_fraction=0.3)
        assert splitter.test_fraction == 0.3

    def test_split_creates_correct_test_size(self) -> None:
        """Split creates approximately correct test size."""
        splitter = ShuffleSplitSplitter(test_fraction=0.2)
        y = _make_labels(50, 50)

        split_info = splitter.split(y, n_folds=5, random_state=42)

        for fold in split_info["folds"]:
            val_size = len(fold["val_indices"])
            # Should be approximately 20% of 100 samples
            assert 15 <= val_size <= 25

    def test_split_preserves_class_ratio(self) -> None:
        """Each fold preserves class ratio in validation set."""
        splitter = ShuffleSplitSplitter(test_fraction=0.2)
        y = _make_labels(30, 70)

        split_info = splitter.split(y, n_folds=5, random_state=42)

        for fold in split_info["folds"]:
            val_idx: NDArray[np.intp] = fold["val_indices"]
            val_labels: NDArray[np.int64] = y[val_idx]
            mask: NDArray[np.bool_] = val_labels == 1
            pos_count: int = int(np.count_nonzero(mask))
            pos_ratio = pos_count / len(val_labels)
            # Should be close to 30%
            assert 0.15 <= pos_ratio <= 0.45

    def test_factory_function(self) -> None:
        """Factory function creates correct splitter."""
        splitter = create_shuffle_split_splitter()
        assert splitter.strategy_name() == "shuffle_split"
        caps = splitter.capabilities()
        assert caps["supports_shuffle"] is True

    def test_error_on_too_few_samples(self) -> None:
        """Raises error when not enough samples for split."""
        splitter = ShuffleSplitSplitter(test_fraction=0.5)
        y = _make_labels(1, 1)  # Only 2 samples (1 each class)

        with pytest.raises(ValueError, match="Not enough"):
            splitter.split(y, n_folds=5, random_state=42)

    def test_error_on_invalid_test_fraction(self) -> None:
        """Raises error when test_fraction is not in (0, 1)."""
        with pytest.raises(ValueError, match="test_fraction must be in"):
            ShuffleSplitSplitter(test_fraction=1.5)

    def test_error_on_not_enough_negative_samples(self) -> None:
        """Raises error when not enough negative samples."""
        splitter = ShuffleSplitSplitter(test_fraction=0.5)
        # 50 positive, only 1 negative - not enough negatives
        y = _make_labels(50, 1)

        with pytest.raises(ValueError, match="Not enough negative samples"):
            splitter.split(y, n_folds=5, random_state=42)

    def test_error_on_zero_folds(self) -> None:
        """Raises error when n_folds is 0."""
        splitter = ShuffleSplitSplitter(test_fraction=0.2)
        y = _make_labels(50, 50)

        with pytest.raises(ValueError, match="n_folds must be >= 1"):
            splitter.split(y, n_folds=0, random_state=42)


class TestStratifiedShuffleSplitHelper:
    """Tests for _stratified_shuffle_split helper function."""

    def test_error_on_invalid_test_fraction(self) -> None:
        """Helper raises error for invalid test_fraction."""
        from covenant_ml.validation.strategies.shuffle_split import (
            _stratified_shuffle_split,
        )

        y = _make_labels(50, 50)
        rng = np.random.default_rng(42)

        with pytest.raises(ValueError, match="test_fraction must be in"):
            _stratified_shuffle_split(y, n_splits=3, test_fraction=1.5, rng=rng)


# =============================================================================
# TimeSeriesSplitter Tests
# =============================================================================


class TestTimeSeriesSplitter:
    """Tests for TimeSeriesSplitter."""

    def test_strategy_name(self) -> None:
        """Strategy name is correct."""
        splitter = TimeSeriesSplitter()
        assert splitter.strategy_name() == "time_series"

    def test_capabilities(self) -> None:
        """Capabilities are correctly reported."""
        splitter = TimeSeriesSplitter()
        caps = splitter.capabilities()

        assert caps["preserves_class_ratio"] is False
        assert caps["supports_groups"] is False
        assert caps["supports_temporal"] is True
        assert caps["supports_shuffle"] is False

    def test_min_train_size_property(self) -> None:
        """Min train size property returns correct value."""
        splitter = TimeSeriesSplitter(min_train_size=20)
        assert splitter.min_train_size == 20

        splitter2 = TimeSeriesSplitter()
        assert splitter2.min_train_size is None

    def test_split_respects_temporal_ordering(self) -> None:
        """Training indices always precede validation indices."""
        splitter = TimeSeriesSplitter()
        y = _make_labels(50, 50)

        split_info = splitter.split(y, n_folds=4, random_state=42)

        for fold in split_info["folds"]:
            train_idx: NDArray[np.intp] = fold["train_indices"]
            val_idx: NDArray[np.intp] = fold["val_indices"]
            max_train_idx: int = int(np.max(train_idx))
            min_val_idx: int = int(np.min(val_idx))
            assert max_train_idx < min_val_idx

    def test_split_has_expanding_window(self) -> None:
        """Each subsequent fold has larger training set."""
        splitter = TimeSeriesSplitter()
        y = _make_labels(50, 50)

        split_info = splitter.split(y, n_folds=4, random_state=42)

        prev_train_size = 0
        for fold in split_info["folds"]:
            train_size = len(fold["train_indices"])
            assert train_size >= prev_train_size
            prev_train_size = train_size

    def test_factory_function(self) -> None:
        """Factory function creates correct splitter."""
        splitter = create_time_series_splitter()
        assert splitter.strategy_name() == "time_series"
        caps = splitter.capabilities()
        assert caps["supports_temporal"] is True

    def test_error_on_invalid_n_splits(self) -> None:
        """Raises error for n_splits < 1."""
        splitter = TimeSeriesSplitter()
        y = _make_labels(50, 50)

        with pytest.raises(ValueError, match="n_splits must be >= 1"):
            splitter.split(y, n_folds=0, random_state=42)

    def test_error_on_too_few_samples(self) -> None:
        """Raises error when not enough samples."""
        splitter = TimeSeriesSplitter()
        y = _make_labels(2, 2)  # Only 4 samples

        with pytest.raises(ValueError, match="Not enough samples"):
            splitter.split(y, n_folds=5, random_state=42)

    def test_error_on_invalid_min_train_size(self) -> None:
        """Raises error for min_train_size < 1."""
        splitter = TimeSeriesSplitter(min_train_size=0)
        y = _make_labels(50, 50)

        with pytest.raises(ValueError, match="min_train_size must be >= 1"):
            splitter.split(y, n_folds=5, random_state=42)

    def test_error_on_min_train_size_too_large(self) -> None:
        """Raises error when min_train_size too large for data."""
        splitter = TimeSeriesSplitter(min_train_size=96)
        y = _make_labels(50, 50)

        with pytest.raises(ValueError, match="too large"):
            splitter.split(y, n_folds=5, random_state=42)

    def test_split_with_min_train_size(self) -> None:
        """Split respects min_train_size parameter."""
        min_train = 20
        splitter = TimeSeriesSplitter(min_train_size=min_train)
        y = _make_labels(50, 50)

        split_info = splitter.split(y, n_folds=4, random_state=42)

        # First fold should have at least min_train_size samples in training
        first_fold = split_info["folds"][0]
        assert len(first_fold["train_indices"]) >= min_train

    def test_split_with_many_folds_small_data(self) -> None:
        """Split handles edge case with many folds relative to data size."""
        splitter = TimeSeriesSplitter()
        # Small dataset with maximum folds to trigger edge cases
        y = _make_labels(5, 5)  # 10 samples

        split_info = splitter.split(y, n_folds=8, random_state=42)

        # All folds should have at least 1 sample in validation
        for fold in split_info["folds"]:
            val_size = len(fold["val_indices"])
            assert val_size > 0, f"Fold {fold['fold_number']} has no validation samples"
