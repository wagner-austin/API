"""Tests for CV testing utilities.

Tests cover:
- FakeCVSplitter
- Factory functions
- Test registry creation
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.validation.testing import (
    FakeCVSplitter,
    make_binary_labels,
    make_fake_cv_splitter,
    make_group_ids,
    make_test_cv_registry,
    make_test_cv_split_info,
)

# =============================================================================
# Test Helpers
# =============================================================================


def _make_labels(n_samples: int) -> NDArray[np.int64]:
    """Create binary label array."""
    half = n_samples // 2
    pos: NDArray[np.int64] = np.ones(half, dtype=np.int64)
    neg: NDArray[np.int64] = np.zeros(n_samples - half, dtype=np.int64)
    result: NDArray[np.int64] = np.concatenate([pos, neg])
    return result


# =============================================================================
# FakeCVSplitter Tests
# =============================================================================


class TestFakeCVSplitter:
    """Tests for FakeCVSplitter."""

    def test_default_strategy_name(self) -> None:
        """Default strategy name is stratified_kfold."""
        splitter = FakeCVSplitter()
        assert splitter.strategy_name() == "stratified_kfold"

    def test_custom_strategy_name(self) -> None:
        """Can set custom strategy name using valid Literal."""
        splitter = FakeCVSplitter(name="time_series")
        assert splitter.strategy_name() == "time_series"

    def test_default_capabilities(self) -> None:
        """Default capabilities are correct."""
        splitter = FakeCVSplitter()
        caps = splitter.capabilities()

        assert caps["preserves_class_ratio"] is True
        assert caps["supports_groups"] is False
        assert caps["supports_temporal"] is False
        assert caps["supports_shuffle"] is True

    def test_custom_capabilities(self) -> None:
        """Can set custom capabilities."""
        from covenant_ml.validation.protocol import CVStrategyCapabilities

        custom_caps = CVStrategyCapabilities(
            preserves_class_ratio=False,
            supports_groups=True,
            supports_temporal=True,
            supports_shuffle=False,
        )
        splitter = FakeCVSplitter(capabilities=custom_caps)
        caps = splitter.capabilities()

        assert caps["preserves_class_ratio"] is False
        assert caps["supports_groups"] is True

    def test_split_returns_generated_folds(self) -> None:
        """Split returns generated fake folds."""
        splitter = FakeCVSplitter()
        y = _make_labels(100)

        split_info = splitter.split(y, n_folds=5, random_state=42)

        assert split_info["n_folds"] == 5
        assert split_info["n_samples"] == 100
        assert len(split_info["folds"]) == 5

    def test_custom_splits(self) -> None:
        """Can provide custom splits."""
        from covenant_ml.validation.types import CVSplit

        train_list: list[int] = [0, 1, 2]
        val_list: list[int] = [3, 4]
        train_idx: NDArray[np.intp] = np.array(train_list, dtype=np.intp)
        val_idx: NDArray[np.intp] = np.array(val_list, dtype=np.intp)

        custom_fold = CVSplit(
            fold_number=0,
            train_indices=train_idx,
            val_indices=val_idx,
        )

        splitter = FakeCVSplitter(splits=(custom_fold,))
        y = _make_labels(5)

        split_info = splitter.split(y, n_folds=3, random_state=42)

        assert split_info["n_folds"] == 1  # Uses custom splits
        assert split_info["folds"][0]["fold_number"] == 0

    def test_split_call_count(self) -> None:
        """Tracks number of split calls."""
        splitter = FakeCVSplitter()
        y = _make_labels(100)

        assert splitter.split_call_count == 0

        splitter.split(y, n_folds=5, random_state=42)
        assert splitter.split_call_count == 1

        splitter.split(y, n_folds=3, random_state=42)
        assert splitter.split_call_count == 2


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestMakeFakeCVSplitter:
    """Tests for make_fake_cv_splitter factory."""

    def test_default_factory(self) -> None:
        """Factory creates splitter with defaults."""
        splitter = make_fake_cv_splitter()
        assert splitter.strategy_name() == "stratified_kfold"
        caps = splitter.capabilities()
        assert caps["preserves_class_ratio"] is True

    def test_factory_with_custom_name(self) -> None:
        """Factory creates splitter with custom name."""
        splitter = make_fake_cv_splitter(name="time_series")
        assert splitter.strategy_name() == "time_series"

    def test_factory_creates_working_splitter(self) -> None:
        """Factory creates splitter that works with split."""
        splitter = make_fake_cv_splitter()
        y = _make_labels(100)
        split_info = splitter.split(y, n_folds=5, random_state=42)
        assert split_info["n_folds"] == 5


# =============================================================================
# Test Registry Tests
# =============================================================================


class TestMakeTestCVRegistry:
    """Tests for make_test_cv_registry factory."""

    def test_registry_has_all_strategies(self) -> None:
        """Test registry has expected fake strategies."""
        registry = make_test_cv_registry()
        strategies = registry.list_strategies()

        assert "stratified_kfold" in strategies
        assert "group_stratified_kfold" in strategies
        assert "shuffle_split" in strategies
        assert "time_series" in strategies

    def test_strategies_are_fake(self) -> None:
        """All strategies are FakeCVSplitter instances."""
        registry = make_test_cv_registry()

        splitter = registry.get("stratified_kfold")
        assert splitter.strategy_name() == "stratified_kfold"

        splitter2 = registry.get("time_series")
        assert splitter2.strategy_name() == "time_series"

    def test_strategies_work(self) -> None:
        """Fake strategies produce valid results."""
        registry = make_test_cv_registry()
        splitter = registry.get("stratified_kfold")

        y = _make_labels(100)
        split_info = splitter.split(y, n_folds=5, random_state=42)

        assert split_info["n_folds"] == 5
        assert split_info["folds"][0]["fold_number"] == 0

    def test_group_stratified_strategy(self) -> None:
        """Group stratified strategy is accessible and works."""
        registry = make_test_cv_registry()
        splitter = registry.get("group_stratified_kfold")

        assert splitter.strategy_name() == "group_stratified_kfold"
        caps = splitter.capabilities()
        assert caps["supports_groups"] is True

    def test_shuffle_split_strategy(self) -> None:
        """Shuffle split strategy is accessible and works."""
        registry = make_test_cv_registry()
        splitter = registry.get("shuffle_split")

        assert splitter.strategy_name() == "shuffle_split"
        caps = splitter.capabilities()
        assert caps["preserves_class_ratio"] is True


# =============================================================================
# make_test_cv_split_info Tests
# =============================================================================


class TestMakeTestCVSplitInfo:
    """Tests for make_test_cv_split_info factory."""

    def test_default_split_info(self) -> None:
        """Factory creates split info with defaults."""
        split_info = make_test_cv_split_info()

        assert split_info["n_samples"] == 100
        assert split_info["n_folds"] == 3
        assert len(split_info["folds"]) == 3

    def test_custom_n_samples(self) -> None:
        """Factory creates split info with custom n_samples."""
        split_info = make_test_cv_split_info(n_samples=200)
        assert split_info["n_samples"] == 200

    def test_custom_n_folds(self) -> None:
        """Factory creates split info with custom n_folds."""
        split_info = make_test_cv_split_info(n_folds=5)
        assert split_info["n_folds"] == 5
        assert len(split_info["folds"]) == 5

    def test_folds_are_non_overlapping(self) -> None:
        """Validation sets are non-overlapping."""
        split_info = make_test_cv_split_info(n_samples=100, n_folds=5)

        all_val_indices: set[int] = set()
        for fold in split_info["folds"]:
            val_indices: NDArray[np.intp] = fold["val_indices"]
            fold_val: set[int] = {np.intp(idx).item() for idx in val_indices.flat}
            # No overlap with previous folds
            assert len(all_val_indices & fold_val) == 0
            all_val_indices.update(fold_val)

    def test_train_and_val_are_disjoint(self) -> None:
        """Train and validation sets are disjoint for each fold."""
        split_info = make_test_cv_split_info(n_samples=100, n_folds=3)

        for fold in split_info["folds"]:
            train_indices: NDArray[np.intp] = fold["train_indices"]
            val_indices: NDArray[np.intp] = fold["val_indices"]
            train_set: set[int] = {np.intp(idx).item() for idx in train_indices.flat}
            val_set: set[int] = {np.intp(idx).item() for idx in val_indices.flat}
            assert len(train_set & val_set) == 0

    def test_fold_numbers_are_correct(self) -> None:
        """Fold numbers are sequential starting from 0."""
        split_info = make_test_cv_split_info(n_folds=4)

        for i, fold in enumerate(split_info["folds"]):
            assert fold["fold_number"] == i


# =============================================================================
# make_binary_labels Tests
# =============================================================================


class TestMakeBinaryLabels:
    """Tests for make_binary_labels factory."""

    def test_default_labels(self) -> None:
        """Factory creates labels with defaults."""
        labels = make_binary_labels()

        assert len(labels) == 100
        assert labels.dtype == np.int64

    def test_custom_n_samples(self) -> None:
        """Factory creates labels with custom n_samples."""
        labels = make_binary_labels(n_samples=50)
        assert len(labels) == 50

    def test_custom_positive_ratio(self) -> None:
        """Factory creates labels with custom positive ratio."""
        labels = make_binary_labels(n_samples=100, positive_ratio=0.5)
        mask: NDArray[np.bool_] = labels == 1
        positive_count: int = int(np.count_nonzero(mask))
        # Should be approximately 50%
        assert 45 <= positive_count <= 55

    def test_labels_are_binary(self) -> None:
        """Labels contain only 0 and 1."""
        labels = make_binary_labels(n_samples=100)
        unique_values: NDArray[np.int64] = np.unique(labels)
        assert len(unique_values) <= 2
        for val in unique_values.flat:
            val_int: int = np.int64(val).item()
            assert val_int in [0, 1]

    def test_reproducibility(self) -> None:
        """Same random state produces same labels."""
        labels1 = make_binary_labels(n_samples=100, random_state=42)
        labels2 = make_binary_labels(n_samples=100, random_state=42)
        assert np.array_equal(labels1, labels2)

    def test_different_random_states(self) -> None:
        """Different random states produce different labels."""
        labels1 = make_binary_labels(n_samples=100, random_state=42)
        labels2 = make_binary_labels(n_samples=100, random_state=123)
        assert not np.array_equal(labels1, labels2)


# =============================================================================
# make_group_ids Tests
# =============================================================================


class TestMakeGroupIds:
    """Tests for make_group_ids factory."""

    def test_default_groups(self) -> None:
        """Factory creates groups with defaults."""
        groups = make_group_ids()

        assert len(groups) == 100
        assert groups.dtype == np.int64

    def test_custom_n_samples(self) -> None:
        """Factory creates groups with custom n_samples."""
        groups = make_group_ids(n_samples=50)
        assert len(groups) == 50

    def test_custom_n_groups(self) -> None:
        """Factory creates groups with custom n_groups."""
        groups = make_group_ids(n_samples=100, n_groups=20)
        unique_groups = np.unique(groups)
        assert len(unique_groups) == 20

    def test_round_robin_assignment(self) -> None:
        """Groups are assigned in round-robin fashion."""
        groups = make_group_ids(n_samples=15, n_groups=5)
        groups_list: list[int] = [np.int64(g).item() for g in groups.flat]

        # First 5 samples should be groups 0-4
        for i in range(5):
            assert groups_list[i] == i

        # Next 5 samples should be groups 0-4 again
        for i in range(5, 10):
            assert groups_list[i] == i - 5

    def test_all_groups_present(self) -> None:
        """All group IDs from 0 to n_groups-1 are present."""
        groups = make_group_ids(n_samples=100, n_groups=10)
        unique_groups: set[int] = {np.int64(val).item() for val in groups.flat}
        expected_groups = set(range(10))
        assert unique_groups == expected_groups
