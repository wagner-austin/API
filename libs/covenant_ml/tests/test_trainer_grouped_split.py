"""Grouped stratified split: whole groups share a split, never rows.

Rows within one group are correlated (1,500 snapshots of one match), so a
row split would place near-duplicates of training rows in the test set and
score memorization as skill. These pin the property the grouped branch
exists for: no group ever straddles splits.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.trainer import stratified_split


def _grouped_data(
    n_groups: int, rows_per_group: int, positive_groups: int
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Build groups whose x column carries the group id for tracing."""
    n = n_groups * rows_per_group
    x = np.zeros((n, 2), dtype=np.float64)
    y = np.zeros(n, dtype=np.int64)
    groups = np.zeros(n, dtype=np.int64)
    for g in range(n_groups):
        start = g * rows_per_group
        x[start : start + rows_per_group, 0] = float(g)
        groups[start : start + rows_per_group] = g
        if g < positive_groups:
            y[start : start + rows_per_group] = 1
    return x, y, groups


def _group_ids(x_split: NDArray[np.float64]) -> set[int]:
    ids: set[int] = set()
    for i in range(len(x_split)):
        value: np.float64 = x_split[i, 0]
        ids.add(int(value))
    return ids


def _count_rows_of_group(x_split: NDArray[np.float64], group: int) -> int:
    count = 0
    for i in range(len(x_split)):
        value: np.float64 = x_split[i, 0]
        if int(value) == group:
            count += 1
    return count


def _count_labels(y_split: NDArray[np.int64], label: int) -> int:
    count = 0
    for i in range(len(y_split)):
        value: np.int64 = y_split[i]
        if int(value) == label:
            count += 1
    return count


def test_no_group_straddles_two_splits() -> None:
    x, y, groups = _grouped_data(n_groups=10, rows_per_group=20, positive_groups=5)
    splits = stratified_split(x, y, 0.6, 0.2, 0.2, random_state=7, groups=groups)
    train_groups = _group_ids(splits.x_train)
    val_groups = _group_ids(splits.x_val)
    test_groups = _group_ids(splits.x_test)
    assert train_groups & val_groups == set()
    assert train_groups & test_groups == set()
    assert val_groups & test_groups == set()
    assert train_groups | val_groups | test_groups == set(range(10))


def test_every_row_follows_its_group_intact() -> None:
    """A group contributes all of its rows to exactly one split."""
    x, y, groups = _grouped_data(n_groups=6, rows_per_group=15, positive_groups=3)
    splits = stratified_split(x, y, 0.5, 0.25, 0.25, random_state=3, groups=groups)
    for part in (splits.x_train, splits.x_val, splits.x_test):
        for g in _group_ids(part):
            assert _count_rows_of_group(part, g) == 15
    total = len(splits.x_train) + len(splits.x_val) + len(splits.x_test)
    assert total == 90


def test_stratification_is_by_group_label() -> None:
    """Half the groups are positive, so each split holds both classes."""
    x, y, groups = _grouped_data(n_groups=10, rows_per_group=10, positive_groups=5)
    splits = stratified_split(x, y, 0.6, 0.2, 0.2, random_state=11, groups=groups)
    assert _count_labels(splits.y_train, 1) == 30
    assert _count_labels(splits.y_train, 0) == 30
    assert _count_labels(splits.y_val, 1) == 10
    assert _count_labels(splits.y_test, 1) == 10


def test_same_seed_reproduces_the_same_grouped_split() -> None:
    x, y, groups = _grouped_data(n_groups=8, rows_per_group=5, positive_groups=4)
    first = stratified_split(x, y, 0.7, 0.15, 0.15, random_state=42, groups=groups)
    second = stratified_split(x, y, 0.7, 0.15, 0.15, random_state=42, groups=groups)
    assert np.array_equal(first.x_train, second.x_train)
    assert np.array_equal(first.x_val, second.x_val)
    assert np.array_equal(first.x_test, second.x_test)


def test_rows_within_a_split_are_shuffled_not_blockwise() -> None:
    """Downstream batch-wise consumers must not see one group contiguous."""
    x, y, groups = _grouped_data(n_groups=10, rows_per_group=50, positive_groups=5)
    splits = stratified_split(x, y, 0.8, 0.1, 0.1, random_state=1, groups=groups)
    changes = 0
    for i in range(1, len(splits.x_train)):
        here: np.float64 = splits.x_train[i, 0]
        previous: np.float64 = splits.x_train[i - 1, 0]
        if int(here) != int(previous):
            changes += 1
    # Blockwise ordering would change value only at group boundaries
    # (n_groups - 1 times); a shuffle changes it far more often.
    assert changes > 50
