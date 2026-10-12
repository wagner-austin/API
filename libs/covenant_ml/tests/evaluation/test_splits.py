"""Tests for the grouped partitions."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.benchmarking.splitting import company_disjoint_masks
from covenant_ml.evaluation.splits import (
    ChronologicalGroupedSplits,
    CompanyDisjointGroupedSplits,
    chronological_masks,
)
from covenant_ml.evaluation.types import (
    ERR_INVALID_EMBARGO,
    ERR_INVALID_FRACTION,
    ERR_LENGTH_MISMATCH,
    ERR_TOO_FEW_UNITS,
)

from ._fixtures import at, floats, grouped_dataset, ints


def test_company_split_codes_each_held_out_row_by_its_company() -> None:
    features, labels, codes = grouped_dataset()
    grouped = CompanyDisjointGroupedSplits(features, labels, codes)(seed=4)
    masks = company_disjoint_masks(len(codes), codes, 4)
    expected: NDArray[np.int64] = codes[masks.test]
    assert np.array_equal(grouped.test_groups, expected)
    assert np.array_equal(grouped.split.x_test, features[masks.test])
    assert len(grouped.test_groups) == len(grouped.split.y_test)


def test_chronological_masks_cut_in_time_order_with_embargoes() -> None:
    masks = chronological_masks(20, 0.5, 0.25, 1)
    train_rows: NDArray[np.int64] = np.flatnonzero(masks.train).astype(np.int64)
    val_rows: NDArray[np.int64] = np.flatnonzero(masks.val).astype(np.int64)
    test_rows: NDArray[np.int64] = np.flatnonzero(masks.test).astype(np.int64)
    assert np.array_equal(train_rows, ints(list(range(10))))
    # Row 10 is embargoed after training and row 15 after validation.
    assert np.array_equal(val_rows, ints([11, 12, 13, 14]))
    assert np.array_equal(test_rows, ints([16, 17, 18, 19]))


@pytest.mark.parametrize(
    ("train", "val"),
    [(0.0, 0.2), (1.0, 0.2), (0.5, 0.0), (0.5, 1.0)],
)
def test_a_fraction_outside_the_open_unit_interval_is_refused(train: float, val: float) -> None:
    with pytest.raises(ValueError, match=ERR_INVALID_FRACTION):
        chronological_masks(20, train, val, 0)


def test_fractions_that_leave_no_test_share_are_refused() -> None:
    with pytest.raises(ValueError, match=ERR_INVALID_FRACTION):
        chronological_masks(20, 0.6, 0.4, 0)


def test_a_negative_embargo_is_refused() -> None:
    with pytest.raises(ValueError, match=ERR_INVALID_EMBARGO):
        chronological_masks(20, 0.5, 0.25, -1)


def test_an_embargo_that_empties_a_fold_is_refused() -> None:
    with pytest.raises(ValueError, match=ERR_TOO_FEW_UNITS):
        chronological_masks(20, 0.5, 0.25, 5)


def test_chronological_split_orders_rows_by_time_and_codes_them_by_time() -> None:
    # Rows arrive out of order; the feature equals the time key so the
    # reordering is visible in the split itself.
    keys = ints([5, 1, 9, 3, 7, 2, 8, 4, 6, 0])
    features: NDArray[np.float64] = keys.astype(np.float64).reshape(-1, 1)
    labels = ints([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    grouped = ChronologicalGroupedSplits(features, labels, keys, 0.5, 0.2, 0)(seed=1)
    train_column: NDArray[np.float64] = grouped.split.x_train[:, 0]
    assert np.array_equal(train_column, floats([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert np.array_equal(grouped.test_groups, ints([7, 8, 9]))
    test_column: NDArray[np.float64] = grouped.split.x_test[:, 0]
    assert at(test_column, 0) == 7.0


def test_the_chronological_partition_does_not_move_with_the_seed() -> None:
    keys = ints(list(range(10)))
    features: NDArray[np.float64] = keys.astype(np.float64).reshape(-1, 1)
    labels = ints([0, 1] * 5)
    factory = ChronologicalGroupedSplits(features, labels, keys, 0.5, 0.2, 0)
    assert np.array_equal(factory(seed=1).test_groups, factory(seed=2).test_groups)


def test_chronological_split_refuses_misaligned_arrays() -> None:
    keys = ints(list(range(10)))
    features: NDArray[np.float64] = keys.astype(np.float64).reshape(-1, 1)
    with pytest.raises(ValueError, match=ERR_LENGTH_MISMATCH):
        ChronologicalGroupedSplits(features, ints([0, 1]), keys, 0.5, 0.2, 0)
