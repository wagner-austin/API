"""Actual label endpoints, not fixed row gaps, govern chronological exclusion."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.benchmarking.splitting import SplitMasks
from covenant_ml.evaluation.splits import ChronologicalGroupedSplits
from covenant_ml.evaluation.temporal import purge_label_overlap

from ._fixtures import ints


def masks() -> SplitMasks:
    """Three explicit chronological folds on a nine-observation history."""
    times = ints(list(range(9)))
    train: NDArray[np.bool_] = times < 3
    val: NDArray[np.bool_] = (times >= 3) & (times < 6)
    test: NDArray[np.bool_] = times >= 6
    return SplitMasks(train, val, test)


def test_labels_reaching_the_next_decision_are_excluded() -> None:
    times = ints(list(range(9)))
    available = ints([0, 2, 3, 4, 6, 7, 7, 8, 9])
    original = masks()
    result = purge_label_overlap(times, available, original)
    np.testing.assert_array_equal(times[result.train], ints([0, 1]))
    np.testing.assert_array_equal(times[result.val], ints([3]))
    np.testing.assert_array_equal(times[result.test], ints([6, 7, 8]))
    np.testing.assert_array_equal(times[original.train], ints([0, 1, 2]))
    available[6:] = 1000
    later = purge_label_overlap(times, available, original)
    np.testing.assert_array_equal(later.train, result.train)
    np.testing.assert_array_equal(later.val, result.val)


@pytest.mark.parametrize(
    "case", ["shape", "length", "empty", "reversed", "overlap", "order", "purged"]
)
def test_invalid_or_empty_temporal_partitions_fail(case: str) -> None:
    times = ints(list(range(9)))
    available = times.copy()
    plan = masks()
    if case == "shape":
        available = available.reshape(9, 1)
    elif case == "length":
        available = available[:-1]
    elif case == "empty":
        plan.train[:] = False
    elif case == "reversed":
        available[0] = -1
    elif case == "overlap":
        plan.train[4] = True
    elif case == "order":
        plan = SplitMasks(plan.val, plan.train, plan.test)
    else:
        available[:3] = 10
    with pytest.raises(ValueError, match="CVML-EVAL"):
        purge_label_overlap(times, available, plan)


def test_label_availability_sorts_with_its_observation() -> None:
    times = ints(list(reversed(range(12))))
    features: NDArray[np.float64] = times.astype(np.float64).reshape(-1, 1)
    labels: NDArray[np.int64] = times % 2
    available: NDArray[np.int64] = times + 2
    split = ChronologicalGroupedSplits(
        features, labels, times, 0.5, 0.25, 0, label_available_at=available
    )(7).split
    np.testing.assert_array_equal(split.x_train[:, 0], ints([0, 1, 2, 3]))
    np.testing.assert_array_equal(split.x_val[:, 0], ints([6]))
    np.testing.assert_array_equal(split.x_test[:, 0], ints([9, 10, 11]))


def test_inner_stopping_also_excludes_unavailable_labels() -> None:
    times = ints(list(range(40)))
    features: NDArray[np.float64] = times.astype(np.float64).reshape(-1, 1)
    labels: NDArray[np.int64] = times % 2
    available: NDArray[np.int64] = times + 3
    split = (
        ChronologicalGroupedSplits(
            features, labels, times, 0.5, 0.25, 1, label_available_at=available
        )
        .validation_split(0.7)
        .split
    )
    np.testing.assert_array_equal(split.x_train[:, 0], ints(list(range(12))))
    np.testing.assert_array_equal(split.x_val[:, 0], ints([15, 16, 17]))
    np.testing.assert_array_equal(split.x_test[:, 0], ints(list(range(21, 30))))


def test_label_availability_requires_one_value_per_row() -> None:
    times = ints(list(range(12)))
    features: NDArray[np.float64] = times.astype(np.float64).reshape(-1, 1)
    with pytest.raises(ValueError, match="CVML-EVAL-001"):
        ChronologicalGroupedSplits(
            features, times, times, 0.5, 0.25, 0, label_available_at=times[:-1]
        )
