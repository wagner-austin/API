"""Tuning scores outer validation and never fits or stops on those rows."""

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.evaluation.splits import ChronologicalGroupedSplits


def test_validation_keeps_whole_groups_and_excludes_final_rows() -> None:
    keys: NDArray[np.int64] = np.repeat(np.arange(40, dtype=np.int64), 2)
    features: NDArray[np.float64] = keys.astype(np.float64).reshape(-1, 1)
    labels: NDArray[np.int64] = keys % 2
    factory = ChronologicalGroupedSplits(
        features, labels, keys, 0.5, 0.25, 1, label_available_at=keys
    )
    actual = factory.validation_split(0.7)
    np.testing.assert_array_equal(
        actual.split.x_train[:, 0], np.repeat(np.arange(14, dtype=np.int64), 2)
    )
    np.testing.assert_array_equal(
        actual.split.x_val[:, 0], np.repeat(np.arange(15, 20, dtype=np.int64), 2)
    )
    np.testing.assert_array_equal(
        actual.test_groups, np.repeat(np.arange(21, 30, dtype=np.int64), 2)
    )
    np.testing.assert_array_equal(actual.split.x_test[:, 0], actual.test_groups)
    # Alter only final-test rows. No fitting, stopping or scoring input may change.
    features[keys >= 31] = 1e9
    labels[keys >= 31] = 1
    changed = ChronologicalGroupedSplits(
        features, labels, keys, 0.5, 0.25, 1, label_available_at=keys
    ).validation_split(0.7)
    np.testing.assert_array_equal(actual.split.x_train, changed.split.x_train)
    np.testing.assert_array_equal(actual.split.y_train, changed.split.y_train)
    np.testing.assert_array_equal(actual.split.x_val, changed.split.x_val)
    np.testing.assert_array_equal(actual.split.y_val, changed.split.y_val)
    np.testing.assert_array_equal(actual.split.x_test, changed.split.x_test)
    np.testing.assert_array_equal(actual.split.y_test, changed.split.y_test)


@pytest.mark.parametrize("fraction", [0.0, 1.0, float("nan")])
def test_invalid_inner_fraction_is_refused(fraction: float) -> None:
    keys: NDArray[np.int64] = np.arange(40, dtype=np.int64)
    features: NDArray[np.float64] = keys.astype(np.float64).reshape(-1, 1)
    with pytest.raises(ValueError, match="inner training fraction"):
        ChronologicalGroupedSplits(
            features, keys % 2, keys, 0.5, 0.25, 1, label_available_at=keys
        ).validation_split(fraction)


@pytest.mark.parametrize("fraction", [0.01, 0.99])
def test_embargo_cannot_leave_an_empty_inner_fold(fraction: float) -> None:
    keys: NDArray[np.int64] = np.arange(40, dtype=np.int64)
    features: NDArray[np.float64] = keys.astype(np.float64).reshape(-1, 1)
    with pytest.raises(ValueError, match="inner training or stopping fold is empty"):
        ChronologicalGroupedSplits(
            features, keys % 2, keys, 0.5, 0.25, 1, label_available_at=keys
        ).validation_split(fraction)
