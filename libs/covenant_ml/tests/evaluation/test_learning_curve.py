"""Tests for nested learning curves."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.benchmarking.adapters import LightGbmTrainer
from covenant_ml.evaluation.learning_curve import (
    learning_curve_seed,
    require_fractions,
    training_prefix,
)
from covenant_ml.evaluation.splits import CompanyDisjointGroupedSplits
from covenant_ml.evaluation.types import ERR_INVALID_FRACTION

from ._fixtures import grouped_dataset, tiny_config


@pytest.mark.parametrize(
    "fractions",
    [[], [0.0, 1.0], [0.5, 1.5], [0.5, 0.5], [0.8, 0.4]],
)
def test_invalid_fractions_are_refused(fractions: list[float]) -> None:
    with pytest.raises(ValueError, match=ERR_INVALID_FRACTION):
        require_fractions(fractions)


def test_prefixes_are_nested_and_never_touch_held_out_rows() -> None:
    features, labels, codes = grouped_dataset()
    split = CompanyDisjointGroupedSplits(features, labels, codes)(seed=1).split
    order: NDArray[np.int64] = np.random.default_rng(0).permutation(len(split.y_train))
    small = training_prefix(split, order, 0.25)
    large = training_prefix(split, order, 0.5)
    n_small = len(small.y_train)
    head: NDArray[np.float64] = large.x_train[:n_small]
    assert np.array_equal(head, small.x_train)
    assert n_small == -(-len(split.y_train) // 4)
    assert np.array_equal(small.x_test, split.x_test)
    assert np.array_equal(small.y_val, split.y_val)


def test_a_tiny_fraction_still_keeps_one_row() -> None:
    features, labels, codes = grouped_dataset()
    split = CompanyDisjointGroupedSplits(features, labels, codes)(seed=1).split
    order: NDArray[np.int64] = np.arange(len(split.y_train), dtype=np.int64)
    assert len(training_prefix(split, order, 1e-9).y_train) == 1


def test_one_point_per_fraction_on_the_full_held_out_fold() -> None:
    features, labels, codes = grouped_dataset()
    split = CompanyDisjointGroupedSplits(features, labels, codes)(seed=2).split
    points = learning_curve_seed("full", LightGbmTrainer(tiny_config()), split, 2, [0.25, 0.5, 1.0])
    assert [p["train_fraction"] for p in points] == [0.25, 0.5, 1.0]
    assert [p["n_train"] for p in points][-1] == len(split.y_train)
    assert all(p["arm"] == "full" and p["seed"] == 2 for p in points)
    assert points[-1]["quality"]["positive_rate"] == points[0]["quality"]["positive_rate"]
