"""Tests for construction and wiring of the benchmark's collaborators."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.benchmarking.factory import (
    make_benchmark_config,
    make_split_factory,
    make_trainers,
)


def build_arrays() -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Build a small grouped dataset.

    Returns:
        Features, labels and company codes.
    """
    n_companies = 40
    rows_per_company = 4
    n_rows = n_companies * rows_per_company
    rng = np.random.default_rng(0)
    features: NDArray[np.float64] = rng.random((n_rows, 3), dtype=np.float64)
    labels: NDArray[np.int64] = np.arange(n_rows, dtype=np.int64) % 2
    codes: NDArray[np.int64] = np.repeat(np.arange(n_companies, dtype=np.int64), rows_per_company)
    return features, labels, codes


def test_default_config_matches_the_tuned_workload() -> None:
    config = make_benchmark_config()
    assert config["n_estimators"] == 200
    assert config["max_depth"] == 6
    assert config["max_bins"] == 64
    assert config["num_leaves"] == 31
    assert config["n_jobs"] == 1


def test_config_overrides_are_applied() -> None:
    config = make_benchmark_config(
        n_estimators=10,
        max_depth=3,
        max_bins=16,
        num_leaves=7,
        repeats=2,
        warmups=1,
    )
    assert config["n_estimators"] == 10
    assert config["max_depth"] == 3
    assert config["max_bins"] == 16
    assert config["num_leaves"] == 7
    assert config["repeats"] == 2
    assert config["warmups"] == 1


def test_trainers_are_built_in_cleargbm_then_lightgbm_order() -> None:
    cleargbm, lightgbm = make_trainers(make_benchmark_config())
    assert cleargbm.model_name == "cleargbm"
    assert lightgbm.model_name == "lightgbm"


def test_split_factory_partitions_for_a_seed() -> None:
    features, labels, codes = build_arrays()
    factory = make_split_factory(features, labels, codes)
    split = factory(42)
    total = len(split.y_train) + len(split.y_val) + len(split.y_test)
    assert total == len(labels)


def test_split_factory_is_deterministic_per_seed() -> None:
    features, labels, codes = build_arrays()
    factory = make_split_factory(features, labels, codes)
    assert np.array_equal(factory(5).y_test, factory(5).y_test)


def test_split_factory_varies_with_the_seed() -> None:
    """Different seeds must select different companies.

    Compares the feature rows, not the labels: the labels alternate 0/1 by
    construction, so two different partitions can produce identical label
    sequences and a label comparison would assert nothing.
    """
    features, labels, codes = build_arrays()
    factory = make_split_factory(features, labels, codes)
    first = factory(1).x_test
    second = factory(2).x_test
    assert first.shape != second.shape or not np.array_equal(first, second)
