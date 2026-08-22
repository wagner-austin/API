"""Tests for construction and wiring of the benchmark's collaborators."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.benchmarking.factory import (
    make_baseline_trainers,
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


def test_trainers_are_built_baseline_variant_then_references() -> None:
    """Arm order sets the rotation's starting point, so it is part of the API.

    The ClearGBM baseline stays first so it occupies slot 0 at the first seed,
    matching every manifest written before variant arms existed.
    """
    trainers = make_trainers(make_benchmark_config())
    names = [trainer.model_name for trainer in trainers]
    assert names == ["cleargbm", "cleargbm@leaf_wise", "lightgbm", "xgboost"]


def test_baseline_trainers_exclude_every_variant_arm() -> None:
    """The reference set stays reachable as its own entry point."""
    trainers = make_baseline_trainers(make_benchmark_config())
    names = [trainer.model_name for trainer in trainers]
    assert names == ["cleargbm", "lightgbm", "xgboost"]


def test_every_arm_has_a_distinct_name() -> None:
    """A manifest groups by arm name, so a repeat would merge two series."""
    names = [trainer.model_name for trainer in make_trainers(make_benchmark_config())]
    assert len(names) == len(set(names))


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
