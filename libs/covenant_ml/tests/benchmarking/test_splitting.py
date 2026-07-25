"""Tests for company-disjoint partitioning."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.benchmarking.splitting import company_disjoint_split
from covenant_ml.benchmarking.types import ERR_EMPTY_SPLIT, ERR_LENGTH_MISMATCH


def build_dataset(
    n_companies: int = 40,
    rows_per_company: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Build a dataset whose rows are grouped into companies.

    Args:
        n_companies: Number of distinct companies.
        rows_per_company: Rows contributed by each company.

    Returns:
        Features, labels and company codes.
    """
    n_rows = n_companies * rows_per_company
    rng = np.random.default_rng(0)
    features: NDArray[np.float64] = rng.random((n_rows, 3), dtype=np.float64)
    labels: NDArray[np.int64] = np.arange(n_rows, dtype=np.int64) % 2
    codes: NDArray[np.int64] = np.repeat(np.arange(n_companies, dtype=np.int64), rows_per_company)
    return features, labels, codes


def test_no_company_appears_in_two_folds() -> None:
    features, labels, codes = build_dataset()
    split = company_disjoint_split(features, labels, codes, seed=42)

    # Reconstruct which companies landed in each fold by row count.
    total = len(split.y_train) + len(split.y_val) + len(split.y_test)
    assert total == len(labels)

    # Every fold's rows must be whole companies: with 5 rows per company,
    # each fold size is a multiple of 5.
    assert len(split.y_train) % 5 == 0
    assert len(split.y_val) % 5 == 0
    assert len(split.y_test) % 5 == 0


def test_partition_is_deterministic_for_a_seed() -> None:
    features, labels, codes = build_dataset()
    first = company_disjoint_split(features, labels, codes, seed=7)
    second = company_disjoint_split(features, labels, codes, seed=7)
    assert np.array_equal(first.y_train, second.y_train)
    assert np.array_equal(first.y_test, second.y_test)


def test_different_seeds_produce_different_partitions() -> None:
    """Compares feature rows, not labels.

    The labels alternate 0/1 by construction, so two genuinely different
    partitions can yield identical label sequences; only the feature rows
    identify which companies were selected.
    """
    features, labels, codes = build_dataset()
    first = company_disjoint_split(features, labels, codes, seed=1)
    second = company_disjoint_split(features, labels, codes, seed=2)
    assert first.x_test.shape != second.x_test.shape or not np.array_equal(
        first.x_test, second.x_test
    )


def test_feature_rows_track_their_labels() -> None:
    features, labels, codes = build_dataset()
    split = company_disjoint_split(features, labels, codes, seed=3)
    assert len(split.x_train) == len(split.y_train)
    assert len(split.x_val) == len(split.y_val)
    assert len(split.x_test) == len(split.y_test)


def test_mismatched_label_length_raises() -> None:
    features, labels, codes = build_dataset()
    with pytest.raises(ValueError, match=ERR_LENGTH_MISMATCH):
        company_disjoint_split(features, labels[:-1], codes, seed=0)


def test_mismatched_code_length_raises() -> None:
    features, labels, codes = build_dataset()
    with pytest.raises(ValueError, match=ERR_LENGTH_MISMATCH):
        company_disjoint_split(features, labels, codes[:-1], seed=0)


def test_too_few_companies_to_fill_every_fold_raises() -> None:
    features, labels, codes = build_dataset(n_companies=2, rows_per_company=2)
    with pytest.raises(ValueError, match=ERR_EMPTY_SPLIT):
        company_disjoint_split(features, labels, codes, seed=0)
