"""Tests for polars ranking and diff feature functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.datasets.loaders._polars_encoding import convert_to_numeric
from covenant_ml.datasets.loaders._polars_ranking import (
    build_diff_feature_names,
    build_rank_feature_names,
    compute_diff_features,
    compute_entity_rank_features,
)
from covenant_ml.datasets.loaders._polars_utils import (
    PolarsDataFrameProtocol,
    PolarsReadCSVProtocol,
)


def _get_test_csv_path() -> Path:
    """Get path to ranking test fixture CSV."""
    return Path(__file__).parent.parent / "fixtures" / "ranking_test" / "data.csv"


def _load_test_df() -> PolarsDataFrameProtocol:
    """Load test DataFrame from CSV fixture using protocol typing.

    Matches loader behavior: reads CSV as strings then converts to numeric.
    """
    polars_mod = __import__("polars")
    read_csv_fn: PolarsReadCSVProtocol = polars_mod.read_csv
    raw_df: PolarsDataFrameProtocol = read_csv_fn(
        _get_test_csv_path(),
        encoding="utf8",
        infer_schema_length=0,
    )
    # Convert feature columns to numeric (matches loader behavior)
    feature_columns = ["feature_1", "feature_2"]
    return convert_to_numeric(raw_df, feature_columns, categorical_columns=set())


# =============================================================================
# Rank Feature Tests
# =============================================================================


def test_compute_entity_rank_features_returns_correct_structure() -> None:
    """compute_entity_rank_features returns dict with expected structure."""
    df = _load_test_df()

    result = compute_entity_rank_features(
        df,
        entity_col="customer_ID",
        feature_columns=["feature_1", "feature_2"],
    )

    # Verify structure via typed access
    features: NDArray[np.float64] = result["features"]
    feature_names: list[str] = result["feature_names"]
    entity_ids: list[str] = result["entity_ids"]

    # 3 entities (A, B, C), 2 features -> shape (3, 2)
    assert features.shape == (3, 2)
    assert feature_names == ["feature_1_rank", "feature_2_rank"]
    assert entity_ids == ["A", "B", "C"]


def test_compute_entity_rank_features_correct_shape() -> None:
    """Rank features have correct shape (n_entities, n_features)."""
    df = _load_test_df()

    result = compute_entity_rank_features(
        df,
        entity_col="customer_ID",
        feature_columns=["feature_1", "feature_2"],
    )

    # 3 unique entities, 2 rank features (one per input feature)
    assert result["features"].shape == (3, 2)
    assert result["feature_names"] == ["feature_1_rank", "feature_2_rank"]
    assert result["entity_ids"] == ["A", "B", "C"]


def test_compute_entity_rank_features_values_in_range() -> None:
    """Rank features are percentiles between 0 and 1."""
    df = _load_test_df()

    result = compute_entity_rank_features(
        df,
        entity_col="customer_ID",
        feature_columns=["feature_1", "feature_2"],
    )

    features: NDArray[np.float64] = result["features"]
    assert np.all(features >= 0.0)
    assert np.all(features <= 1.0)


def test_compute_entity_rank_features_sorted_entities() -> None:
    """Entity IDs are sorted alphabetically."""
    df = _load_test_df()

    result = compute_entity_rank_features(
        df,
        entity_col="customer_ID",
        feature_columns=["feature_1"],
    )

    assert result["entity_ids"] == ["A", "B", "C"]


def test_compute_entity_rank_features_correct_names() -> None:
    """Feature names have _rank suffix."""
    df = _load_test_df()

    result = compute_entity_rank_features(
        df,
        entity_col="customer_ID",
        feature_columns=["feature_1", "feature_2"],
    )

    assert result["feature_names"] == ["feature_1_rank", "feature_2_rank"]


def test_compute_entity_rank_features_raises_on_empty_columns() -> None:
    """Raises ValueError when feature_columns is empty."""
    df = _load_test_df()

    with pytest.raises(ValueError, match="feature_columns cannot be empty"):
        compute_entity_rank_features(
            df,
            entity_col="customer_ID",
            feature_columns=[],
        )


# =============================================================================
# Diff Feature Tests
# =============================================================================


def test_compute_diff_features_returns_correct_structure() -> None:
    """compute_diff_features returns dict with expected structure."""
    df = _load_test_df()

    result = compute_diff_features(
        df,
        entity_col="customer_ID",
        time_col="S_2",
        feature_columns=["feature_1", "feature_2"],
    )

    # Verify structure via typed access
    features: NDArray[np.float64] = result["features"]
    feature_names: list[str] = result["feature_names"]
    entity_ids: list[str] = result["entity_ids"]

    # 3 entities, 2 features * 5 aggs = 10 output features
    assert features.shape == (3, 10)
    assert feature_names[0] == "feature_1_diff_mean"
    assert entity_ids == ["A", "B", "C"]


def test_compute_diff_features_correct_shape() -> None:
    """Diff features have correct shape (n_entities, n_features * 5)."""
    df = _load_test_df()

    result = compute_diff_features(
        df,
        entity_col="customer_ID",
        time_col="S_2",
        feature_columns=["feature_1", "feature_2"],
    )

    # 3 unique entities, 2 input features * 5 aggregations = 10 output features
    assert result["features"].shape == (3, 10)
    assert result["feature_names"][0] == "feature_1_diff_mean"
    assert result["entity_ids"] == ["A", "B", "C"]


def test_compute_diff_features_sorted_entities() -> None:
    """Entity IDs are sorted alphabetically."""
    df = _load_test_df()

    result = compute_diff_features(
        df,
        entity_col="customer_ID",
        time_col="S_2",
        feature_columns=["feature_1"],
    )

    assert result["entity_ids"] == ["A", "B", "C"]


def test_compute_diff_features_correct_names() -> None:
    """Feature names have _diff_{agg} suffix."""
    df = _load_test_df()

    result = compute_diff_features(
        df,
        entity_col="customer_ID",
        time_col="S_2",
        feature_columns=["feature_1"],
    )

    expected_names = [
        "feature_1_diff_mean",
        "feature_1_diff_std",
        "feature_1_diff_min",
        "feature_1_diff_max",
        "feature_1_diff_last",
    ]
    assert result["feature_names"] == expected_names


def test_compute_diff_features_values_reasonable() -> None:
    """Diff features have expected values for known input."""
    # Customer A: feature_1 = [1, 2, 3], diffs = [NaN, 1, 1]
    # mean=1, std=0, min=1, max=1, last=1
    df = _load_test_df()

    result = compute_diff_features(
        df,
        entity_col="customer_ID",
        time_col="S_2",
        feature_columns=["feature_1"],
    )

    # Entity A is first (sorted), feature_1 diffs are [1, 1]
    # mean=1.0, std=0.0, min=1.0, max=1.0, last=1.0
    features: NDArray[np.float64] = result["features"]

    # Extract entity A row (first row, all 5 aggregations)
    a_row: NDArray[np.float64] = features[0, :]

    # Expected values: [mean=1.0, std=0.0, min=1.0, max=1.0, last=1.0]
    expected: NDArray[np.float64] = np.zeros(5, dtype=np.float64)
    expected[0] = 1.0  # mean
    expected[1] = 0.0  # std
    expected[2] = 1.0  # min
    expected[3] = 1.0  # max
    expected[4] = 1.0  # last

    # Use allclose for comparison (returns bool)
    assert np.allclose(a_row, expected, atol=0.01)


def test_compute_diff_features_raises_on_empty_columns() -> None:
    """Raises ValueError when feature_columns is empty."""
    df = _load_test_df()

    with pytest.raises(ValueError, match="feature_columns cannot be empty"):
        compute_diff_features(
            df,
            entity_col="customer_ID",
            time_col="S_2",
            feature_columns=[],
        )


# =============================================================================
# Name Builder Tests
# =============================================================================


def test_build_rank_feature_names() -> None:
    """build_rank_feature_names creates correct names."""
    names = build_rank_feature_names(["a", "b", "c"])

    assert names == ["a_rank", "b_rank", "c_rank"]


def test_build_rank_feature_names_empty() -> None:
    """build_rank_feature_names handles empty input."""
    names = build_rank_feature_names([])

    assert names == []


def test_build_diff_feature_names() -> None:
    """build_diff_feature_names creates correct names."""
    names = build_diff_feature_names(["a", "b"])

    expected = [
        "a_diff_mean",
        "a_diff_std",
        "a_diff_min",
        "a_diff_max",
        "a_diff_last",
        "b_diff_mean",
        "b_diff_std",
        "b_diff_min",
        "b_diff_max",
        "b_diff_last",
    ]
    assert names == expected


def test_build_diff_feature_names_empty() -> None:
    """build_diff_feature_names handles empty input."""
    names = build_diff_feature_names([])

    assert names == []
