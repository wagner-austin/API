"""Tests for polars window aggregation functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.datasets.loaders._polars_encoding import convert_to_numeric
from covenant_ml.datasets.loaders._polars_utils import (
    PolarsDataFrameProtocol,
    PolarsReadCSVProtocol,
)
from covenant_ml.datasets.loaders._polars_window import (
    WindowFeatureResult,
    build_multi_window_feature_names,
    build_window_feature_names,
    compute_multi_window_features,
    compute_window_features,
)


def _get_test_csv_path() -> Path:
    """Get path to ranking test fixture CSV.

    Returns:
        Path to test fixture CSV file.
    """
    return Path(__file__).parent.parent / "fixtures" / "ranking_test" / "data.csv"


def _load_test_df() -> PolarsDataFrameProtocol:
    """Load test DataFrame from CSV fixture using protocol typing.

    Matches loader behavior: reads CSV as strings then converts to numeric.

    Returns:
        Polars DataFrame with numeric feature columns.
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
# Single Window Feature Tests
# =============================================================================


def test_compute_window_features_returns_correct_structure() -> None:
    """compute_window_features returns TypedDict with expected structure."""
    df = _load_test_df()

    result: WindowFeatureResult = compute_window_features(
        df,
        entity_col="customer_ID",
        time_col="S_2",
        feature_columns=["feature_1", "feature_2"],
        window_size=2,
    )

    # Verify structure via typed access
    features: NDArray[np.float64] = result["features"]
    feature_names: list[str] = result["feature_names"]
    entity_ids: list[str] = result["entity_ids"]

    # 3 entities (A, B, C), 2 features * 4 aggs = 8 output features
    assert features.shape == (3, 8)
    assert feature_names[0] == "feature_1_last2_mean"
    assert entity_ids == ["A", "B", "C"]


def test_compute_window_features_correct_shape() -> None:
    """Window features have correct shape (n_entities, n_features * 4)."""
    df = _load_test_df()

    result = compute_window_features(
        df,
        entity_col="customer_ID",
        time_col="S_2",
        feature_columns=["feature_1", "feature_2"],
        window_size=3,
    )

    # 3 unique entities, 2 input features * 4 aggregations = 8 output features
    assert result["features"].shape == (3, 8)
    assert result["feature_names"][0] == "feature_1_last3_mean"
    assert result["entity_ids"] == ["A", "B", "C"]


def test_compute_window_features_sorted_entities() -> None:
    """Entity IDs are sorted alphabetically."""
    df = _load_test_df()

    result = compute_window_features(
        df,
        entity_col="customer_ID",
        time_col="S_2",
        feature_columns=["feature_1"],
        window_size=2,
    )

    assert result["entity_ids"] == ["A", "B", "C"]


def test_compute_window_features_correct_names() -> None:
    """Feature names have _lastN_{agg} suffix."""
    df = _load_test_df()

    result = compute_window_features(
        df,
        entity_col="customer_ID",
        time_col="S_2",
        feature_columns=["feature_1"],
        window_size=3,
    )

    expected_names = [
        "feature_1_last3_mean",
        "feature_1_last3_std",
        "feature_1_last3_min",
        "feature_1_last3_max",
    ]
    assert result["feature_names"] == expected_names


def test_compute_window_features_values_for_known_input() -> None:
    """Window features have expected values for known input.

    Customer A: feature_1 = [1, 2, 3] (ordered by S_2)
    window_size=2 means we take last 2: [2, 3]
    mean=2.5, std=0.5, min=2, max=3
    """
    df = _load_test_df()

    result = compute_window_features(
        df,
        entity_col="customer_ID",
        time_col="S_2",
        feature_columns=["feature_1"],
        window_size=2,
    )

    features: NDArray[np.float64] = result["features"]

    # Entity A is first (sorted), feature_1 window of 2 = [2, 3]
    # mean=2.5, std=0.5, min=2.0, max=3.0
    a_row: NDArray[np.float64] = features[0, :]

    expected: NDArray[np.float64] = np.zeros(4, dtype=np.float64)
    expected[0] = 2.5  # mean
    expected[1] = 0.5  # std
    expected[2] = 2.0  # min
    expected[3] = 3.0  # max

    assert np.allclose(a_row, expected, atol=0.01)


def test_compute_window_features_window_larger_than_data() -> None:
    """Window size larger than entity's data uses all available rows.

    Customer B has 3 rows. With window_size=5, uses all 3 rows.
    feature_1 = [10, 20, 30], mean=20, std=8.165, min=10, max=30
    """
    df = _load_test_df()

    result = compute_window_features(
        df,
        entity_col="customer_ID",
        time_col="S_2",
        feature_columns=["feature_1"],
        window_size=5,
    )

    features: NDArray[np.float64] = result["features"]

    # Entity B is second (sorted), has 3 rows, window uses all
    b_row: NDArray[np.float64] = features[1, :]

    # feature_1 = [10, 20, 30]
    # mean = 20, std (population) = sqrt(200/3) = 8.165, min = 10, max = 30
    expected: NDArray[np.float64] = np.zeros(4, dtype=np.float64)
    expected[0] = 20.0  # mean
    expected[1] = 8.165  # std (population)
    expected[2] = 10.0  # min
    expected[3] = 30.0  # max

    assert np.allclose(b_row, expected, atol=0.01)


def test_compute_window_features_raises_on_empty_columns() -> None:
    """Raises ValueError when feature_columns is empty."""
    df = _load_test_df()

    with pytest.raises(ValueError, match="feature_columns cannot be empty"):
        compute_window_features(
            df,
            entity_col="customer_ID",
            time_col="S_2",
            feature_columns=[],
            window_size=3,
        )


def test_compute_window_features_raises_on_invalid_window_size() -> None:
    """Raises ValueError when window_size < 1."""
    df = _load_test_df()

    with pytest.raises(ValueError, match="window_size must be >= 1"):
        compute_window_features(
            df,
            entity_col="customer_ID",
            time_col="S_2",
            feature_columns=["feature_1"],
            window_size=0,
        )


# =============================================================================
# Multi-Window Feature Tests
# =============================================================================


def test_compute_multi_window_features_correct_structure() -> None:
    """compute_multi_window_features returns combined features from all windows."""
    df = _load_test_df()

    result = compute_multi_window_features(
        df,
        entity_col="customer_ID",
        time_col="S_2",
        feature_columns=["feature_1"],
        window_sizes=(2, 3),
    )

    # 3 entities, 1 feature * 4 aggs * 2 windows = 8 output features
    assert result["features"].shape == (3, 8)
    assert result["entity_ids"] == ["A", "B", "C"]


def test_compute_multi_window_features_correct_names() -> None:
    """Multi-window features have correct names for all window sizes."""
    df = _load_test_df()

    result = compute_multi_window_features(
        df,
        entity_col="customer_ID",
        time_col="S_2",
        feature_columns=["feature_1"],
        window_sizes=(3, 6),
    )

    # Should have _last3 and _last6 versions
    expected_names = [
        "feature_1_last3_mean",
        "feature_1_last3_std",
        "feature_1_last3_min",
        "feature_1_last3_max",
        "feature_1_last6_mean",
        "feature_1_last6_std",
        "feature_1_last6_min",
        "feature_1_last6_max",
    ]
    assert result["feature_names"] == expected_names


def test_compute_multi_window_features_raises_on_empty_columns() -> None:
    """Raises ValueError when feature_columns is empty."""
    df = _load_test_df()

    with pytest.raises(ValueError, match="feature_columns cannot be empty"):
        compute_multi_window_features(
            df,
            entity_col="customer_ID",
            time_col="S_2",
            feature_columns=[],
            window_sizes=(3, 6),
        )


def test_compute_multi_window_features_raises_on_empty_window_sizes() -> None:
    """Raises ValueError when window_sizes is empty."""
    df = _load_test_df()

    with pytest.raises(ValueError, match="window_sizes cannot be empty"):
        compute_multi_window_features(
            df,
            entity_col="customer_ID",
            time_col="S_2",
            feature_columns=["feature_1"],
            window_sizes=(),
        )


# =============================================================================
# Name Builder Tests
# =============================================================================


def test_build_window_feature_names() -> None:
    """build_window_feature_names creates correct names."""
    names = build_window_feature_names(["a", "b"], window_size=3)

    expected = [
        "a_last3_mean",
        "a_last3_std",
        "a_last3_min",
        "a_last3_max",
        "b_last3_mean",
        "b_last3_std",
        "b_last3_min",
        "b_last3_max",
    ]
    assert names == expected


def test_build_window_feature_names_empty() -> None:
    """build_window_feature_names handles empty input."""
    names = build_window_feature_names([], window_size=5)

    assert names == []


def test_build_multi_window_feature_names() -> None:
    """build_multi_window_feature_names creates correct names."""
    names = build_multi_window_feature_names(["a"], window_sizes=(3, 6))

    expected = [
        "a_last3_mean",
        "a_last3_std",
        "a_last3_min",
        "a_last3_max",
        "a_last6_mean",
        "a_last6_std",
        "a_last6_min",
        "a_last6_max",
    ]
    assert names == expected


def test_build_multi_window_feature_names_empty_sizes() -> None:
    """build_multi_window_feature_names handles empty window_sizes."""
    names = build_multi_window_feature_names(["a", "b"], window_sizes=())

    assert names == []
