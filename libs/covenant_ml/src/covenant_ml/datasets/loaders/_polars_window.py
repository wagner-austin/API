"""Polars-native window aggregation functions for competition features.

Computes aggregations over the last N observations per entity (customer).
These "recency-weighted" features capture recent behavior which is often
more predictive than full-history aggregations.

Used by Kaggle competition solutions (AMEX 1st place used last3, last6).

Internal module - used by timeseries_csv_loader.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from covenant_ml.datasets.loaders._polars_utils import (
    PolarsColFnProtocol,
    PolarsDataFrameProtocol,
    PolarsExprProtocol,
    PolarsGroupByProtocol,
    extract_entity_ids,
    extract_feature_array,
)


class WindowFeatureResult(TypedDict):
    """Result of window feature computation."""

    features: NDArray[np.float64]
    feature_names: list[str]
    entity_ids: list[str]


def compute_window_features(
    df: PolarsDataFrameProtocol,
    entity_col: str,
    time_col: str,
    feature_columns: list[str],
    window_size: int,
) -> WindowFeatureResult:
    """Compute aggregations over the last N observations per entity.

    For each feature, computes mean, std, min, max over only the last
    window_size observations per entity. This captures recent behavior
    which is often more predictive than full-history aggregations.

    Example with window_size=3 and a customer with 10 statements:
    - Only uses statements 8, 9, 10 (the last 3)
    - Computes mean/std/min/max of each feature over those 3 rows

    Args:
        df: Polars DataFrame with time-series data.
        entity_col: Column name for entity identifier (e.g., customer_ID).
        time_col: Column name for time ordering (e.g., S_2 date).
        feature_columns: Numeric columns to compute window aggregations for.
        window_size: Number of most recent observations to use (e.g., 3, 6).

    Returns:
        WindowFeatureResult with window-aggregated features.

    Raises:
        ValueError: If feature_columns is empty or window_size < 1.
    """
    if not feature_columns:
        raise ValueError("feature_columns cannot be empty")
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    polars_mod = __import__("polars")
    col_fn: PolarsColFnProtocol = polars_mod.col

    # Sort by entity and time, then take last N rows per entity
    sorted_df: PolarsDataFrameProtocol = df.sort([entity_col, time_col])
    windowed_df: PolarsDataFrameProtocol = sorted_df.group_by(entity_col).tail(window_size)

    # Now aggregate the windowed data per entity
    grouped: PolarsGroupByProtocol = windowed_df.group_by(entity_col)

    agg_exprs: list[PolarsExprProtocol] = []
    output_names: list[str] = []
    suffix = f"_last{window_size}"

    for col_name in feature_columns:
        col_expr = col_fn(col_name)

        # Mean over window
        agg_exprs.append(col_expr.mean().alias(f"{col_name}{suffix}_mean"))
        output_names.append(f"{col_name}{suffix}_mean")

        # Std over window (population std with ddof=0)
        agg_exprs.append(col_expr.std(ddof=0).alias(f"{col_name}{suffix}_std"))
        output_names.append(f"{col_name}{suffix}_std")

        # Min over window
        agg_exprs.append(col_expr.min().alias(f"{col_name}{suffix}_min"))
        output_names.append(f"{col_name}{suffix}_min")

        # Max over window
        agg_exprs.append(col_expr.max().alias(f"{col_name}{suffix}_max"))
        output_names.append(f"{col_name}{suffix}_max")

    result_df: PolarsDataFrameProtocol = grouped.agg(agg_exprs)
    result_df = result_df.sort(entity_col)

    # Extract results
    entity_ids = extract_entity_ids(result_df, entity_col)
    features = extract_feature_array(result_df, output_names)

    return WindowFeatureResult(
        features=features,
        feature_names=output_names,
        entity_ids=entity_ids,
    )


def compute_multi_window_features(
    df: PolarsDataFrameProtocol,
    entity_col: str,
    time_col: str,
    feature_columns: list[str],
    window_sizes: tuple[int, ...],
) -> WindowFeatureResult:
    """Compute window aggregations for multiple window sizes.

    Combines features from multiple window sizes (e.g., last3 and last6)
    into a single result. This captures behavior at different time scales.

    Args:
        df: Polars DataFrame with time-series data.
        entity_col: Column name for entity identifier.
        time_col: Column name for time ordering.
        feature_columns: Numeric columns to compute window aggregations for.
        window_sizes: Tuple of window sizes (e.g., (3, 6)).

    Returns:
        WindowFeatureResult with combined window features for all sizes.

    Raises:
        ValueError: If feature_columns is empty, window_sizes is empty,
            or any window_size < 1.
    """
    if not feature_columns:
        raise ValueError("feature_columns cannot be empty")
    if not window_sizes:
        raise ValueError("window_sizes cannot be empty")

    all_features: list[NDArray[np.float64]] = []
    all_names: list[str] = []
    entity_ids: list[str] = []

    for idx, window_size in enumerate(window_sizes):
        result = compute_window_features(
            df=df,
            entity_col=entity_col,
            time_col=time_col,
            feature_columns=feature_columns,
            window_size=window_size,
        )
        all_features.append(result["features"])
        all_names.extend(result["feature_names"])

        # Capture entity IDs from first window (all should be identical)
        if idx == 0:
            entity_ids = result["entity_ids"]

    # Concatenate all window features horizontally
    combined: NDArray[np.float64] = np.hstack(all_features)

    return WindowFeatureResult(
        features=combined,
        feature_names=all_names,
        entity_ids=entity_ids,
    )


def build_window_feature_names(base_names: list[str], window_size: int) -> list[str]:
    """Build feature names for window aggregations.

    Creates 4 names per base feature: _lastN_mean, _lastN_std, _lastN_min, _lastN_max.

    Args:
        base_names: Original feature names.
        window_size: Window size for naming (e.g., 3 -> "_last3").

    Returns:
        Expanded feature names with window suffixes.
    """
    suffix = f"_last{window_size}"
    result: list[str] = []
    for name in base_names:
        result.append(f"{name}{suffix}_mean")
        result.append(f"{name}{suffix}_std")
        result.append(f"{name}{suffix}_min")
        result.append(f"{name}{suffix}_max")
    return result


def build_multi_window_feature_names(
    base_names: list[str],
    window_sizes: tuple[int, ...],
) -> list[str]:
    """Build feature names for multiple window sizes.

    Args:
        base_names: Original feature names.
        window_sizes: Tuple of window sizes.

    Returns:
        Expanded feature names for all window sizes.
    """
    result: list[str] = []
    for window_size in window_sizes:
        result.extend(build_window_feature_names(base_names, window_size))
    return result


__all__ = [
    "WindowFeatureResult",
    "build_multi_window_feature_names",
    "build_window_feature_names",
    "compute_multi_window_features",
    "compute_window_features",
]
