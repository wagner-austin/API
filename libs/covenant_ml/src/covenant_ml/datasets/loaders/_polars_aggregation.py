"""Polars-native aggregation functions for time-series data.

Provides memory-efficient groupby aggregations using Polars operations.
Internal module - used by timeseries_csv_loader.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.datasets.loaders._polars_utils import (
    PolarsColFnProtocol,
    PolarsDataFrameProtocol,
    PolarsExprProtocol,
    extract_entity_ids,
    extract_feature_array,
)
from covenant_ml.datasets.types import AggregationStrategy


def aggregate_timeseries(
    df: PolarsDataFrameProtocol,
    entity_col: str,
    time_col: str,
    feature_columns: list[str],
    aggregation: AggregationStrategy,
) -> tuple[NDArray[np.float64], list[str]]:
    """Aggregate time-series features by entity using Polars groupby.

    Args:
        df: Polars DataFrame with numeric features.
        entity_col: Entity column name.
        time_col: Time column name for sorting.
        feature_columns: List of feature column names.
        aggregation: Aggregation strategy (last, first, mean, statistics).

    Returns:
        Tuple of (feature_array, entity_ids).
    """
    if aggregation == "last":
        return _aggregate_last(df, entity_col, time_col, feature_columns)
    if aggregation == "first":
        return _aggregate_first(df, entity_col, time_col, feature_columns)
    if aggregation == "mean":
        return _aggregate_mean(df, entity_col, feature_columns)
    return _aggregate_statistics(df, entity_col, feature_columns)


def _aggregate_last(
    df: PolarsDataFrameProtocol,
    entity_col: str,
    time_col: str,
    feature_columns: list[str],
) -> tuple[NDArray[np.float64], list[str]]:
    """Aggregate by taking last observation per entity.

    Args:
        df: Polars DataFrame.
        entity_col: Entity column name.
        time_col: Time column name.
        feature_columns: Feature column names.

    Returns:
        Tuple of (feature_array, entity_ids).
    """
    sorted_df = df.sort([entity_col, time_col])
    grouped = sorted_df.group_by(entity_col)
    result_df = grouped.last()
    result_df = result_df.sort(entity_col)

    entity_ids = extract_entity_ids(result_df, entity_col)
    x_array = extract_feature_array(result_df, feature_columns)

    return x_array, entity_ids


def _aggregate_first(
    df: PolarsDataFrameProtocol,
    entity_col: str,
    time_col: str,
    feature_columns: list[str],
) -> tuple[NDArray[np.float64], list[str]]:
    """Aggregate by taking first observation per entity.

    Args:
        df: Polars DataFrame.
        entity_col: Entity column name.
        time_col: Time column name.
        feature_columns: Feature column names.

    Returns:
        Tuple of (feature_array, entity_ids).
    """
    sorted_df = df.sort([entity_col, time_col])
    grouped = sorted_df.group_by(entity_col)
    result_df = grouped.first()
    result_df = result_df.sort(entity_col)

    entity_ids = extract_entity_ids(result_df, entity_col)
    x_array = extract_feature_array(result_df, feature_columns)

    return x_array, entity_ids


def _aggregate_mean(
    df: PolarsDataFrameProtocol,
    entity_col: str,
    feature_columns: list[str],
) -> tuple[NDArray[np.float64], list[str]]:
    """Aggregate by computing mean per entity.

    Args:
        df: Polars DataFrame.
        entity_col: Entity column name.
        feature_columns: Feature column names.

    Returns:
        Tuple of (feature_array, entity_ids).
    """
    polars_mod = __import__("polars")
    col_fn: PolarsColFnProtocol = polars_mod.col

    grouped = df.group_by(entity_col)

    agg_exprs: list[PolarsExprProtocol] = []
    for col_name in feature_columns:
        col_expr = col_fn(col_name)
        agg_exprs.append(col_expr.mean())

    result_df = grouped.agg(agg_exprs)
    result_df = result_df.sort(entity_col)

    entity_ids = extract_entity_ids(result_df, entity_col)
    x_array = extract_feature_array(result_df, feature_columns)

    return x_array, entity_ids


def _aggregate_statistics(
    df: PolarsDataFrameProtocol,
    entity_col: str,
    feature_columns: list[str],
) -> tuple[NDArray[np.float64], list[str]]:
    """Aggregate by computing mean, std, min, max per entity.

    Args:
        df: Polars DataFrame.
        entity_col: Entity column name.
        feature_columns: Feature column names.

    Returns:
        Tuple of (feature_array, entity_ids).
    """
    polars_mod = __import__("polars")
    col_fn: PolarsColFnProtocol = polars_mod.col

    grouped = df.group_by(entity_col)

    agg_exprs: list[PolarsExprProtocol] = []
    output_columns: list[str] = []

    for col_name in feature_columns:
        col_expr = col_fn(col_name)

        agg_exprs.append(col_expr.mean().alias(f"{col_name}_mean"))
        output_columns.append(f"{col_name}_mean")

        agg_exprs.append(col_expr.std(ddof=0).alias(f"{col_name}_std"))
        output_columns.append(f"{col_name}_std")

        agg_exprs.append(col_expr.min().alias(f"{col_name}_min"))
        output_columns.append(f"{col_name}_min")

        agg_exprs.append(col_expr.max().alias(f"{col_name}_max"))
        output_columns.append(f"{col_name}_max")

    result_df = grouped.agg(agg_exprs)
    result_df = result_df.sort(entity_col)

    entity_ids = extract_entity_ids(result_df, entity_col)
    x_array = extract_feature_array(result_df, output_columns)

    return x_array, entity_ids


def build_statistics_feature_names(base_names: list[str]) -> list[str]:
    """Build feature names for statistics aggregation.

    Creates 4 names per base feature: _mean, _std, _min, _max.

    Args:
        base_names: Original feature names.

    Returns:
        Expanded feature names with statistical suffixes.
    """
    result: list[str] = []
    for name in base_names:
        result.append(f"{name}_mean")
        result.append(f"{name}_std")
        result.append(f"{name}_min")
        result.append(f"{name}_max")
    return result


__all__ = [
    "aggregate_timeseries",
    "build_statistics_feature_names",
]
