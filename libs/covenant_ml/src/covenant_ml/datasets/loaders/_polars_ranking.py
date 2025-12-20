"""Polars-native ranking and diff functions for competition features.

Provides per-entity percentile rankings and row-to-row differences.
These features are critical for time-series prediction competitions.

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
    extract_entity_ids,
    extract_feature_array,
)


class RankFeatureResult(TypedDict):
    """Result of rank feature computation."""

    features: NDArray[np.float64]
    feature_names: list[str]
    entity_ids: list[str]


class DiffFeatureResult(TypedDict):
    """Result of diff feature computation."""

    features: NDArray[np.float64]
    feature_names: list[str]
    entity_ids: list[str]


def compute_entity_rank_features(
    df: PolarsDataFrameProtocol,
    entity_col: str,
    feature_columns: list[str],
) -> RankFeatureResult:
    """Compute per-entity percentile rankings for each feature.

    For each feature, computes the percentile rank (0.0 to 1.0) within each
    entity's observations. Higher values mean the observation is higher
    relative to the entity's history.

    This captures relative position in customer history, which is more
    predictive than absolute values for many use cases.

    Args:
        df: Polars DataFrame with time-series data.
        entity_col: Column name for entity identifier (e.g., customer_ID).
        feature_columns: Numeric columns to compute rankings for.

    Returns:
        RankFeatureResult with rank features (values in [0, 1]).

    Raises:
        ValueError: If feature_columns is empty.
    """
    if not feature_columns:
        raise ValueError("feature_columns cannot be empty")

    polars_mod = __import__("polars")
    col_fn: PolarsColFnProtocol = polars_mod.col

    # Build rank expressions for each feature
    rank_exprs: list[PolarsExprProtocol] = []
    output_names: list[str] = []

    for col_name in feature_columns:
        # Compute rank within each entity, normalized to [0, 1]
        # rank() with method="average" + divide by count gives percentile
        col_expr = col_fn(col_name)
        count_expr = col_fn(col_name).len().over(entity_col)

        # Rank expression: rank / count gives percentile (0 to 1)
        rank_expr = (col_expr.rank(method="average").over(entity_col) / count_expr).alias(
            f"{col_name}_rank"
        )

        rank_exprs.append(rank_expr)
        output_names.append(f"{col_name}_rank")

    # Add entity column for grouping output
    rank_exprs.insert(0, col_fn(entity_col))

    # Apply ranking transformations
    ranked_df = df.select(rank_exprs)

    # Aggregate to one row per entity (take last rank value)
    grouped = ranked_df.group_by(entity_col)
    result_df = grouped.last()
    result_df = result_df.sort(entity_col)

    # Extract results
    entity_ids = extract_entity_ids(result_df, entity_col)
    features = extract_feature_array(result_df, output_names)

    return RankFeatureResult(
        features=features,
        feature_names=output_names,
        entity_ids=entity_ids,
    )


def compute_diff_features(
    df: PolarsDataFrameProtocol,
    entity_col: str,
    time_col: str,
    feature_columns: list[str],
) -> DiffFeatureResult:
    """Compute row-to-row differences and aggregate per entity.

    For each feature, computes diff[i] = value[i] - value[i-1] within each
    entity's time-ordered observations. Then aggregates diffs using
    mean, std, min, max, and last.

    This captures trends and volatility in customer behavior over time.

    Args:
        df: Polars DataFrame with time-series data.
        entity_col: Column name for entity identifier (e.g., customer_ID).
        time_col: Column name for time ordering (e.g., S_2 date).
        feature_columns: Numeric columns to compute diffs for.

    Returns:
        DiffFeatureResult with aggregated diff features.

    Raises:
        ValueError: If feature_columns is empty.
    """
    if not feature_columns:
        raise ValueError("feature_columns cannot be empty")

    polars_mod = __import__("polars")
    col_fn: PolarsColFnProtocol = polars_mod.col

    # Sort by entity and time first
    sorted_df = df.sort([entity_col, time_col])

    # Build diff expressions for each feature
    diff_exprs: list[PolarsExprProtocol] = [col_fn(entity_col)]

    for col_name in feature_columns:
        # Compute diff within each entity
        col_expr = col_fn(col_name)
        diff_expr = col_expr.diff().over(entity_col).alias(f"{col_name}_diff")
        diff_exprs.append(diff_expr)

    # Apply diff transformations
    diff_df = sorted_df.select(diff_exprs)

    # Now aggregate the diffs per entity
    grouped = diff_df.group_by(entity_col)

    agg_exprs: list[PolarsExprProtocol] = []
    output_names: list[str] = []

    for col_name in feature_columns:
        diff_col = f"{col_name}_diff"
        col_expr = col_fn(diff_col)

        # Aggregate diff statistics
        agg_exprs.append(col_expr.mean().alias(f"{diff_col}_mean"))
        output_names.append(f"{diff_col}_mean")

        agg_exprs.append(col_expr.std(ddof=0).alias(f"{diff_col}_std"))
        output_names.append(f"{diff_col}_std")

        agg_exprs.append(col_expr.min().alias(f"{diff_col}_min"))
        output_names.append(f"{diff_col}_min")

        agg_exprs.append(col_expr.max().alias(f"{diff_col}_max"))
        output_names.append(f"{diff_col}_max")

        agg_exprs.append(col_expr.last().alias(f"{diff_col}_last"))
        output_names.append(f"{diff_col}_last")

    result_df = grouped.agg(agg_exprs)
    result_df = result_df.sort(entity_col)

    # Extract results
    entity_ids = extract_entity_ids(result_df, entity_col)
    features = extract_feature_array(result_df, output_names)

    return DiffFeatureResult(
        features=features,
        feature_names=output_names,
        entity_ids=entity_ids,
    )


def build_rank_feature_names(base_names: list[str]) -> list[str]:
    """Build feature names for rank features.

    Creates 1 name per base feature: _rank.

    Args:
        base_names: Original feature names.

    Returns:
        Feature names with rank suffix.
    """
    return [f"{name}_rank" for name in base_names]


def build_diff_feature_names(base_names: list[str]) -> list[str]:
    """Build feature names for diff features.

    Creates 5 names per base feature: _diff_mean, _diff_std, _diff_min,
    _diff_max, _diff_last.

    Args:
        base_names: Original feature names.

    Returns:
        Expanded feature names with diff suffixes.
    """
    result: list[str] = []
    for name in base_names:
        result.append(f"{name}_diff_mean")
        result.append(f"{name}_diff_std")
        result.append(f"{name}_diff_min")
        result.append(f"{name}_diff_max")
        result.append(f"{name}_diff_last")
    return result


__all__ = [
    "DiffFeatureResult",
    "RankFeatureResult",
    "build_diff_feature_names",
    "build_rank_feature_names",
    "compute_diff_features",
    "compute_entity_rank_features",
]
