"""Polars-native categorical encoding for time-series data.

Detects and encodes categorical columns using Polars operations.
Internal module - used by timeseries_csv_loader.
"""

from __future__ import annotations

from covenant_ml.datasets.loaders._parsing import CATEGORICAL_MISSING, MISSING_VALUES
from covenant_ml.datasets.loaders._polars_utils import (
    PolarsColFnProtocol,
    PolarsDataFrameProtocol,
    PolarsDataTypeProtocol,
    PolarsExprProtocol,
    PolarsLitFnProtocol,
    PolarsWhenFnProtocol,
    is_numeric_string,
)
from covenant_ml.datasets.types import CategoricalEncoding


def detect_categorical_columns(
    df: PolarsDataFrameProtocol,
    feature_columns: list[str],
) -> set[int]:
    """Detect categorical columns using Polars operations.

    Samples rows to determine if columns contain non-numeric values.
    A column is categorical if ANY non-missing value cannot be parsed as float.

    Args:
        df: Polars DataFrame with string columns.
        feature_columns: List of feature column names.

    Returns:
        Set of feature indices that are categorical.
    """
    polars_mod = __import__("polars")
    col_fn: PolarsColFnProtocol = polars_mod.col

    categorical: set[int] = set()

    sample_df = df
    if df.height > 1000:
        sample_df = df.sample(n=1000, seed=42)

    feature_df = sample_df.select([col_fn(c) for c in feature_columns])

    for row in feature_df.iter_rows():
        for feat_idx, value in enumerate(row):
            if feat_idx in categorical:
                continue

            str_value = str(value) if value is not None else ""
            stripped = str_value.strip()

            if stripped in MISSING_VALUES:
                continue

            if not is_numeric_string(stripped):
                categorical.add(feat_idx)

    return categorical


def build_categorical_encodings(
    df: PolarsDataFrameProtocol,
    feature_columns: list[str],
    categorical_columns: set[int],
) -> list[CategoricalEncoding]:
    """Build categorical encodings using Polars operations.

    Args:
        df: Polars DataFrame.
        feature_columns: List of feature column names.
        categorical_columns: Set of categorical feature indices.

    Returns:
        List of CategoricalEncoding for each categorical column.
    """
    polars_mod = __import__("polars")
    col_fn: PolarsColFnProtocol = polars_mod.col

    encodings: list[CategoricalEncoding] = []

    for feat_idx in sorted(categorical_columns):
        column_name = feature_columns[feat_idx]

        col_expr = col_fn(column_name)
        unique_df = df.select(col_expr.unique())

        series = unique_df.to_series()
        unique_values_raw: list[str | None] = series.to_list()

        unique_values: set[str] = set()
        has_missing = False

        for val in unique_values_raw:
            str_val = str(val) if val is not None else ""
            stripped = str_val.strip()

            if stripped in MISSING_VALUES:
                has_missing = True
            else:
                unique_values.add(stripped)

        sorted_values = sorted(unique_values)

        mapping_list: list[tuple[str, int]] = []
        code_offset = 0

        if has_missing:
            mapping_list.append((CATEGORICAL_MISSING, 0))
            code_offset = 1

        for idx, val in enumerate(sorted_values):
            mapping_list.append((val, idx + code_offset))

        n_categories = len(mapping_list)

        encodings.append(
            CategoricalEncoding(
                column_name=column_name,
                mapping=tuple(mapping_list),
                n_categories=n_categories,
            )
        )

    return encodings


def apply_encodings(
    df: PolarsDataFrameProtocol,
    feature_columns: list[str],
    encodings: list[CategoricalEncoding],
    categorical_columns: set[int],
) -> PolarsDataFrameProtocol:
    """Apply categorical encodings to DataFrame using Polars.

    Args:
        df: Polars DataFrame with string columns.
        feature_columns: List of feature column names.
        encodings: List of categorical encodings.
        categorical_columns: Set of categorical feature indices.

    Returns:
        DataFrame with encoded categorical columns.
    """
    if not encodings:
        return df

    polars_mod = __import__("polars")
    col_fn: PolarsColFnProtocol = polars_mod.col
    lit_fn: PolarsLitFnProtocol = polars_mod.lit
    when_fn: PolarsWhenFnProtocol = polars_mod.when

    encoding_by_name: dict[str, dict[str, int]] = {}
    for enc in encodings:
        encoding_by_name[enc["column_name"]] = dict(enc["mapping"])

    result_df = df

    for feat_idx in sorted(categorical_columns):
        column_name = feature_columns[feat_idx]
        mapping = encoding_by_name[column_name]
        missing_code = mapping.get(CATEGORICAL_MISSING, 0)

        col_expr = col_fn(column_name)
        missing_condition = col_expr.is_null() | col_expr.is_in(list(MISSING_VALUES))

        expr: PolarsExprProtocol = when_fn(missing_condition).then(lit_fn(float(missing_code)))

        for val, code in mapping.items():
            if val != CATEGORICAL_MISSING:
                val_condition = col_fn(column_name).eq(lit_fn(val))
                expr = expr.when(val_condition).then(lit_fn(float(code)))

        expr = expr.otherwise(lit_fn(float(missing_code)))
        expr = expr.alias(column_name)

        result_df = result_df.with_columns(expr)

    return result_df


def convert_to_numeric(
    df: PolarsDataFrameProtocol,
    feature_columns: list[str],
    categorical_columns: set[int],
) -> PolarsDataFrameProtocol:
    """Convert non-categorical columns to numeric.

    Handles all-missing columns by first replacing missing strings with null,
    then casting to Float64 (which handles null gracefully), then filling nulls.

    Args:
        df: Polars DataFrame.
        feature_columns: List of feature column names.
        categorical_columns: Set of categorical feature indices.

    Returns:
        DataFrame with numeric feature columns.
    """
    polars_mod = __import__("polars")
    col_fn: PolarsColFnProtocol = polars_mod.col
    lit_fn: PolarsLitFnProtocol = polars_mod.lit
    when_fn: PolarsWhenFnProtocol = polars_mod.when
    float64_dtype: PolarsDataTypeProtocol = polars_mod.Float64

    result_df = df

    for feat_idx, column_name in enumerate(feature_columns):
        if feat_idx in categorical_columns:
            continue

        col_expr = col_fn(column_name)
        missing_list = list(MISSING_VALUES)
        missing_condition = col_expr.is_null() | col_expr.is_in(missing_list)

        # Replace missing strings with null first, then cast - handles all-missing columns
        nullify_expr: PolarsExprProtocol = (
            when_fn(missing_condition).then(lit_fn(None)).otherwise(col_expr)
        )
        nullified_col = nullify_expr.alias(column_name)
        result_df = result_df.with_columns(nullified_col)

        # Now cast and fill nulls - works even when all values are null
        cast_expr = col_fn(column_name).cast(float64_dtype).fill_null(0.0)
        final_expr = cast_expr.alias(column_name)
        result_df = result_df.with_columns(final_expr)

    return result_df


__all__ = [
    "apply_encodings",
    "build_categorical_encodings",
    "convert_to_numeric",
    "detect_categorical_columns",
]
