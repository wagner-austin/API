"""Parquet cache keys for the CSV and time-series loaders.

A loader caches its parsed dataset under a directory named by a hash of the
config fields that change what it would parse. The key spells every field as
its wire word, so a vocabulary member and the string it replaced hash alike
and a cache written before the vocabularies became enums is still found.
"""

from __future__ import annotations

from covenant_ml.datasets.loaders.parquet_cache import _compute_config_hash
from covenant_ml.datasets.types import (
    DatasetConfig,
    TargetColumnSpec,
    TimeSeriesDatasetConfig,
)


def target_cache_key(target: TargetColumnSpec) -> str:
    """Spell a target spec for a cache key, in declaration order.

    Args:
        target: Target column specification.

    Returns:
        The spec as a dict literal whose label type is its wire word.
    """
    fields: dict[str, str | tuple[str | int, ...]] = {
        "column_name": target["column_name"],
        "label_type": target["label_type"].value,
        "positive_values": target["positive_values"],
        "negative_values": target["negative_values"],
    }
    return str(fields)


def csv_config_hash(config: DatasetConfig) -> str:
    """Hash the fields of a flat CSV config that determine its parse.

    Args:
        config: Dataset configuration.

    Returns:
        The cache directory's hash.
    """
    parts = [
        config["name"],
        config["file_name"],
        config["encoding"].value,
        target_cache_key(config["target"]),
        str(config["exclude_columns"]),
        str(config.get("group_column")),
    ]
    return _compute_config_hash("|".join(parts))


def timeseries_config_hash(config: TimeSeriesDatasetConfig) -> str:
    """Hash the fields of a time-series config that determine its parse.

    Args:
        config: Time-series dataset configuration.

    Returns:
        The cache directory's hash.
    """
    ts_spec = config["time_series"]
    parts = [
        config["name"],
        config["file_name"],
        config["encoding"].value,
        target_cache_key(config["target"]),
        str(config["exclude_columns"]),
        ts_spec["entity_column"],
        ts_spec["time_column"],
        ts_spec["aggregation"].value,
        ts_spec["labels_file"],
        str(ts_spec["include_rank_features"]),
        str(ts_spec["include_diff_features"]),
        str(ts_spec["include_window_features"]),
        str(ts_spec["window_sizes"]),
    ]
    return _compute_config_hash("|".join(parts))


__all__ = ["csv_config_hash", "target_cache_key", "timeseries_config_hash"]
