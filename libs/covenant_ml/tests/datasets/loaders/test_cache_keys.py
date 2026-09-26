"""Tests for the parquet cache keys, which must not move with the enums.

Each expected string below is what the loaders hashed while the dataset
vocabularies were Literal strings, so a cache written then is still found.
"""

from __future__ import annotations

from covenant_ml.datasets.loaders._cache_keys import (
    csv_config_hash,
    target_cache_key,
    timeseries_config_hash,
)
from covenant_ml.datasets.loaders.parquet_cache import _compute_config_hash
from covenant_ml.datasets.types import (
    AggregationStrategy,
    DatasetConfig,
    FileEncoding,
    FileFormat,
    LabelType,
    TargetColumnSpec,
    TimeSeriesDatasetConfig,
    TimeSeriesSpec,
)

_TARGET = TargetColumnSpec(
    column_name="Bankrupt?",
    label_type=LabelType.BINARY_INT,
    positive_values=(1,),
    negative_values=(0,),
)
_TARGET_TEXT = (
    "{'column_name': 'Bankrupt?', 'label_type': 'binary_int', "
    "'positive_values': (1,), 'negative_values': (0,)}"
)


def _flat_config() -> DatasetConfig:
    return DatasetConfig(
        name="taiwan",
        display_name="Taiwan",
        folder="taiwan_data",
        file_name="data.csv",
        file_format=FileFormat.CSV,
        encoding=FileEncoding.UTF_8_SIG,
        target=_TARGET,
        exclude_columns=("id",),
        n_samples_expected=10,
        n_features_expected=2,
        positive_class_ratio_expected=0.5,
    )


def test_target_key_spells_the_label_type_as_its_wire_word() -> None:
    """The target reads exactly as the plain-string spec's repr did."""
    assert target_cache_key(_TARGET) == _TARGET_TEXT


def test_csv_hash_is_the_hash_of_the_pre_enum_string() -> None:
    """An ungrouped flat config hashes as it did before the enums."""
    expected = f"taiwan|data.csv|utf-8-sig|{_TARGET_TEXT}|('id',)|None"
    assert csv_config_hash(_flat_config()) == _compute_config_hash(expected)


def test_csv_hash_includes_the_group_column() -> None:
    """A grouped config keys on its group column."""
    config = _flat_config()
    config["group_column"] = "match"
    expected = f"taiwan|data.csv|utf-8-sig|{_TARGET_TEXT}|('id',)|match"
    assert csv_config_hash(config) == _compute_config_hash(expected)


def test_timeseries_hash_is_the_hash_of_the_pre_enum_string() -> None:
    """A time-series config hashes its spec's aggregation as its wire word."""
    config = TimeSeriesDatasetConfig(
        name="amex",
        display_name="AMEX",
        folder="amex",
        file_name="train.csv",
        file_format=FileFormat.CSV,
        encoding=FileEncoding.UTF_8,
        target=_TARGET,
        exclude_columns=(),
        n_samples_expected=10,
        n_features_expected=2,
        positive_class_ratio_expected=0.5,
        time_series=TimeSeriesSpec(
            entity_column="customer_ID",
            time_column="S_2",
            aggregation=AggregationStrategy.STATISTICS,
            labels_file="labels.csv",
            labels_entity_column="customer_ID",
            include_rank_features=True,
            include_diff_features=False,
            include_window_features=True,
            window_sizes=(3, 6),
        ),
    )
    expected = (
        f"amex|train.csv|utf-8|{_TARGET_TEXT}|()|customer_ID|S_2|statistics|"
        "labels.csv|True|False|True|(3, 6)"
    )
    assert timeseries_config_hash(config) == _compute_config_hash(expected)
