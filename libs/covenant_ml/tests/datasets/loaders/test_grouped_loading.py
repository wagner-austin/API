"""Grouped dataset loading: the group column rides beside x and y.

A grouped dataset (rw_matches) names the column that says which rows are one
correlated entity. The loader must keep it out of the feature matrix — a
model reading it would memorize entities — factorize it to stable integer
codes, and round-trip those codes through the parquet cache, because a cache
hit that dropped them would silently re-enable the row-split leak.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.datasets.loaders.arff_loader import ARFFLoader
from covenant_ml.datasets.loaders.csv_loader import CSVLoader
from covenant_ml.datasets.loaders.timeseries_csv_loader import TimeSeriesCSVLoader
from covenant_ml.datasets.types import (
    DatasetConfig,
    TargetColumnSpec,
    TimeSeriesDatasetConfig,
    TimeSeriesSpec,
)

_CSV = (
    "match,frame,worth,won\n"
    "night/a-s1,0,100,1\n"
    "night/a-s1,75,140,1\n"
    "night/b-s2,0,90,0\n"
    "night/b-s2,75,80,0\n"
    "night/a-s1,150,180,1\n"
)


def _codes(groups: NDArray[np.int64] | None) -> list[int]:
    """Group codes as plain ints; a missing array is its own failure."""
    if groups is None:
        raise AssertionError("grouped config must produce group codes")
    out: list[int] = []
    for i in range(len(groups)):
        value: np.int64 = groups[i]
        out.append(int(value))
    return out


def _grouped_config(group_column: str | None = "match") -> DatasetConfig:
    config = DatasetConfig(
        name="grouped_test",
        display_name="Grouped Test",
        folder="grouped",
        file_name="data.csv",
        file_format="csv",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="won",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=(),
        n_samples_expected=5,
        n_features_expected=2,
        positive_class_ratio_expected=0.6,
    )
    if group_column is not None:
        config["group_column"] = group_column
    return config


def _write_dataset(tmp_path: Path) -> Path:
    folder = tmp_path / "grouped"
    folder.mkdir()
    (folder / "data.csv").write_text(_CSV, encoding="utf-8")
    return tmp_path


def test_group_column_is_factorized_and_never_a_feature(tmp_path: Path) -> None:
    external = _write_dataset(tmp_path)
    dataset = CSVLoader().load(_grouped_config(), external)
    assert dataset["meta"]["feature_names"] == ("frame", "worth")
    # Codes in first-appearance order; the third a-s1 row rejoins code 0.
    assert _codes(dataset["groups"]) == [0, 0, 1, 1, 0]
    assert dataset["x"].shape == (5, 2)


def test_groups_survive_the_parquet_cache_round_trip(tmp_path: Path) -> None:
    external = _write_dataset(tmp_path)
    first = CSVLoader().load(_grouped_config(), external)
    second = CSVLoader().load(_grouped_config(), external)
    assert _codes(first["groups"]) == [0, 0, 1, 1, 0]
    assert _codes(second["groups"]) == [0, 0, 1, 1, 0]
    assert np.array_equal(first["x"], second["x"])


def test_an_ungrouped_config_loads_with_groups_none(tmp_path: Path) -> None:
    external = _write_dataset(tmp_path)
    config = _grouped_config(group_column=None)
    config["exclude_columns"] = ("match",)
    dataset = CSVLoader().load(config, external)
    assert dataset["groups"] is None


def test_arff_loader_refuses_a_grouped_config(tmp_path: Path) -> None:
    folder = tmp_path / "grouped"
    folder.mkdir()
    arff = "@relation t\n@attribute a numeric\n@attribute won numeric\n@data\n1,1\n"
    (folder / "data.arff").write_text(arff, encoding="utf-8")
    config = _grouped_config()
    config["file_name"] = "data.arff"
    config["file_format"] = "arff"
    with pytest.raises(ValueError, match="only supported by the CSV loader"):
        ARFFLoader().load(config, tmp_path)


def test_timeseries_loader_refuses_a_grouped_config(tmp_path: Path) -> None:
    config = TimeSeriesDatasetConfig(
        name="grouped_ts",
        display_name="Grouped TS",
        folder="grouped",
        file_name="data.csv",
        file_format="csv",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="won",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=(),
        n_samples_expected=5,
        n_features_expected=2,
        positive_class_ratio_expected=0.6,
        time_series=TimeSeriesSpec(
            entity_column="match",
            time_column="frame",
            aggregation="last",
            labels_file="labels.csv",
            labels_entity_column="match",
            include_rank_features=False,
            include_diff_features=False,
            include_window_features=False,
            window_sizes=(),
        ),
    )
    config["group_column"] = "match"
    with pytest.raises(ValueError, match="only supported by the CSV loader"):
        TimeSeriesCSVLoader().load(config, tmp_path)
