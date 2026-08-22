"""Shared fixtures and helpers for test_timeseries_csv_loader splits."""

from __future__ import annotations

from pathlib import Path

from covenant_ml.datasets.types import (
    AggregationStrategy,
    FileEncoding,
    LabelType,
    TargetColumnSpec,
    TimeSeriesDatasetConfig,
    TimeSeriesSpec,
)


def _get_fixtures_dir() -> Path:
    """Get path to test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


def _copy_fixture_to_temp(tmp_path: Path, folder: str) -> Path:
    """Copy fixture folder to temp directory for isolated testing.

    Args:
        tmp_path: Pytest temp directory fixture.
        folder: Fixture folder name to copy.

    Returns:
        Path to temp directory containing fixtures.
    """
    import shutil

    fixtures_dir = _get_fixtures_dir()
    src_folder = fixtures_dir / folder
    dst_folder = tmp_path / folder
    dst_folder.mkdir(parents=True)

    for item in src_folder.iterdir():
        if item.is_file():
            shutil.copy(item, dst_folder / item.name)

    return tmp_path


def _make_timeseries_config(
    name: str = "test_ts",
    folder: str = "timeseries_simple",
    file_name: str = "data.csv",
    target_column: str = "target",
    label_type: LabelType = "binary_int",
    positive_values: tuple[str | int, ...] = (1,),
    negative_values: tuple[str | int, ...] = (0,),
    exclude_columns: tuple[str, ...] = (),
    encoding: FileEncoding = "utf-8",
    n_samples_expected: int = 3,
    n_features_expected: int = 2,
    entity_column: str = "entity_id",
    time_column: str = "timestamp",
    aggregation: AggregationStrategy = "last",
    labels_file: str = "labels.csv",
    labels_entity_column: str = "entity_id",
    include_rank_features: bool = False,
    include_diff_features: bool = False,
    include_window_features: bool = False,
    window_sizes: tuple[int, ...] = (),
) -> TimeSeriesDatasetConfig:
    """Create a test time-series dataset config."""
    return TimeSeriesDatasetConfig(
        name=name,
        display_name=f"Test {name}",
        folder=folder,
        file_name=file_name,
        file_format="csv",
        encoding=encoding,
        target=TargetColumnSpec(
            column_name=target_column,
            label_type=label_type,
            positive_values=positive_values,
            negative_values=negative_values,
        ),
        exclude_columns=exclude_columns,
        n_samples_expected=n_samples_expected,
        n_features_expected=n_features_expected,
        positive_class_ratio_expected=0.5,
        time_series=TimeSeriesSpec(
            entity_column=entity_column,
            time_column=time_column,
            aggregation=aggregation,
            labels_file=labels_file,
            labels_entity_column=labels_entity_column,
            include_rank_features=include_rank_features,
            include_diff_features=include_diff_features,
            include_window_features=include_window_features,
            window_sizes=window_sizes,
        ),
    )
