"""Tests for unified DatasetLoader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from covenant_ml.datasets.loader import DatasetLoader, create_dataset_loader
from covenant_ml.datasets.loaders.parquet_cache import (
    _compute_config_hash,
    get_cache_dir,
    invalidate_cache,
)
from covenant_ml.datasets.types import (
    DatasetConfig,
    LoadedDataset,
    RegressionDatasetConfig,
    RegressionTargetSpec,
    TargetColumnSpec,
    TimeSeriesDatasetConfig,
    TimeSeriesSpec,
)


def _get_fixtures_dir() -> Path:
    """Get path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


def _clear_csv_cache(config: DatasetConfig, fixtures_dir: Path) -> None:
    """Clear parquet cache for a CSV dataset config.

    Args:
        config: Dataset configuration.
        fixtures_dir: Path to fixtures directory.
    """
    config_parts = [
        config["name"],
        config["file_name"],
        config["encoding"],
        str(config["target"]),
        str(config["exclude_columns"]),
        str(config.get("group_column")),
    ]
    config_str = "|".join(config_parts)
    config_hash = _compute_config_hash(config_str)
    cache_dir = get_cache_dir(fixtures_dir, config["folder"], config_hash)
    invalidate_cache(cache_dir)


def _make_csv_config(
    name: str = "test",
    folder: str = "small_csv",
    file_name: str = "data.csv",
) -> DatasetConfig:
    """Create a test CSV dataset config."""
    return DatasetConfig(
        name=name,
        display_name=f"Test {name}",
        folder=folder,
        file_name=file_name,
        file_format="csv",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="target",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=(),
        n_samples_expected=5,
        n_features_expected=3,
        positive_class_ratio_expected=0.4,
    )


def _make_arff_config(
    name: str = "test",
    folder: str = "small_arff",
    file_name: str = "data.arff",
) -> DatasetConfig:
    """Create a test ARFF dataset config."""
    return DatasetConfig(
        name=name,
        display_name=f"Test {name}",
        folder=folder,
        file_name=file_name,
        file_format="arff",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="class",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=(),
        n_samples_expected=5,
        n_features_expected=3,
        positive_class_ratio_expected=0.4,
    )


class TestDatasetLoader:
    """Tests for DatasetLoader class."""

    def test_load_csv_format(self) -> None:
        """Load routes CSV format to CSV loader."""
        loader = DatasetLoader()
        config = _make_csv_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["name"] == "test"
        assert result["meta"]["n_samples"] == 5
        assert result["meta"]["n_features"] == 3
        assert result["x"].shape == (5, 3)
        assert result["y"].shape == (5,)

    def test_load_arff_format(self) -> None:
        """Load routes ARFF format to ARFF loader."""
        loader = DatasetLoader()
        config = _make_arff_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["name"] == "test"
        assert result["meta"]["n_samples"] == 5
        assert result["meta"]["n_features"] == 3
        assert result["x"].shape == (5, 3)
        assert result["y"].shape == (5,)

    def test_load_csv_correct_values(self) -> None:
        """Load CSV returns correct feature values."""
        loader = DatasetLoader()
        config = _make_csv_config()
        fixtures_dir = _get_fixtures_dir()

        result: LoadedDataset = loader.load(config, fixtures_dir)
        x_list: list[list[float]] = result["x"].tolist()

        # First row: 1.0, 2.0, 3.0
        assert x_list[0] == [1.0, 2.0, 3.0]

    def test_load_arff_correct_values(self) -> None:
        """Load ARFF returns correct feature values."""
        loader = DatasetLoader()
        config = _make_arff_config()
        fixtures_dir = _get_fixtures_dir()

        result: LoadedDataset = loader.load(config, fixtures_dir)
        x_list: list[list[float]] = result["x"].tolist()

        # First row: 1.0, 2.0, 3.0
        assert x_list[0] == [1.0, 2.0, 3.0]

    def test_load_excel_not_implemented(self) -> None:
        """Load raises ValueError for Excel format (not implemented)."""
        loader = DatasetLoader()
        config = DatasetConfig(
            name="excel_test",
            display_name="Excel Test",
            folder="excel_folder",
            file_name="data.xlsx",
            file_format="excel",
            encoding="utf-8",
            target=TargetColumnSpec(
                column_name="target",
                label_type="binary_int",
                positive_values=(1,),
                negative_values=(0,),
            ),
            exclude_columns=(),
            n_samples_expected=5,
            n_features_expected=3,
            positive_class_ratio_expected=0.4,
        )
        fixtures_dir = _get_fixtures_dir()

        with pytest.raises(ValueError, match="Excel format not yet implemented"):
            loader.load(config, fixtures_dir)

    def test_load_file_not_found_csv(self) -> None:
        """Load raises FileNotFoundError for missing CSV file."""
        loader = DatasetLoader()
        config = _make_csv_config(file_name="nonexistent.csv")
        fixtures_dir = _get_fixtures_dir()

        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            loader.load(config, fixtures_dir)

    def test_load_file_not_found_arff(self) -> None:
        """Load raises FileNotFoundError for missing ARFF file."""
        loader = DatasetLoader()
        config = _make_arff_config(file_name="nonexistent.arff")
        fixtures_dir = _get_fixtures_dir()

        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            loader.load(config, fixtures_dir)

    def test_load_returns_correct_dtypes(self) -> None:
        """Load returns arrays with correct dtypes."""
        loader = DatasetLoader()
        config = _make_csv_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        assert result["x"].dtype == np.float64
        assert result["y"].dtype == np.int64


def _make_timeseries_config(
    name: str = "test_ts",
    folder: str = "timeseries_simple",
    file_name: str = "data.csv",
) -> TimeSeriesDatasetConfig:
    """Create a test time-series dataset config."""
    return TimeSeriesDatasetConfig(
        name=name,
        display_name=f"Test {name}",
        folder=folder,
        file_name=file_name,
        file_format="csv",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="target",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=(),
        n_samples_expected=3,
        n_features_expected=2,
        positive_class_ratio_expected=0.5,
        time_series=TimeSeriesSpec(
            entity_column="entity_id",
            time_column="timestamp",
            aggregation="last",
            labels_file="labels.csv",
            labels_entity_column="entity_id",
            include_rank_features=False,
            include_diff_features=False,
            include_window_features=False,
            window_sizes=(),
        ),
    )


class TestDatasetLoaderTimeSeries:
    """Tests for DatasetLoader time-series functionality."""

    def test_load_timeseries_method(self) -> None:
        """load_timeseries loads time-series data correctly."""
        loader = DatasetLoader()
        config = _make_timeseries_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load_timeseries(config, fixtures_dir)

        assert result["meta"]["name"] == "test_ts"
        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 2
        assert result["x"].shape == (3, 2)
        assert result["y"].shape == (3,)


class TestCreateDatasetLoader:
    """Tests for create_dataset_loader factory."""

    def test_create_dataset_loader_can_load_csv(self) -> None:
        """Factory creates loader that can load CSV datasets."""
        loader = create_dataset_loader()
        config = _make_csv_config()
        fixtures_dir = _get_fixtures_dir()

        # Clear cache to ensure clean test
        _clear_csv_cache(config, fixtures_dir)

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 5
        assert result["meta"]["n_features"] == 3

    def test_create_dataset_loader_can_load_arff(self) -> None:
        """Factory creates loader that can load ARFF datasets."""
        loader = create_dataset_loader()
        config = _make_arff_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 5
        assert result["meta"]["n_features"] == 3

    def test_create_dataset_loader_can_load_timeseries(self) -> None:
        """Factory creates loader that can load time-series datasets."""
        loader = create_dataset_loader()
        config = _make_timeseries_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load_timeseries(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 2


# ---- Regression helpers ----


def _make_regression_config() -> RegressionDatasetConfig:
    """Create a regression dataset config for the fixture."""
    return RegressionDatasetConfig(
        name="test_regression",
        display_name="Test Regression",
        folder="regression_csv",
        file_name="data.csv",
        file_format="csv",
        encoding="utf-8",
        target=RegressionTargetSpec(column_name="target_value"),
        exclude_columns=("entity_id",),
        n_samples_expected=5,
        n_features_expected=3,
        target_mean_expected=0.86,
    )


class TestDatasetLoaderRegression:
    """Tests for DatasetLoader regression loading."""

    def test_load_regression_csv(self) -> None:
        """load_regression loads CSV regression data correctly."""
        loader = DatasetLoader()
        config = _make_regression_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load_regression(config, fixtures_dir)

        assert result["meta"]["name"] == "test_regression"
        assert result["meta"]["n_samples"] == 5
        assert result["meta"]["n_features"] == 3
        assert result["x"].shape == (5, 3)
        assert result["y"].shape == (5,)

    def test_load_regression_unsupported_format_raises(self) -> None:
        """load_regression raises ValueError for non-CSV format."""
        loader = DatasetLoader()
        config = RegressionDatasetConfig(
            name="test_excel",
            display_name="Test Excel",
            folder="regression_csv",
            file_name="data.xlsx",
            file_format="excel",
            encoding="utf-8",
            target=RegressionTargetSpec(column_name="target_value"),
            exclude_columns=(),
            n_samples_expected=5,
            n_features_expected=3,
            target_mean_expected=0.0,
        )
        fixtures_dir = _get_fixtures_dir()

        with pytest.raises(ValueError, match="not yet implemented for format 'excel'"):
            loader.load_regression(config, fixtures_dir)
