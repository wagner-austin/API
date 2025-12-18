"""Tests for TimeSeriesCSVLoader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from covenant_ml.datasets.loaders.timeseries_csv_loader import (
    TimeSeriesCSVLoader,
    create_timeseries_csv_loader,
)
from covenant_ml.datasets.types import (
    AggregationStrategy,
    FileEncoding,
    LabelType,
    LoadedDataset,
    TargetColumnSpec,
    TimeSeriesDatasetConfig,
    TimeSeriesSpec,
)


def _get_fixtures_dir() -> Path:
    """Get path to test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


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
        ),
    )


class TestTimeSeriesCSVLoader:
    """Tests for TimeSeriesCSVLoader class."""

    def test_load_aggregation_last(self) -> None:
        """Load with 'last' aggregation takes most recent observation."""
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(aggregation="last")
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        # 3 entities: A, B, C
        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 2

        # Check that we got the last row for each entity
        x_list: list[list[float]] = result["x"].tolist()
        # Entity A: last row is (12.0, 22.0)
        # Entity B: last row is (31.0, 41.0)
        # Entity C: last row is (53.0, 63.0)
        # Order depends on dict iteration, but values should be correct
        expected_features = {(12.0, 22.0), (31.0, 41.0), (53.0, 63.0)}
        actual_features = {tuple(row) for row in x_list}
        assert actual_features == expected_features

    def test_load_aggregation_first(self) -> None:
        """Load with 'first' aggregation takes oldest observation."""
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(aggregation="first")
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3

        x_list: list[list[float]] = result["x"].tolist()
        # Entity A: first row is (10.0, 20.0)
        # Entity B: first row is (30.0, 40.0)
        # Entity C: first row is (50.0, 60.0)
        expected_features = {(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)}
        actual_features = {tuple(row) for row in x_list}
        assert actual_features == expected_features

    def test_load_aggregation_mean(self) -> None:
        """Load with 'mean' aggregation computes mean of features."""
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(aggregation="mean")
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3

        x_list: list[list[float]] = result["x"].tolist()
        # Entity A (3 rows): mean of [(10,20), (11,21), (12,22)] = (11, 21)
        # Entity B (2 rows): mean of [(30,40), (31,41)] = (30.5, 40.5)
        # Entity C (4 rows): mean of [(50,60), (51,61), (52,62), (53,63)] = (51.5, 61.5)
        expected_features = {(11.0, 21.0), (30.5, 40.5), (51.5, 61.5)}
        actual_features = {tuple(row) for row in x_list}
        assert actual_features == expected_features

    def test_load_aggregation_statistics(self) -> None:
        """Load with 'statistics' aggregation creates mean/std/min/max features."""
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(aggregation="statistics")
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        # 2 base features * 4 stats = 8 output features
        assert result["meta"]["n_features"] == 8

        # Check feature names
        expected_names = (
            "feature_1_mean",
            "feature_1_std",
            "feature_1_min",
            "feature_1_max",
            "feature_2_mean",
            "feature_2_std",
            "feature_2_min",
            "feature_2_max",
        )
        assert result["meta"]["feature_names"] == expected_names

    def test_load_returns_correct_labels(self) -> None:
        """Load returns correct labels from labels file."""
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(aggregation="last")
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        # Labels: A=1, B=0, C=1
        y_list: list[int] = result["y"].tolist()
        # Count positives and negatives
        assert sum(y_list) == 2  # A and C are positive
        assert len(y_list) - sum(y_list) == 1  # B is negative

    def test_load_returns_correct_array_types(self) -> None:
        """Load returns correct numpy array types."""
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        assert result["x"].dtype == np.float64
        assert result["y"].dtype == np.int64
        assert result["x"].shape == (3, 2)
        assert result["y"].shape == (3,)

    def test_load_file_not_found_raises(self) -> None:
        """Load raises FileNotFoundError for missing file."""
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(file_name="nonexistent.csv")
        fixtures_dir = _get_fixtures_dir()

        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            loader.load(config, fixtures_dir)

    def test_load_labels_file_not_found_raises(self) -> None:
        """Load raises FileNotFoundError for missing labels file."""
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(labels_file="nonexistent_labels.csv")
        fixtures_dir = _get_fixtures_dir()

        with pytest.raises(FileNotFoundError, match="Labels file not found"):
            loader.load(config, fixtures_dir)

    def test_load_missing_column_raises(self) -> None:
        """Load raises ValueError for missing entity column."""
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(entity_column="nonexistent_column")
        fixtures_dir = _get_fixtures_dir()

        with pytest.raises(ValueError, match="Column 'nonexistent_column' not found"):
            loader.load(config, fixtures_dir)

    def test_load_with_categorical_columns(self) -> None:
        """Load handles categorical columns correctly."""
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            folder="timeseries_categorical",
            aggregation="last",
        )
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 2

        # Check categorical encodings are captured
        encodings = result["meta"]["categorical_encodings"]
        assert len(encodings) == 1
        assert encodings[0]["column_name"] == "feature_cat"

    def test_load_positive_ratio_calculated(self) -> None:
        """Load calculates positive class ratio correctly."""
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        # 2 positive out of 3 (A=1, B=0, C=1)
        expected_ratio = 2 / 3
        assert result["meta"]["positive_ratio"] == pytest.approx(expected_ratio, abs=0.01)


class TestTimeSeriesCSVLoaderAggregationDetails:
    """Detailed tests for aggregation behavior."""

    def test_statistics_aggregation_values(self) -> None:
        """Verify statistics aggregation computes correct values."""
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(aggregation="statistics")
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)
        x_list: list[list[float]] = result["x"].tolist()

        # Find entity A's row (has values 10,11,12 for feature_1)
        # mean=11, std=0.816..., min=10, max=12
        found_entity_a = False
        for row in x_list:
            # Check if this is entity A (feature_1_mean should be 11)
            if abs(row[0] - 11.0) < 0.01:
                assert row[0] == pytest.approx(11.0, abs=0.01)  # mean
                assert row[1] == pytest.approx(0.816, abs=0.01)  # std
                assert row[2] == pytest.approx(10.0, abs=0.01)  # min
                assert row[3] == pytest.approx(12.0, abs=0.01)  # max
                found_entity_a = True
                break

        if not found_entity_a:
            pytest.fail("Entity A not found in results")


class TestTimeSeriesCSVLoaderEdgeCases:
    """Tests for edge cases and error handling."""

    def test_load_empty_file_raises(self) -> None:
        """Load raises ValueError for CSV file with no data rows."""
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            folder="empty_csv",
            file_name="data.csv",
        )
        fixtures_dir = _get_fixtures_dir()

        with pytest.raises(ValueError, match="No data rows found"):
            loader.load(config, fixtures_dir)

    def test_load_missing_entity_labels_raises(self) -> None:
        """Load raises ValueError when entities have no labels."""
        loader = TimeSeriesCSVLoader()
        # Use the simple timeseries data but with a labels file that's missing entities
        config = _make_timeseries_config(
            folder="timeseries_missing_labels",
        )
        fixtures_dir = _get_fixtures_dir()

        with pytest.raises(ValueError, match="Missing labels for"):
            loader.load(config, fixtures_dir)

    def test_load_no_labels_file_raises(self) -> None:
        """Load raises ValueError when labels_file is empty string."""
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            labels_file="",  # Empty string triggers the error
        )
        fixtures_dir = _get_fixtures_dir()

        with pytest.raises(ValueError, match="must have labels_file specified"):
            loader.load(config, fixtures_dir)

    def test_load_string_time_column_sorting(self) -> None:
        """Load correctly sorts string time values lexicographically."""
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            folder="timeseries_string_time",
            aggregation="last",
        )
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        # Should have aggregated correctly based on lexicographic order
        assert result["meta"]["n_samples"] == 2

    def test_load_missing_time_value_sorting(self) -> None:
        """Load handles missing values in time column during sorting."""
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            folder="timeseries_missing_time",
            aggregation="last",
        )
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 2

    def test_load_categorical_mean_aggregation(self) -> None:
        """Load handles categorical columns with mean aggregation."""
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            folder="timeseries_categorical",
            aggregation="mean",
        )
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        # Categorical column should be included in mean (encoded values averaged)
        assert result["meta"]["n_features"] == 2

    def test_load_categorical_statistics_aggregation(self) -> None:
        """Load handles categorical columns with statistics aggregation."""
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            folder="timeseries_categorical",
            aggregation="statistics",
        )
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        # 2 features * 4 stats = 8 output features
        assert result["meta"]["n_features"] == 8

    def test_load_all_missing_feature_values_mean(self) -> None:
        """Load handles feature columns with all missing values in mean aggregation."""
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            folder="timeseries_all_missing",
            aggregation="mean",
        )
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        # Should handle gracefully (zeros for missing)
        assert result["meta"]["n_samples"] == 2

    def test_load_all_missing_feature_values_statistics(self) -> None:
        """Load handles feature columns with all missing values in statistics aggregation."""
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            folder="timeseries_all_missing",
            aggregation="statistics",
        )
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        # Should handle gracefully (zeros for missing stats)
        assert result["meta"]["n_samples"] == 2
        # 2 features * 4 stats = 8 output features
        assert result["meta"]["n_features"] == 8
        # All values should be 0.0 since all inputs are missing
        x_list: list[list[float]] = result["x"].tolist()
        for row in x_list:
            for val in row:
                assert val == 0.0


class TestCreateTimeseriesCSVLoader:
    """Tests for create_timeseries_csv_loader factory."""

    def test_factory_creates_working_loader(self) -> None:
        """Factory creates loader that can successfully load data."""
        loader = create_timeseries_csv_loader()
        config = _make_timeseries_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 2
