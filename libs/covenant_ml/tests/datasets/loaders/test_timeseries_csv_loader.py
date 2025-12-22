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
    LoadProgress,
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


class TestTimeSeriesCSVLoader:
    """Tests for TimeSeriesCSVLoader class."""

    def test_load_aggregation_last(self, tmp_path: Path) -> None:
        """Load with 'last' aggregation takes most recent observation.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(aggregation="last")
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

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

    def test_load_aggregation_first(self, tmp_path: Path) -> None:
        """Load with 'first' aggregation takes oldest observation.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(aggregation="first")
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3

        x_list: list[list[float]] = result["x"].tolist()
        # Entity A: first row is (10.0, 20.0)
        # Entity B: first row is (30.0, 40.0)
        # Entity C: first row is (50.0, 60.0)
        expected_features = {(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)}
        actual_features = {tuple(row) for row in x_list}
        assert actual_features == expected_features

    def test_load_aggregation_mean(self, tmp_path: Path) -> None:
        """Load with 'mean' aggregation computes mean of features.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(aggregation="mean")
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3

        x_list: list[list[float]] = result["x"].tolist()
        # Entity A (3 rows): mean of [(10,20), (11,21), (12,22)] = (11, 21)
        # Entity B (2 rows): mean of [(30,40), (31,41)] = (30.5, 40.5)
        # Entity C (4 rows): mean of [(50,60), (51,61), (52,62), (53,63)] = (51.5, 61.5)
        expected_features = {(11.0, 21.0), (30.5, 40.5), (51.5, 61.5)}
        actual_features = {tuple(row) for row in x_list}
        assert actual_features == expected_features

    def test_load_aggregation_statistics(self, tmp_path: Path) -> None:
        """Load with 'statistics' aggregation creates mean/std/min/max features.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(aggregation="statistics")
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

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

    def test_load_with_categorical_columns(self, tmp_path: Path) -> None:
        """Load handles categorical columns correctly.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            folder="timeseries_categorical",
            aggregation="last",
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_categorical")

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

    def test_load_with_progress_callback(self, tmp_path: Path) -> None:
        """Load reports progress via callback including aggregation phase.

        Uses tmp_path to avoid cache race conditions with parallel tests.

        Args:
            tmp_path: Pytest fixture providing isolated temp directory.
        """
        import shutil

        # Copy fixture files to temp directory to avoid cache conflicts
        fixtures_dir = _get_fixtures_dir()
        src_folder = fixtures_dir / "timeseries_simple"
        dst_folder = tmp_path / "timeseries_simple"
        dst_folder.mkdir(parents=True)

        for file_name in ["data.csv", "labels.csv"]:
            shutil.copy(src_folder / file_name, dst_folder / file_name)

        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config()

        progress_updates: list[LoadProgress] = []

        def capture(progress: LoadProgress) -> None:
            progress_updates.append(progress)

        result: LoadedDataset = loader.load(config, tmp_path, progress_callback=capture)

        # Should have progress updates
        assert len(progress_updates) >= 4

        # Check for aggregating phase (start and complete)
        aggregating_updates = [p for p in progress_updates if p["phase"] == "aggregating"]
        assert len(aggregating_updates) == 2
        # First should be start (0%)
        assert aggregating_updates[0]["percent_complete"] == 0.0
        assert "Aggregating" in aggregating_updates[0]["message"]
        # Second should be complete (100%)
        assert aggregating_updates[1]["percent_complete"] == 100.0
        assert "Aggregated" in aggregating_updates[1]["message"]

        # Result should still be correct
        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 2


class TestTimeSeriesCSVLoaderAggregationDetails:
    """Detailed tests for aggregation behavior."""

    def test_statistics_aggregation_values(self, tmp_path: Path) -> None:
        """Verify statistics aggregation computes correct values.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(aggregation="statistics")
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

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

    def test_load_empty_labels_file_raises(self) -> None:
        """Load raises ValueError for labels file with no data rows."""
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            folder="timeseries_empty_labels",
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

    def test_load_categorical_with_missing_values(self, tmp_path: Path) -> None:
        """Load handles categorical columns with missing values.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            folder="timeseries_categorical_missing",
            aggregation="last",
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_categorical_missing")

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 2

        # Check categorical encoding has missing category
        encodings = result["meta"]["categorical_encodings"]
        assert len(encodings) == 1
        cat_enc = encodings[0]
        mapping_dict = dict(cat_enc["mapping"])
        # Should have _MISSING_ as first category
        assert "_MISSING_" in mapping_dict
        assert mapping_dict["_MISSING_"] == 0

    def test_load_large_file_triggers_sampling(self, tmp_path: Path) -> None:
        """Load with >1000 rows triggers sampling for categorical detection.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        # Create large data file with 1100 rows
        folder = tmp_path / "large_data"
        folder.mkdir()

        # Write data file with >1000 rows
        with open(folder / "data.csv", "w") as f:
            f.write("entity_id,timestamp,feature_1,feature_cat\n")
            for i in range(1100):
                entity = chr(65 + (i % 10))  # A-J
                cat_val = ["LOW", "HIGH", "MEDIUM"][i % 3]
                f.write(f"{entity},{i},{i * 1.5},{cat_val}\n")

        # Write labels file
        with open(folder / "labels.csv", "w") as f:
            f.write("entity_id,target\n")
            for c in "ABCDEFGHIJ":
                f.write(f"{c},{1 if c in 'ACEGI' else 0}\n")

        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            folder="large_data",
            aggregation="last",
        )

        result = loader.load(config, tmp_path)

        # Should load successfully with sampling
        assert result["meta"]["n_samples"] == 10
        assert result["meta"]["n_features"] == 2
        # Check categorical was detected
        assert len(result["meta"]["categorical_encodings"]) == 1

    def test_load_categorical_mean_aggregation(self, tmp_path: Path) -> None:
        """Load handles categorical columns with mean aggregation.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            folder="timeseries_categorical",
            aggregation="mean",
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_categorical")

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        # Categorical column should be included in mean (encoded values averaged)
        assert result["meta"]["n_features"] == 2

    def test_load_categorical_statistics_aggregation(self, tmp_path: Path) -> None:
        """Load handles categorical columns with statistics aggregation.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            folder="timeseries_categorical",
            aggregation="statistics",
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_categorical")

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        # 2 features * 4 stats = 8 output features
        assert result["meta"]["n_features"] == 8

    def test_load_all_missing_feature_values_mean(self, tmp_path: Path) -> None:
        """Load handles feature columns with all missing values in mean aggregation.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            folder="timeseries_all_missing",
            aggregation="mean",
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_all_missing")

        result = loader.load(config, fixtures_dir)

        # Should handle gracefully (zeros for missing)
        assert result["meta"]["n_samples"] == 2

    def test_load_all_missing_feature_values_statistics(self, tmp_path: Path) -> None:
        """Load handles feature columns with all missing values in statistics aggregation.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            folder="timeseries_all_missing",
            aggregation="statistics",
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_all_missing")

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


class TestTimeSeriesCSVLoaderAMEXSample:
    """Tests for loading AMEX-style data."""

    def test_load_amex_sample(self, tmp_path: Path) -> None:
        """Load AMEX sample data with real column structure.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            name="amex_sample",
            folder="timeseries_amex_sample",
            entity_column="customer_ID",
            time_column="S_2",
            aggregation="last",
            labels_file="labels.csv",
            labels_entity_column="customer_ID",
            n_samples_expected=4,
            n_features_expected=188,
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_amex_sample")

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 4
        assert result["meta"]["n_features"] == 188
        # All 4 customers in sample have target=0
        assert result["meta"]["n_positive"] == 0
        assert result["meta"]["n_negative"] == 4


class TestCreateTimeseriesCSVLoader:
    """Tests for create_timeseries_csv_loader factory."""

    def test_factory_creates_working_loader(self, tmp_path: Path) -> None:
        """Factory creates loader that can successfully load data.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = create_timeseries_csv_loader()
        config = _make_timeseries_config()
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 2


class TestTimeSeriesRankDiffFeatures:
    """Tests for rank and diff feature computation in loader."""

    def test_load_with_rank_features(self, tmp_path: Path) -> None:
        """Load adds rank features when enabled.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        # Base: 2 features, rank adds 2 features (one per base feature)
        config = _make_timeseries_config(
            n_features_expected=4,
            include_rank_features=True,
            include_diff_features=False,
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 4
        # Check feature names include rank suffix
        feature_names = result["meta"]["feature_names"]
        assert "feature_1_rank" in feature_names
        assert "feature_2_rank" in feature_names

    def test_load_with_diff_features(self, tmp_path: Path) -> None:
        """Load adds diff features when enabled.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        # Base: 2 features, diff adds 10 features (5 aggs per base feature)
        config = _make_timeseries_config(
            n_features_expected=12,
            include_rank_features=False,
            include_diff_features=True,
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 12
        # Check feature names include diff suffixes
        feature_names = result["meta"]["feature_names"]
        assert "feature_1_diff_mean" in feature_names
        assert "feature_2_diff_last" in feature_names

    def test_load_with_both_rank_and_diff_features(self, tmp_path: Path) -> None:
        """Load adds both rank and diff features when both enabled.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        # Base: 2 features, rank adds 2, diff adds 10 -> total 14
        config = _make_timeseries_config(
            n_features_expected=14,
            include_rank_features=True,
            include_diff_features=True,
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 14
        feature_names = result["meta"]["feature_names"]
        # Both rank and diff features present
        assert "feature_1_rank" in feature_names
        assert "feature_1_diff_mean" in feature_names

    def test_rank_features_values_in_range(self, tmp_path: Path) -> None:
        """Rank features are in [0, 1] range.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        import numpy as np

        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            include_rank_features=True,
            include_diff_features=False,
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

        result = loader.load(config, fixtures_dir)

        # Rank features are columns 2 and 3 (after base features 0 and 1)
        x_array = result["x"]
        rank_cols = x_array[:, 2:]
        assert np.all(rank_cols >= 0.0)
        assert np.all(rank_cols <= 1.0)


class TestTimeSeriesWindowFeatures:
    """Tests for window feature computation in loader."""

    def test_load_with_window_features(self, tmp_path: Path) -> None:
        """Load adds window features when enabled.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        # Base: 2 features, window size 2 adds 2*4=8 features
        config = _make_timeseries_config(
            n_features_expected=10,
            include_rank_features=False,
            include_diff_features=False,
            include_window_features=True,
            window_sizes=(2,),
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 10
        # Check feature names include window suffix
        feature_names = result["meta"]["feature_names"]
        assert "feature_1_last2_mean" in feature_names
        assert "feature_2_last2_max" in feature_names

    def test_load_with_multiple_window_sizes(self, tmp_path: Path) -> None:
        """Load adds window features for multiple window sizes.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        # Base: 2 features, window sizes (2, 3) adds 2*4*2=16 features
        config = _make_timeseries_config(
            n_features_expected=18,
            include_rank_features=False,
            include_diff_features=False,
            include_window_features=True,
            window_sizes=(2, 3),
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 18
        feature_names = result["meta"]["feature_names"]
        # Both window sizes should be present
        assert "feature_1_last2_mean" in feature_names
        assert "feature_1_last3_mean" in feature_names

    def test_load_with_all_feature_types(self, tmp_path: Path) -> None:
        """Load adds rank, diff, and window features when all enabled.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        # Base: 2 features
        # Rank: 2 features
        # Diff: 2*5=10 features
        # Window (size 2): 2*4=8 features
        # Total: 2 + 2 + 10 + 8 = 22
        config = _make_timeseries_config(
            n_features_expected=22,
            include_rank_features=True,
            include_diff_features=True,
            include_window_features=True,
            window_sizes=(2,),
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 22
        feature_names = result["meta"]["feature_names"]
        # All feature types present
        assert "feature_1_rank" in feature_names
        assert "feature_1_diff_mean" in feature_names
        assert "feature_1_last2_mean" in feature_names

    def test_load_window_features_disabled_with_empty_sizes(self, tmp_path: Path) -> None:
        """Load skips window features when flag True but sizes empty.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            n_features_expected=2,
            include_rank_features=False,
            include_diff_features=False,
            include_window_features=True,
            window_sizes=(),  # Empty sizes
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

        result = loader.load(config, fixtures_dir)

        # No window features added because sizes is empty
        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 2

    def test_load_window_features_disabled_flag_false(self, tmp_path: Path) -> None:
        """Load skips window features when flag is False.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            n_features_expected=2,
            include_rank_features=False,
            include_diff_features=False,
            include_window_features=False,
            window_sizes=(2, 3),  # Sizes provided but flag is False
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

        result = loader.load(config, fixtures_dir)

        # No window features added because flag is False
        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 2
