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
    LoadedDataset,
    LoadProgress,
)
from tests.datasets.loaders._timeseries_fixtures import (
    _copy_fixture_to_temp,
    _get_fixtures_dir,
    _make_timeseries_config,
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
