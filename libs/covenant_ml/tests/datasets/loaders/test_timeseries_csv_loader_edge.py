"""Tests for TimeSeriesCSVLoader."""

from __future__ import annotations

from pathlib import Path

import pytest

from covenant_ml.datasets.loaders.timeseries_csv_loader import (
    TimeSeriesCSVLoader,
)
from tests.datasets.loaders._timeseries_fixtures import (
    _copy_fixture_to_temp,
    _get_fixtures_dir,
    _make_timeseries_config,
)


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
