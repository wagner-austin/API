"""Tests for RegressionCSVLoader.

Uses local fixture data (tests/datasets/fixtures/regression_csv/data.csv).
Integration tests with real datasets live in services/covenant-radar-api/tests/.
"""

from __future__ import annotations

import math as _math
from pathlib import Path

import numpy as np
import pytest

from covenant_ml.datasets.loaders._regression_csv import (
    RegressionCSVLoader,
    create_regression_csv_loader,
)
from covenant_ml.datasets.types import (
    LoadProgress,
    RegressionDatasetConfig,
    RegressionTargetSpec,
)


def _get_fixtures_dir() -> Path:
    """Get path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


def _make_regression_config(
    name: str = "test_regression",
    folder: str = "regression_csv",
    file_name: str = "data.csv",
    target_column: str = "target_value",
    exclude_columns: tuple[str, ...] = ("entity_id",),
) -> RegressionDatasetConfig:
    """Create a test regression dataset config for the fixture.

    Args:
        name: Dataset name.
        folder: Fixture subfolder.
        file_name: CSV file name.
        target_column: Target column name.
        exclude_columns: Columns to exclude.

    Returns:
        RegressionDatasetConfig for testing.
    """
    return RegressionDatasetConfig(
        name=name,
        display_name=f"Test {name}",
        folder=folder,
        file_name=file_name,
        file_format="csv",
        encoding="utf-8",
        target=RegressionTargetSpec(column_name=target_column),
        exclude_columns=exclude_columns,
        n_samples_expected=5,
        n_features_expected=3,
        target_mean_expected=0.86,
    )


class TestRegressionCSVLoader:
    """Tests for RegressionCSVLoader with fixture data."""

    def test_load_returns_regression_loaded_dataset(self) -> None:
        """load() returns RegressionLoadedDataset with correct types."""
        loader = RegressionCSVLoader()
        config = _make_regression_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        assert result["x"].dtype == np.float64
        assert result["y"].dtype == np.float64
        assert result["meta"]["name"] == "test_regression"

    def test_sample_count(self) -> None:
        """Loader produces correct number of samples."""
        loader = RegressionCSVLoader()
        config = _make_regression_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 5
        assert int(result["x"].shape[0]) == 5
        assert int(result["y"].shape[0]) == 5

    def test_feature_count(self) -> None:
        """Loader produces correct number of features (excludes entity_id and target)."""
        loader = RegressionCSVLoader()
        config = _make_regression_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        # 5 columns total - entity_id (excluded) - target_value (target) = 3 features
        assert result["meta"]["n_features"] == 3
        assert int(result["x"].shape[1]) == 3

    def test_feature_matrix_shape(self) -> None:
        """Feature matrix has shape (n_samples, n_features)."""
        loader = RegressionCSVLoader()
        config = _make_regression_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        assert result["x"].shape == (5, 3)

    def test_excludes_columns(self) -> None:
        """Excluded columns do not appear in features."""
        loader = RegressionCSVLoader()
        config = _make_regression_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        feature_names = result["meta"]["feature_names"]
        assert "entity_id" not in feature_names
        assert "target_value" not in feature_names
        assert feature_names == ("feature_1", "feature_2", "feature_3")

    def test_target_values_correct(self) -> None:
        """Target values are parsed as continuous floats."""
        loader = RegressionCSVLoader()
        config = _make_regression_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        y = result["y"]
        expected = [0.5, 1.2, -0.3, 2.1, 0.8]
        for i in range(5):
            assert abs(float(y.flat[i]) - expected[i]) < 1e-10

    def test_feature_values_correct(self) -> None:
        """Feature values are parsed correctly from CSV."""
        loader = RegressionCSVLoader()
        config = _make_regression_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        x = result["x"]
        # First row: feature_1=1.0, feature_2=2.0, feature_3=3.0
        assert abs(float(x.flat[0]) - 1.0) < 1e-10
        assert abs(float(x.flat[1]) - 2.0) < 1e-10
        assert abs(float(x.flat[2]) - 3.0) < 1e-10

    def test_no_nan_in_features(self) -> None:
        """Feature matrix has no NaN values."""
        loader = RegressionCSVLoader()
        config = _make_regression_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        x = result["x"]
        n_elements = int(x.shape[0]) * int(x.shape[1])
        nan_found = False
        for i in range(n_elements):
            val: float = float(x.flat[i])
            if val != val:  # NaN check without np.isnan
                nan_found = True
        assert not nan_found

    def test_meta_has_target_stats(self) -> None:
        """Metadata contains target distribution statistics."""
        loader = RegressionCSVLoader()
        config = _make_regression_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)
        meta = result["meta"]

        # Verify all target stat fields have sensible values
        assert meta["target_min"] <= meta["target_mean"] <= meta["target_max"]
        assert meta["target_std"] >= 0.0

    def test_meta_target_stats_correct(self) -> None:
        """Target statistics are computed correctly from fixture data."""
        loader = RegressionCSVLoader()
        config = _make_regression_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)
        meta = result["meta"]

        # Fixture target values: [0.5, 1.2, -0.3, 2.1, 0.8]
        # Mean = (0.5 + 1.2 + (-0.3) + 2.1 + 0.8) / 5 = 4.3 / 5 = 0.86
        expected_mean = 0.86
        # Variance = sum((xi - mean)^2) / N
        sq_diffs = (0.5 - 0.86) ** 2 + (1.2 - 0.86) ** 2 + (-0.3 - 0.86) ** 2
        sq_diffs += (2.1 - 0.86) ** 2 + (0.8 - 0.86) ** 2
        expected_std: float = _math.sqrt(sq_diffs / 5.0)
        expected_min = -0.3
        expected_max = 2.1

        assert abs(meta["target_mean"] - expected_mean) < 1e-10
        assert abs(meta["target_std"] - expected_std) < 1e-10
        assert abs(meta["target_min"] - expected_min) < 1e-10
        assert abs(meta["target_max"] - expected_max) < 1e-10

    def test_with_progress_callback(self) -> None:
        """Progress callback receives updates during loading."""
        loader = RegressionCSVLoader()
        config = _make_regression_config()
        fixtures_dir = _get_fixtures_dir()

        progress_updates: list[LoadProgress] = []

        def on_progress(progress: LoadProgress) -> None:
            progress_updates.append(progress)

        result = loader.load(config, fixtures_dir, progress_callback=on_progress)

        assert result["meta"]["n_samples"] == 5
        # At least the encoding start + encoding complete updates
        encoding_updates = [p for p in progress_updates if p["phase"] == "encoding"]
        assert encoding_updates[0]["rows_processed"] == 0
        assert encoding_updates[-1]["rows_processed"] == 5

    def test_file_not_found_raises(self) -> None:
        """FileNotFoundError raised when dataset file doesn't exist."""
        loader = RegressionCSVLoader()
        config = _make_regression_config(folder="nonexistent")
        fixtures_dir = _get_fixtures_dir()

        with pytest.raises(FileNotFoundError):
            loader.load(config, fixtures_dir)

    def test_no_exclude_columns(self) -> None:
        """Loader works with no excluded columns (entity_id becomes feature)."""
        loader = RegressionCSVLoader()
        config = _make_regression_config(exclude_columns=())
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        # 5 columns - target_value = 4 features (including entity_id as categorical)
        assert result["meta"]["n_features"] == 4


class TestCreateRegressionCSVLoader:
    """Tests for factory function."""

    def test_factory_returns_working_loader(self) -> None:
        """create_regression_csv_loader returns a working loader."""
        loader = create_regression_csv_loader()
        config = _make_regression_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)
        assert result["meta"]["n_samples"] == 5


class TestGroupColumn:
    """Group-column extraction on regression datasets."""

    def _grouped_config(self) -> RegressionDatasetConfig:
        """Return a config naming ``match`` as the group column."""
        return RegressionDatasetConfig(
            name="test_grouped_regression",
            display_name="Test Grouped Regression",
            folder="regression_grouped",
            file_name="data.csv",
            file_format="csv",
            encoding="utf-8",
            target=RegressionTargetSpec(column_name="target_value"),
            exclude_columns=(),
            n_samples_expected=8,
            n_features_expected=2,
            target_mean_expected=0.0,
            group_column="match",
        )

    def test_groups_factorize_in_first_appearance_order(self) -> None:
        """Rows of one match share a code; codes count up from zero."""
        loader = RegressionCSVLoader()
        result = loader.load(self._grouped_config(), _get_fixtures_dir())
        groups = result["groups"]
        if groups is None:
            raise AssertionError("grouped config must produce group codes")
        codes: list[int] = []
        for i in range(len(groups)):
            value: np.int64 = groups[i]
            codes.append(int(value))
        assert codes == [0, 0, 0, 1, 1, 2, 2, 2]

    def test_group_column_is_never_a_feature(self) -> None:
        """The group column is excluded from the feature matrix."""
        loader = RegressionCSVLoader()
        result = loader.load(self._grouped_config(), _get_fixtures_dir())
        assert result["meta"]["feature_names"] == ("frame", "army")
        assert result["meta"]["n_features"] == 2

    def test_no_group_column_yields_none(self) -> None:
        """Row-independent datasets carry groups=None."""
        loader = RegressionCSVLoader()
        result = loader.load(_make_regression_config(), _get_fixtures_dir())
        assert result["groups"] is None
