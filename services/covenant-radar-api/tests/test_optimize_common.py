"""Tests for worker/_optimize_common.py time-series dataset support.

Tests use dependency injection via worker/_test_hooks to verify actual code paths.
All code paths are tested with strong assertions on actual behavior.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from covenant_ml.datasets import (
    DatasetConfig,
    DatasetMeta,
    DatasetRegistry,
    LoadedDataset,
    TimeSeriesDatasetConfig,
    TimeSeriesDatasetRegistry,
)
from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from numpy.typing import NDArray

from covenant_radar_api.worker import _test_hooks as hooks
from covenant_radar_api.worker._optimize_common import (
    DatasetType,
    get_dataset_type,
    load_any_dataset,
    load_dataset_with_progress,
    load_timeseries_dataset,
    parse_dataset_name,
)

# =============================================================================
# Fake Implementations for Testing
# =============================================================================


def _make_fake_standard_dataset(name: str = "taiwan") -> LoadedDataset:
    """Create fake standard dataset for testing.

    Args:
        name: Dataset name.

    Returns:
        LoadedDataset with synthetic data.
    """
    rng = np.random.default_rng(42)
    x: NDArray[np.float64] = rng.random((100, 10)).astype(np.float64)
    y: NDArray[np.int64] = rng.integers(0, 2, size=100).astype(np.int64)
    n_positive = int(np.sum(y))
    meta: DatasetMeta = {
        "name": name,
        "n_samples": 100,
        "n_features": 10,
        "n_positive": n_positive,
        "n_negative": 100 - n_positive,
        "positive_ratio": n_positive / 100,
        "feature_names": tuple(f"feature_{i}" for i in range(10)),
        "categorical_encodings": (),
    }
    return {"meta": meta, "x": x, "y": y}


def _make_fake_timeseries_dataset(name: str = "kaggle_amex_default") -> LoadedDataset:
    """Create fake time-series dataset for testing.

    Args:
        name: Dataset name.

    Returns:
        LoadedDataset with synthetic aggregated time-series data.
    """
    rng = np.random.default_rng(123)
    # Time-series datasets typically have more features after aggregation
    x: NDArray[np.float64] = rng.random((500, 188)).astype(np.float64)
    y: NDArray[np.int64] = rng.integers(0, 2, size=500).astype(np.int64)
    n_positive = int(np.sum(y))
    meta: DatasetMeta = {
        "name": name,
        "n_samples": 500,
        "n_features": 188,
        "n_positive": n_positive,
        "n_negative": 500 - n_positive,
        "positive_ratio": n_positive / 500,
        "feature_names": tuple(f"ts_feature_{i}" for i in range(188)),
        "categorical_encodings": (),
    }
    return {"meta": meta, "x": x, "y": y}


def _make_fake_standard_config(name: str) -> DatasetConfig:
    """Create fake standard dataset config.

    Args:
        name: Dataset name.

    Returns:
        DatasetConfig for standard dataset.
    """
    return {
        "name": name,
        "display_name": f"Fake {name}",
        "folder": f"{name}_data",
        "file_name": "data.csv",
        "file_format": "csv",
        "encoding": "utf-8",
        "target": {
            "column_name": "target",
            "label_type": "binary_int",
            "positive_values": (1,),
            "negative_values": (0,),
        },
        "exclude_columns": (),
        "n_samples_expected": 100,
        "n_features_expected": 10,
        "positive_class_ratio_expected": 0.3,
    }


def _make_fake_timeseries_config(name: str) -> TimeSeriesDatasetConfig:
    """Create fake time-series dataset config.

    Args:
        name: Dataset name.

    Returns:
        TimeSeriesDatasetConfig for time-series dataset.
    """
    return TimeSeriesDatasetConfig(
        name=name,
        display_name=f"Fake {name} Time Series",
        folder=f"{name}_data",
        file_name="train_data.csv",
        file_format="csv",
        encoding="utf-8",
        target={
            "column_name": "target",
            "label_type": "binary_int",
            "positive_values": (1,),
            "negative_values": (0,),
        },
        exclude_columns=(),
        n_samples_expected=500,
        n_features_expected=188,
        positive_class_ratio_expected=0.26,
        time_series={
            "entity_column": "customer_ID",
            "time_column": "S_2",
            "aggregation": "last",
            "labels_file": "train_labels.csv",
            "labels_entity_column": "customer_ID",
            "include_rank_features": False,
            "include_diff_features": False,
            "include_window_features": False,
            "window_sizes": (),
        },
    )


def _make_fake_standard_registry() -> DatasetRegistry:
    """Create fake standard dataset registry.

    Returns:
        DatasetRegistry with taiwan, us, polish datasets.
    """
    configs = (
        _make_fake_standard_config("taiwan"),
        _make_fake_standard_config("us"),
        _make_fake_standard_config("polish"),
    )
    return DatasetRegistry(configs)


def _make_fake_timeseries_registry() -> TimeSeriesDatasetRegistry:
    """Create fake time-series dataset registry.

    Returns:
        TimeSeriesDatasetRegistry with kaggle_amex_default dataset.
    """
    configs = (_make_fake_timeseries_config("kaggle_amex_default"),)
    return TimeSeriesDatasetRegistry(configs)


# =============================================================================
# Tests for get_dataset_type
# =============================================================================


class TestGetDatasetType:
    """Tests for get_dataset_type function."""

    def test_standard_dataset_returns_standard(self) -> None:
        """Test that standard datasets return 'standard' type."""
        # Save original hooks
        orig_registry = hooks.dataset_registry_factory
        orig_ts_registry = hooks.timeseries_registry_factory

        # Set fake hooks
        hooks.dataset_registry_factory = _make_fake_standard_registry
        hooks.timeseries_registry_factory = _make_fake_timeseries_registry

        try:
            result: DatasetType = get_dataset_type("taiwan")
            assert result == "standard"

            result = get_dataset_type("us")
            assert result == "standard"

            result = get_dataset_type("polish")
            assert result == "standard"
        finally:
            hooks.dataset_registry_factory = orig_registry
            hooks.timeseries_registry_factory = orig_ts_registry

    def test_timeseries_dataset_returns_timeseries(self) -> None:
        """Test that time-series datasets return 'timeseries' type."""
        orig_registry = hooks.dataset_registry_factory
        orig_ts_registry = hooks.timeseries_registry_factory

        hooks.dataset_registry_factory = _make_fake_standard_registry
        hooks.timeseries_registry_factory = _make_fake_timeseries_registry

        try:
            result: DatasetType = get_dataset_type("kaggle_amex_default")
            assert result == "timeseries"
        finally:
            hooks.dataset_registry_factory = orig_registry
            hooks.timeseries_registry_factory = orig_ts_registry

    def test_unknown_dataset_raises_value_error(self) -> None:
        """Test that unknown datasets raise ValueError."""
        orig_registry = hooks.dataset_registry_factory
        orig_ts_registry = hooks.timeseries_registry_factory

        hooks.dataset_registry_factory = _make_fake_standard_registry
        hooks.timeseries_registry_factory = _make_fake_timeseries_registry

        try:
            with pytest.raises(ValueError) as exc_info:
                get_dataset_type("nonexistent")
            assert "nonexistent" in str(exc_info.value)
            # Should list available datasets
            assert "taiwan" in str(exc_info.value) or "amex" in str(exc_info.value)
        finally:
            hooks.dataset_registry_factory = orig_registry
            hooks.timeseries_registry_factory = orig_ts_registry


# =============================================================================
# Tests for parse_dataset_name
# =============================================================================


class TestParseDatasetName:
    """Tests for parse_dataset_name function with both registry types."""

    def test_parse_standard_dataset(self) -> None:
        """Test parsing standard dataset names."""
        orig_registry = hooks.dataset_registry_factory
        orig_ts_registry = hooks.timeseries_registry_factory

        hooks.dataset_registry_factory = _make_fake_standard_registry
        hooks.timeseries_registry_factory = _make_fake_timeseries_registry

        try:
            result: str = parse_dataset_name("taiwan")
            assert result == "taiwan"
        finally:
            hooks.dataset_registry_factory = orig_registry
            hooks.timeseries_registry_factory = orig_ts_registry

    def test_parse_timeseries_dataset(self) -> None:
        """Test parsing time-series dataset names."""
        orig_registry = hooks.dataset_registry_factory
        orig_ts_registry = hooks.timeseries_registry_factory

        hooks.dataset_registry_factory = _make_fake_standard_registry
        hooks.timeseries_registry_factory = _make_fake_timeseries_registry

        try:
            result: str = parse_dataset_name("kaggle_amex_default")
            assert result == "kaggle_amex_default"
        finally:
            hooks.dataset_registry_factory = orig_registry
            hooks.timeseries_registry_factory = orig_ts_registry

    def test_parse_unknown_raises_value_error(self) -> None:
        """Test parsing unknown dataset raises ValueError."""
        orig_registry = hooks.dataset_registry_factory
        orig_ts_registry = hooks.timeseries_registry_factory

        hooks.dataset_registry_factory = _make_fake_standard_registry
        hooks.timeseries_registry_factory = _make_fake_timeseries_registry

        try:
            with pytest.raises(ValueError) as exc_info:
                parse_dataset_name("unknown")
            assert "unknown" in str(exc_info.value)
        finally:
            hooks.dataset_registry_factory = orig_registry
            hooks.timeseries_registry_factory = orig_ts_registry


# =============================================================================
# Tests for load_timeseries_dataset
# =============================================================================


class TestLoadTimeseriesDataset:
    """Tests for load_timeseries_dataset function."""

    def test_loads_timeseries_dataset_via_hook(self) -> None:
        """Test loading time-series dataset uses timeseries_loader hook."""
        orig_ts_registry = hooks.timeseries_registry_factory
        orig_ts_loader = hooks.timeseries_loader

        hooks.timeseries_registry_factory = _make_fake_timeseries_registry

        fake_dataset = _make_fake_timeseries_dataset("kaggle_amex_default")

        def fake_loader(
            config: TimeSeriesDatasetConfig,
            external_dir: Path,
            progress_callback: ProgressCallbackProtocol | None = None,
        ) -> LoadedDataset:
            # Verify config is from registry
            _ = progress_callback  # Available but not used in test
            assert config["name"] == "kaggle_amex_default"
            return fake_dataset

        hooks.timeseries_loader = fake_loader

        try:
            result = load_timeseries_dataset("kaggle_amex_default", Path("/fake/external"))
            assert result["meta"]["name"] == "kaggle_amex_default"
            assert result["meta"]["n_samples"] == 500
            assert result["meta"]["n_features"] == 188
        finally:
            hooks.timeseries_registry_factory = orig_ts_registry
            hooks.timeseries_loader = orig_ts_loader


# =============================================================================
# Tests for load_any_dataset
# =============================================================================


class TestLoadAnyDataset:
    """Tests for load_any_dataset unified loader function."""

    def test_routes_standard_dataset_to_standard_loader(self) -> None:
        """Test standard datasets are routed to standard loader."""
        orig_registry = hooks.dataset_registry_factory
        orig_ts_registry = hooks.timeseries_registry_factory
        orig_loader = hooks.dataset_loader
        orig_ts_loader = hooks.timeseries_loader

        hooks.dataset_registry_factory = _make_fake_standard_registry
        hooks.timeseries_registry_factory = _make_fake_timeseries_registry

        fake_standard = _make_fake_standard_dataset("taiwan")
        standard_loader_called = False

        def fake_standard_loader(
            config: DatasetConfig,
            external_dir: Path,
            progress_callback: ProgressCallbackProtocol | None = None,
        ) -> LoadedDataset:
            nonlocal standard_loader_called
            _ = progress_callback  # Available but not used in test
            standard_loader_called = True
            return fake_standard

        def fake_ts_loader(
            config: TimeSeriesDatasetConfig,
            external_dir: Path,
            progress_callback: ProgressCallbackProtocol | None = None,
        ) -> LoadedDataset:
            _ = progress_callback  # Available but not used in test
            raise AssertionError("Time-series loader should not be called")

        hooks.dataset_loader = fake_standard_loader
        hooks.timeseries_loader = fake_ts_loader

        try:
            result = load_any_dataset("taiwan", Path("/fake"))
            assert standard_loader_called
            assert result["meta"]["name"] == "taiwan"
        finally:
            hooks.dataset_registry_factory = orig_registry
            hooks.timeseries_registry_factory = orig_ts_registry
            hooks.dataset_loader = orig_loader
            hooks.timeseries_loader = orig_ts_loader

    def test_routes_timeseries_dataset_to_timeseries_loader(self) -> None:
        """Test time-series datasets are routed to time-series loader."""
        orig_registry = hooks.dataset_registry_factory
        orig_ts_registry = hooks.timeseries_registry_factory
        orig_loader = hooks.dataset_loader
        orig_ts_loader = hooks.timeseries_loader

        hooks.dataset_registry_factory = _make_fake_standard_registry
        hooks.timeseries_registry_factory = _make_fake_timeseries_registry

        fake_timeseries = _make_fake_timeseries_dataset("kaggle_amex_default")
        timeseries_loader_called = False

        def fake_standard_loader(
            config: DatasetConfig,
            external_dir: Path,
            progress_callback: ProgressCallbackProtocol | None = None,
        ) -> LoadedDataset:
            _ = progress_callback  # Available but not used in test
            raise AssertionError("Standard loader should not be called")

        def fake_ts_loader(
            config: TimeSeriesDatasetConfig,
            external_dir: Path,
            progress_callback: ProgressCallbackProtocol | None = None,
        ) -> LoadedDataset:
            nonlocal timeseries_loader_called
            _ = progress_callback  # Available but not used in test
            timeseries_loader_called = True
            return fake_timeseries

        hooks.dataset_loader = fake_standard_loader
        hooks.timeseries_loader = fake_ts_loader

        try:
            result = load_any_dataset("kaggle_amex_default", Path("/fake"))
            assert timeseries_loader_called
            assert result["meta"]["name"] == "kaggle_amex_default"
            assert result["meta"]["n_features"] == 188
        finally:
            hooks.dataset_registry_factory = orig_registry
            hooks.timeseries_registry_factory = orig_ts_registry
            hooks.dataset_loader = orig_loader
            hooks.timeseries_loader = orig_ts_loader

    def test_unknown_dataset_raises_value_error(self) -> None:
        """Test unknown dataset raises ValueError from get_dataset_type."""
        orig_registry = hooks.dataset_registry_factory
        orig_ts_registry = hooks.timeseries_registry_factory

        hooks.dataset_registry_factory = _make_fake_standard_registry
        hooks.timeseries_registry_factory = _make_fake_timeseries_registry

        try:
            with pytest.raises(ValueError) as exc_info:
                load_any_dataset("nonexistent", Path("/fake"))
            assert "nonexistent" in str(exc_info.value)
        finally:
            hooks.dataset_registry_factory = orig_registry
            hooks.timeseries_registry_factory = orig_ts_registry


# =============================================================================
# Tests for load_dataset_with_progress
# =============================================================================


class TestLoadDatasetWithProgress:
    """Tests for load_dataset_with_progress helper function."""

    def test_delegates_to_load_any_dataset(self) -> None:
        """Test load_dataset_with_progress delegates to load_any_dataset."""
        orig_registry = hooks.dataset_registry_factory
        orig_ts_registry = hooks.timeseries_registry_factory
        orig_loader = hooks.dataset_loader

        hooks.dataset_registry_factory = _make_fake_standard_registry
        hooks.timeseries_registry_factory = _make_fake_timeseries_registry

        fake_standard = _make_fake_standard_dataset("taiwan")

        def fake_standard_loader(
            config: DatasetConfig,
            external_dir: Path,
            progress_callback: ProgressCallbackProtocol | None = None,
        ) -> LoadedDataset:
            _ = progress_callback  # Available but not used in test
            return fake_standard

        hooks.dataset_loader = fake_standard_loader

        try:
            result = load_dataset_with_progress("taiwan", Path("/fake"), None)
            assert result["meta"]["name"] == "taiwan"
            assert result["meta"]["n_samples"] == 100
        finally:
            hooks.dataset_registry_factory = orig_registry
            hooks.timeseries_registry_factory = orig_ts_registry
            hooks.dataset_loader = orig_loader


# =============================================================================
# Tests for Worker Time-Series Hooks
# =============================================================================


class TestWorkerTimeseriesHooks:
    """Tests for worker/_test_hooks.py time-series hooks."""

    def test_real_timeseries_loader_loads_sample(self, tmp_path: Path) -> None:
        """Test _real_timeseries_loader loads sample time-series dataset."""
        from shutil import copyfile

        from covenant_radar_api.worker._test_hooks import _real_timeseries_loader

        # Create a minimal time-series config for testing
        sample_config: TimeSeriesDatasetConfig = TimeSeriesDatasetConfig(
            name="amex_sample",
            display_name="AMEX Sample",
            folder="amex_sample",
            file_name="data.csv",
            file_format="csv",
            encoding="utf-8",
            target={
                "column_name": "target",
                "label_type": "binary_int",
                "positive_values": (1,),
                "negative_values": (0,),
            },
            exclude_columns=(),
            n_samples_expected=10,
            n_features_expected=10,
            positive_class_ratio_expected=0.3,
            time_series={
                "entity_column": "customer_ID",
                "time_column": "S_2",
                "aggregation": "last",
                "labels_file": "labels.csv",
                "labels_entity_column": "customer_ID",
                "include_rank_features": False,
                "include_diff_features": False,
                "include_window_features": False,
                "window_sizes": (),
            },
        )

        # Copy sample fixtures
        external_dir = tmp_path / "external"
        sample_dir = external_dir / "amex_sample"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Path to test fixtures
        fixture_dir = (
            Path(__file__).parent.parent.parent.parent
            / "libs"
            / "covenant_ml"
            / "tests"
            / "datasets"
            / "fixtures"
            / "timeseries_amex_sample"
        )
        copyfile(str(fixture_dir / "data.csv"), str(sample_dir / "data.csv"))
        copyfile(str(fixture_dir / "labels.csv"), str(sample_dir / "labels.csv"))

        # Load using real worker loader
        dataset = _real_timeseries_loader(sample_config, external_dir)

        assert dataset["meta"]["n_samples"] > 0
        assert dataset["meta"]["n_features"] > 0
        assert len(dataset["x"]) == len(dataset["y"])
