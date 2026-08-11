"""Tests for worker/_optimize_common.py shared utilities and time-series dataset support.

Tests use dependency injection via worker/_test_hooks to verify actual code paths.
All code paths are tested with strong assertions on actual behavior.
"""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile

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
from platform_core.json_utils import JSONTypeError

from covenant_radar_api.worker import _test_hooks as hooks
from covenant_radar_api.worker._optimize_common import (
    DatasetType,
    build_optimization_config,
    get_dataset_type,
    load_any_dataset,
    load_dataset,
    load_dataset_with_progress,
    load_timeseries_dataset,
    optional_int,
    parse_backend_name,
    parse_dataset_name,
    parse_device,
    parse_feature_preset,
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
    return {"meta": meta, "x": x, "y": y, "groups": None}


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
    return {"meta": meta, "x": x, "y": y, "groups": None}


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


# =============================================================================
# Tests for parse_device
# =============================================================================


class TestParseDevice:
    """Tests for parse_device function."""

    def test_parse_device_defaults_to_auto(self) -> None:
        """None input returns 'auto'."""
        assert parse_device(None) == "auto"

    def test_parse_device_accepts_cpu(self) -> None:
        """'cpu' is accepted."""
        assert parse_device("cpu") == "cpu"

    def test_parse_device_accepts_cuda(self) -> None:
        """'cuda' is accepted."""
        assert parse_device("cuda") == "cuda"

    def test_parse_device_accepts_auto(self) -> None:
        """'auto' is accepted."""
        assert parse_device("auto") == "auto"

    def test_parse_device_rejects_invalid_string(self) -> None:
        """Invalid device string raises ValueError."""
        with pytest.raises(ValueError, match="device must be one of"):
            parse_device("tpu")

    def test_parse_device_rejects_non_string(self) -> None:
        """Non-string input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="device must be a string"):
            parse_device(123)


# =============================================================================
# Tests for parse_feature_preset
# =============================================================================


class TestParseFeaturePreset:
    """Tests for parse_feature_preset function."""

    def test_parse_feature_preset_defaults_to_none(self) -> None:
        """None input returns 'none'."""
        assert parse_feature_preset(None) == "none"

    def test_parse_feature_preset_accepts_none(self) -> None:
        """'none' is accepted."""
        assert parse_feature_preset("none") == "none"

    def test_parse_feature_preset_accepts_log_only(self) -> None:
        """'log_only' is accepted."""
        assert parse_feature_preset("log_only") == "log_only"

    def test_parse_feature_preset_accepts_ratios_only(self) -> None:
        """'ratios_only' is accepted."""
        assert parse_feature_preset("ratios_only") == "ratios_only"

    def test_parse_feature_preset_accepts_full(self) -> None:
        """'full' is accepted."""
        assert parse_feature_preset("full") == "full"

    def test_parse_feature_preset_rejects_invalid_string(self) -> None:
        """Invalid feature_preset string raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="feature_preset must be one of"):
            parse_feature_preset("invalid")

    def test_parse_feature_preset_rejects_non_string(self) -> None:
        """Non-string input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="feature_preset must be a string"):
            parse_feature_preset(123)


# =============================================================================
# Tests for parse_backend_name
# =============================================================================


class TestParseBackendName:
    """Tests for parse_backend_name function."""

    def test_parse_backend_name_defaults_to_xgboost(self) -> None:
        """None input returns 'xgboost'."""
        assert parse_backend_name(None) == "xgboost"

    def test_parse_backend_name_accepts_xgboost(self) -> None:
        """'xgboost' is accepted."""
        assert parse_backend_name("xgboost") == "xgboost"

    def test_parse_backend_name_accepts_mlp(self) -> None:
        """'mlp' is accepted."""
        assert parse_backend_name("mlp") == "mlp"

    def test_parse_backend_name_accepts_lstm(self) -> None:
        """'lstm' is accepted."""
        assert parse_backend_name("lstm") == "lstm"

    def test_parse_backend_name_accepts_lightgbm(self) -> None:
        """'lightgbm' is accepted."""
        assert parse_backend_name("lightgbm") == "lightgbm"

    def test_parse_backend_name_accepts_cleargbm(self) -> None:
        """'cleargbm' is accepted."""
        assert parse_backend_name("cleargbm") == "cleargbm"

    def test_parse_backend_name_accepts_logreg(self) -> None:
        """'logreg' is accepted."""
        assert parse_backend_name("logreg") == "logreg"

    def test_parse_backend_name_accepts_random_forest(self) -> None:
        """'random_forest' is accepted."""
        assert parse_backend_name("random_forest") == "random_forest"

    def test_parse_backend_name_rejects_invalid_string(self) -> None:
        """Invalid backend name raises ValueError."""
        with pytest.raises(ValueError, match="backend must be one of"):
            parse_backend_name("invalid")

    def test_parse_backend_name_rejects_non_string(self) -> None:
        """Non-string input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="backend must be a string"):
            parse_backend_name(123)


# =============================================================================
# Tests for optional_int
# =============================================================================


class TestOptionalInt:
    """Tests for optional_int function."""

    def test_optional_int_returns_default_on_missing(self) -> None:
        """optional_int returns default when key is missing."""
        assert optional_int({}, "missing", 10) == 10

    def test_optional_int_returns_value_when_present(self) -> None:
        """optional_int returns value when present."""
        assert optional_int({"val": 20}, "val", 10) == 20

    def test_optional_int_converts_float_to_int(self) -> None:
        """optional_int converts float to int."""
        assert optional_int({"val": 15.5}, "val", 0) == 15

    def test_optional_int_raises_on_invalid_type(self) -> None:
        """optional_int raises JSONTypeError on invalid type."""
        with pytest.raises(JSONTypeError, match="must be a number"):
            optional_int({"val": "string"}, "val", 0)


# =============================================================================
# Tests for build_optimization_config
# =============================================================================


class TestBuildOptimizationConfig:
    """Tests for build_optimization_config function."""

    def test_build_config_with_timeout(self) -> None:
        """build_optimization_config creates config with timeout."""
        config = build_optimization_config(
            n_trials=50,
            timeout_seconds=3600,
            random_state=42,
        )

        assert config["n_trials"] == 50
        assert config["timeout_seconds"] == 3600
        assert config["random_state"] == 42

    def test_build_config_without_timeout(self) -> None:
        """build_optimization_config creates config without timeout."""
        config = build_optimization_config(
            n_trials=25,
            timeout_seconds=None,
            random_state=123,
        )

        assert config["n_trials"] == 25
        assert config["timeout_seconds"] is None
        assert config["random_state"] == 123


# =============================================================================
# Tests for load_dataset (standard)
# =============================================================================


def _copy_real_taiwan(external_root: Path) -> tuple[Path, int, list[str]]:
    """Copy full Taiwan dataset into external_root and return (path, n_rows, feature_names)."""
    src = Path(__file__).parent.parent / "data" / "external" / "taiwan_data" / "data.csv"
    if not src.exists():
        raise FileNotFoundError("Taiwan dataset not found in repository data")
    dst_dir = external_root / "taiwan_data"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "data.csv"
    copyfile(str(src), str(dst))
    header = (dst.read_text(encoding="utf-8").splitlines())[0]
    cols = [c.strip() for c in header.split(",")]
    feature_names = cols[1:]  # all columns after label
    n_rows = sum(1 for _ in dst.open(encoding="utf-8")) - 1
    return dst, n_rows, feature_names


def _copy_real_us(external_root: Path) -> tuple[Path, int, list[str]]:
    """Copy full US dataset into external_root and return (path, n_rows, feature_names)."""
    src = Path(__file__).parent.parent / "data" / "external" / "us_data" / "american_bankruptcy.csv"
    if not src.exists():
        raise FileNotFoundError("US dataset not found in repository data")
    dst_dir = external_root / "us_data"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "american_bankruptcy.csv"
    copyfile(str(src), str(dst))
    header = (dst.read_text(encoding="utf-8-sig").splitlines())[0]
    cols = [c.strip() for c in header.split(",")]
    feature_names = [c for c in cols if c.startswith("X")]
    n_rows = sum(1 for _ in dst.open(encoding="utf-8-sig")) - 1
    return dst, n_rows, feature_names


def _copy_real_polish(external_root: Path) -> tuple[Path, int, list[str]]:
    """Copy full Polish dataset into external_root and return (path, n_rows, feature_names)."""
    src = Path(__file__).parent.parent / "data" / "external" / "polish_data" / "1year.arff"
    if not src.exists():
        raise FileNotFoundError("Polish dataset not found in repository data")
    dst_dir = external_root / "polish_data"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "1year.arff"
    copyfile(str(src), str(dst))
    lines = dst.read_text(encoding="utf-8").splitlines()
    data_idx = -1
    for i, line in enumerate(lines):
        if line.strip().lower() == "@data":
            data_idx = i
            break
    if data_idx < 0:
        raise RuntimeError("ARFF file missing @data section")
    n_rows = len(lines) - (data_idx + 1)
    feature_names: list[str] = []
    for line in lines[: data_idx + 1]:
        s = line.strip()
        if s.lower().startswith("@attribute"):
            parts = s.split()
            if len(parts) >= 2 and parts[1].lower() != "class":
                feature_names.append(parts[1])
    return dst, n_rows, feature_names


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_taiwan_dataset(self, tmp_path: Path) -> None:
        """load_dataset loads Taiwan data successfully."""
        _, n_rows, feature_names = _copy_real_taiwan(tmp_path)
        dataset = load_dataset("taiwan", tmp_path)
        meta = dataset["meta"]

        assert meta["n_samples"] == n_rows
        assert meta["n_features"] == len(feature_names)

    def test_load_us_dataset(self, tmp_path: Path) -> None:
        """load_dataset loads US data successfully."""
        _, n_rows_us, feature_names_us = _copy_real_us(tmp_path)
        dataset = load_dataset("us", tmp_path)
        meta = dataset["meta"]

        assert meta["n_samples"] == n_rows_us
        assert meta["n_features"] == len(feature_names_us)

    def test_load_polish_dataset(self, tmp_path: Path) -> None:
        """load_dataset loads Polish data successfully."""
        _, n_rows_pl, feature_names_pl = _copy_real_polish(tmp_path)
        dataset = load_dataset("polish", tmp_path)
        meta = dataset["meta"]

        assert meta["n_samples"] == n_rows_pl
        assert meta["n_features"] == len(feature_names_pl)

    def test_load_dataset_missing_taiwan(self, tmp_path: Path) -> None:
        """load_dataset raises FileNotFoundError for missing Taiwan data."""
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            load_dataset("taiwan", tmp_path)

    def test_load_dataset_missing_us(self, tmp_path: Path) -> None:
        """load_dataset raises FileNotFoundError for missing US data."""
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            load_dataset("us", tmp_path)

    def test_load_dataset_missing_polish(self, tmp_path: Path) -> None:
        """load_dataset raises FileNotFoundError for missing Polish data."""
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            load_dataset("polish", tmp_path)
