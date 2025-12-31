"""Shared fixtures for replay script tests."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from covenant_ml.datasets.types import (
    DatasetConfig,
    DatasetMeta,
    LoadedDataset,
    TargetColumnSpec,
    TimeSeriesDatasetConfig,
    TimeSeriesSpec,
)
from scripts.replay_data._test_hooks import FakeProducer

from scripts.replay_data import _test_hooks

# =============================================================================
# Fake Dataset Loaders
# =============================================================================


class FakeDatasetLoader:
    """Fake dataset loader for testing."""

    def __init__(self, dataset: LoadedDataset) -> None:
        """Initialize with fixed dataset to return."""
        self._dataset = dataset
        self.load_calls: list[tuple[str, Path]] = []
        self.load_timeseries_calls: list[tuple[str, Path]] = []

    def load(
        self,
        config: DatasetConfig,
        external_dir: Path,
    ) -> LoadedDataset:
        """Return configured dataset."""
        self.load_calls.append((config["name"], external_dir))
        return self._dataset

    def load_timeseries(
        self,
        config: TimeSeriesDatasetConfig,
        external_dir: Path,
    ) -> LoadedDataset:
        """Return configured dataset."""
        self.load_timeseries_calls.append((config["name"], external_dir))
        return self._dataset


class FakeDatasetRegistry:
    """Fake dataset registry for testing."""

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize with fixed config."""
        self._config = config

    def get(self, name: str) -> DatasetConfig:
        """Return configured config."""
        if name != self._config["name"]:
            raise KeyError(f"Dataset '{name}' not found")
        return self._config

    def list_names(self) -> tuple[str, ...]:
        """Return config name."""
        return (self._config["name"],)

    def __contains__(self, name: str) -> bool:
        """Check if name matches config."""
        return name == self._config["name"]


class FakeTimeSeriesRegistry:
    """Fake time-series registry for testing."""

    def __init__(self, config: TimeSeriesDatasetConfig | None = None) -> None:
        """Initialize with optional config."""
        self._config = config

    def get(self, name: str) -> TimeSeriesDatasetConfig:
        """Return configured config."""
        if self._config is None:
            raise KeyError(f"Time-series dataset '{name}' not found")
        if name != self._config["name"]:
            raise KeyError(f"Time-series dataset '{name}' not found")
        return self._config

    def list_names(self) -> tuple[str, ...]:
        """Return config name or empty."""
        if self._config is None:
            return ()
        return (self._config["name"],)

    def __contains__(self, name: str) -> bool:
        """Check if name matches config."""
        if self._config is None:
            return False
        return name == self._config["name"]


# =============================================================================
# Factory Functions for Test Data
# =============================================================================


def make_test_dataset_config(name: str = "test_dataset") -> DatasetConfig:
    """Create a test dataset config.

    Args:
        name: Dataset name.

    Returns:
        DatasetConfig for testing.
    """
    return DatasetConfig(
        name=name,
        display_name=f"Test Dataset ({name})",
        folder=f"{name}_data",
        file_name="data.csv",
        file_format="csv",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="target",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=(),
        n_samples_expected=100,
        n_features_expected=5,
        positive_class_ratio_expected=0.1,
    )


def make_test_timeseries_config(name: str = "test_ts") -> TimeSeriesDatasetConfig:
    """Create a test time-series dataset config.

    Args:
        name: Dataset name.

    Returns:
        TimeSeriesDatasetConfig for testing.
    """
    return TimeSeriesDatasetConfig(
        name=name,
        display_name=f"Test Time-Series ({name})",
        folder=f"{name}_data",
        file_name="data.csv",
        file_format="csv",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="target",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=(),
        n_samples_expected=100,
        n_features_expected=5,
        positive_class_ratio_expected=0.1,
        time_series=TimeSeriesSpec(
            entity_column="entity_id",
            time_column="date",
            aggregation="last",
            labels_file="",
            labels_entity_column="",
            include_rank_features=False,
            include_diff_features=False,
            include_window_features=False,
            window_sizes=(),
        ),
    )


def make_test_loaded_dataset(
    n_samples: int = 10,
    n_features: int = 3,
    feature_prefix: str = "feat",
) -> LoadedDataset:
    """Create a test loaded dataset.

    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        feature_prefix: Prefix for feature names.

    Returns:
        LoadedDataset for testing.
    """
    feature_names = tuple(f"{feature_prefix}_{i}" for i in range(n_features))
    x = np.random.rand(n_samples, n_features).astype(np.float64)
    y = np.random.randint(0, 2, size=n_samples).astype(np.int64)
    n_positive = int(np.sum(y))
    n_negative = n_samples - n_positive

    meta: DatasetMeta = {
        "name": "test",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_ratio": n_positive / n_samples if n_samples > 0 else 0.0,
        "feature_names": feature_names,
        "categorical_encodings": (),
    }

    return LoadedDataset(
        meta=meta,
        x=x,
        y=y,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def fake_producer() -> FakeProducer:
    """Create a fake producer for testing."""
    return FakeProducer()


@pytest.fixture()
def test_dataset() -> LoadedDataset:
    """Create a test dataset with predictable data."""
    return make_test_loaded_dataset(n_samples=5, n_features=3)


@pytest.fixture()
def test_config() -> DatasetConfig:
    """Create a test dataset config."""
    return make_test_dataset_config()


@pytest.fixture()
def external_dir(tmp_path: Path) -> Path:
    """Create a temporary external data directory."""
    ext = tmp_path / "external"
    ext.mkdir()
    return ext


@pytest.fixture()
def restore_hooks() -> Generator[None, None, None]:
    """Fixture that restores _test_hooks after test."""
    # Save original hooks
    orig_perf_counter = _test_hooks.perf_counter
    orig_sleep = _test_hooks.sleep
    orig_generate_uuid = _test_hooks.generate_uuid
    orig_loader_factory = _test_hooks.dataset_loader_factory
    orig_registry_factory = _test_hooks.registry_factory
    orig_ts_registry_factory = _test_hooks.timeseries_registry_factory

    yield

    # Restore original hooks
    _test_hooks.perf_counter = orig_perf_counter
    _test_hooks.sleep = orig_sleep
    _test_hooks.generate_uuid = orig_generate_uuid
    _test_hooks.dataset_loader_factory = orig_loader_factory
    _test_hooks.registry_factory = orig_registry_factory
    _test_hooks.timeseries_registry_factory = orig_ts_registry_factory
