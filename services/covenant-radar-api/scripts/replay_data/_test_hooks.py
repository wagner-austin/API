"""Test hooks for data replay script.

Production code uses real implementations; tests can override these
module-level symbols to inject fakes without conditionals in core logic.

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Protocol

import numpy as np
from covenant_ml.datasets import (
    DatasetConfig,
    LoadedDataset,
    TimeSeriesDatasetConfig,
    create_dataset_loader,
    make_default_registry,
    make_default_timeseries_registry,
)
from numpy.typing import NDArray

from covenant_radar_api.streaming.schemas import MeasurementEventV1

# =============================================================================
# Time and UUID Hooks
# =============================================================================


class TimeProtocol(Protocol):
    """Protocol for time functions."""

    def __call__(self) -> float:
        """Return current time in seconds since epoch."""
        ...


class SleepProtocol(Protocol):
    """Protocol for sleep function."""

    def __call__(self, seconds: float) -> None:
        """Sleep for specified seconds."""
        ...


class UUIDProtocol(Protocol):
    """Protocol for UUID generation."""

    def __call__(self) -> str:
        """Generate a UUID string."""
        ...


def _real_time() -> float:
    """Real implementation using time.perf_counter."""
    return time.perf_counter()


def _real_sleep(seconds: float) -> None:
    """Real implementation using time.sleep."""
    time.sleep(seconds)


def _real_uuid() -> str:
    """Real implementation using uuid.uuid4."""
    return str(uuid.uuid4())


# Module-level hooks
perf_counter: TimeProtocol = _real_time
sleep: SleepProtocol = _real_sleep
generate_uuid: UUIDProtocol = _real_uuid


# =============================================================================
# Dataset Loading Hooks
# =============================================================================


class DatasetLoaderProtocol(Protocol):
    """Protocol for dataset loader."""

    def load(
        self,
        config: DatasetConfig,
        external_dir: Path,
    ) -> LoadedDataset:
        """Load a standard dataset."""
        ...

    def load_timeseries(
        self,
        config: TimeSeriesDatasetConfig,
        external_dir: Path,
    ) -> LoadedDataset:
        """Load a time-series dataset."""
        ...


class DatasetRegistryProtocol(Protocol):
    """Protocol for dataset registry."""

    def get(self, name: str) -> DatasetConfig:
        """Get config by name."""
        ...

    def list_names(self) -> tuple[str, ...]:
        """List all dataset names."""
        ...

    def __contains__(self, name: str) -> bool:
        """Check if dataset exists."""
        ...


class TimeSeriesRegistryProtocol(Protocol):
    """Protocol for time-series dataset registry."""

    def get(self, name: str) -> TimeSeriesDatasetConfig:
        """Get config by name."""
        ...

    def list_names(self) -> tuple[str, ...]:
        """List all dataset names."""
        ...

    def __contains__(self, name: str) -> bool:
        """Check if dataset exists."""
        ...


class DatasetLoaderFactoryProtocol(Protocol):
    """Protocol for dataset loader factory."""

    def __call__(self) -> DatasetLoaderProtocol:
        """Create a dataset loader."""
        ...


class RegistryFactoryProtocol(Protocol):
    """Protocol for registry factory."""

    def __call__(self) -> DatasetRegistryProtocol:
        """Create a dataset registry."""
        ...


class TimeSeriesRegistryFactoryProtocol(Protocol):
    """Protocol for time-series registry factory."""

    def __call__(self) -> TimeSeriesRegistryProtocol:
        """Create a time-series registry."""
        ...


def _real_dataset_loader_factory() -> DatasetLoaderProtocol:
    """Real implementation creating DatasetLoader."""
    loader: DatasetLoaderProtocol = create_dataset_loader()
    return loader


def _real_registry_factory() -> DatasetRegistryProtocol:
    """Real implementation creating default registry."""
    registry: DatasetRegistryProtocol = make_default_registry()
    return registry


def _real_timeseries_registry_factory() -> TimeSeriesRegistryProtocol:
    """Real implementation creating time-series registry."""
    registry: TimeSeriesRegistryProtocol = make_default_timeseries_registry()
    return registry


# Module-level hooks
dataset_loader_factory: DatasetLoaderFactoryProtocol = _real_dataset_loader_factory
registry_factory: RegistryFactoryProtocol = _real_registry_factory
timeseries_registry_factory: TimeSeriesRegistryFactoryProtocol = _real_timeseries_registry_factory


# =============================================================================
# Feature Array Protocol
# =============================================================================


class Features2DProtocol(Protocol):
    """Protocol for 2D feature array with typed indexing.

    Provides type-safe access to numpy arrays for mypy strict mode.
    The __getitem__ method returns float instead of Any.
    """

    @property
    def shape(self) -> tuple[int, int]:
        """Return (n_rows, n_cols) shape."""
        ...

    def __getitem__(self, idx: tuple[int, int]) -> float:
        """Get value at (row, col) indices."""
        ...


class _Features2DWrapper:
    """Wrapper providing typed access to numpy float64 arrays.

    Implements Features2DProtocol by converting numpy array
    element access to Python floats for mypy compatibility.
    """

    def __init__(self, data: NDArray[np.float64]) -> None:
        """Initialize with numpy array.

        Args:
            data: 2D numpy array of float64 values.
        """
        self._data = data

    @property
    def shape(self) -> tuple[int, int]:
        """Return shape as (n_rows, n_cols).

        Returns:
            Tuple of (n_rows, n_cols).
        """
        return (int(self._data.shape[0]), int(self._data.shape[1]))

    def __getitem__(self, idx: tuple[int, int]) -> float:
        """Get value at (row, col) indices.

        Args:
            idx: Tuple of (row, col) indices.

        Returns:
            Float value at the specified position.
        """
        row, col = idx
        # Use slice indexing pattern to get typed float (avoids mypy Any issues)
        val_slice = np.asarray(self._data[row : row + 1, col : col + 1], dtype=np.float64).flat
        result: float = float(val_slice[0])
        return result


def wrap_features(data: NDArray[np.float64]) -> Features2DProtocol:
    """Wrap numpy array for typed 2D access.

    Args:
        data: 2D numpy array of float64 values.

    Returns:
        Wrapped array implementing Features2DProtocol.
    """
    return _Features2DWrapper(data)


# =============================================================================
# Producer Hook
# =============================================================================


class ProducerProtocol(Protocol):
    """Protocol for Kafka producer operations needed by replay."""

    def produce_event(
        self,
        event: MeasurementEventV1,
        topic: str,
    ) -> None:
        """Produce a measurement event."""
        ...

    def poll(self, timeout_seconds: float) -> int:
        """Poll for delivery reports."""
        ...

    def flush(self, timeout_seconds: float) -> int:
        """Flush pending messages."""
        ...


class FakeProducer:
    """Fake producer for testing.

    Records all produced events for verification.
    """

    def __init__(self) -> None:
        """Initialize with empty event list."""
        self.events: list[tuple[MeasurementEventV1, str]] = []
        self.poll_count: int = 0
        self.flush_count: int = 0

    def produce_event(
        self,
        event: MeasurementEventV1,
        topic: str,
    ) -> None:
        """Record the event."""
        self.events.append((event, topic))

    def poll(self, timeout_seconds: float) -> int:
        """Record poll call."""
        self.poll_count += 1
        return 0

    def flush(self, timeout_seconds: float) -> int:
        """Record flush call."""
        self.flush_count += 1
        return 0


__all__ = [
    "DatasetLoaderFactoryProtocol",
    "DatasetLoaderProtocol",
    "DatasetRegistryProtocol",
    "FakeProducer",
    "Features2DProtocol",
    "ProducerProtocol",
    "RegistryFactoryProtocol",
    "SleepProtocol",
    "TimeProtocol",
    "TimeSeriesRegistryFactoryProtocol",
    "TimeSeriesRegistryProtocol",
    "UUIDProtocol",
    "_real_dataset_loader_factory",
    "_real_registry_factory",
    "_real_sleep",
    "_real_time",
    "_real_timeseries_registry_factory",
    "_real_uuid",
    "dataset_loader_factory",
    "generate_uuid",
    "perf_counter",
    "registry_factory",
    "sleep",
    "timeseries_registry_factory",
    "wrap_features",
]
