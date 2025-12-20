"""Protocols for the pluggable dataset loading system.

Defines interfaces for dataset loaders, validators, and progress callbacks.
All protocols use strict signatures without Any types.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from covenant_ml.datasets.types import (
    DatasetConfig,
    DatasetValidationResult,
    LoadedDataset,
    LoadProgress,
    TimeSeriesDatasetConfig,
)


class DatasetLoaderProtocol(Protocol):
    """Protocol for dataset loaders.

    Implementations load datasets from disk into LoadedDataset format.
    """

    def load(
        self,
        config: DatasetConfig,
        external_dir: Path,
    ) -> LoadedDataset:
        """Load a dataset from disk.

        Args:
            config: Dataset configuration specifying file, format, target, etc.
            external_dir: Root directory containing dataset folders.

        Returns:
            LoadedDataset with features, labels, and metadata.

        Raises:
            FileNotFoundError: If dataset file doesn't exist.
            ValueError: If data doesn't match expected format.
        """
        ...


class DatasetValidatorProtocol(Protocol):
    """Protocol for dataset validators.

    Implementations validate datasets without fully loading them.
    """

    def validate(
        self,
        config: DatasetConfig,
        external_dir: Path,
    ) -> DatasetValidationResult:
        """Validate a dataset against its configuration.

        Args:
            config: Dataset configuration to validate against.
            external_dir: Root directory containing dataset folders.

        Returns:
            DatasetValidationResult with is_valid flag and error messages.
        """
        ...


class ProgressCallbackProtocol(Protocol):
    """Protocol for progress callbacks during dataset loading.

    Implementations receive progress updates during loading.
    Called periodically with current loading state.
    """

    def __call__(self, progress: LoadProgress) -> None:
        """Handle progress update.

        Args:
            progress: Current loading progress state.
        """
        ...


class DatasetLoaderWithProgressProtocol(Protocol):
    """Protocol for dataset loaders with progress reporting.

    Extends DatasetLoaderProtocol with optional progress callback support.
    """

    def load(
        self,
        config: DatasetConfig,
        external_dir: Path,
        progress_callback: ProgressCallbackProtocol | None = None,
    ) -> LoadedDataset:
        """Load a dataset from disk with optional progress reporting.

        Args:
            config: Dataset configuration specifying file, format, target, etc.
            external_dir: Root directory containing dataset folders.
            progress_callback: Optional callback for progress updates.

        Returns:
            LoadedDataset with features, labels, and metadata.

        Raises:
            FileNotFoundError: If dataset file doesn't exist.
            ValueError: If data doesn't match expected format.
        """
        ...


class TimeSeriesLoaderWithProgressProtocol(Protocol):
    """Protocol for time-series dataset loaders with progress reporting.

    Handles loading time-series datasets with aggregation and progress reporting.
    """

    def load(
        self,
        config: TimeSeriesDatasetConfig,
        external_dir: Path,
        progress_callback: ProgressCallbackProtocol | None = None,
    ) -> LoadedDataset:
        """Load a time-series dataset with optional progress reporting.

        Args:
            config: Time-series dataset configuration with aggregation spec.
            external_dir: Root directory containing dataset folders.
            progress_callback: Optional callback for progress updates.

        Returns:
            LoadedDataset with aggregated features, labels, and metadata.

        Raises:
            FileNotFoundError: If dataset file doesn't exist.
            ValueError: If data doesn't match expected format.
        """
        ...


__all__ = [
    "DatasetLoaderProtocol",
    "DatasetLoaderWithProgressProtocol",
    "DatasetValidatorProtocol",
    "ProgressCallbackProtocol",
    "TimeSeriesLoaderWithProgressProtocol",
]
