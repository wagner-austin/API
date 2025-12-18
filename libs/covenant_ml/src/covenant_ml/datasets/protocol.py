"""Protocols for the pluggable dataset loading system.

Defines interfaces for dataset loaders and validators.
All protocols use strict signatures without Any types.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from covenant_ml.datasets.types import (
    DatasetConfig,
    DatasetValidationResult,
    LoadedDataset,
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


__all__ = [
    "DatasetLoaderProtocol",
    "DatasetValidatorProtocol",
]
