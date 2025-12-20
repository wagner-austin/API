"""Unified dataset loader.

Routes loading to format-specific loaders based on configuration.
Supports standard datasets and time-series datasets with aggregation.

Usage:
    loader = create_dataset_loader()

    # Standard datasets (CSV, ARFF)
    loaded = loader.load(config, external_dir)

    # Time-series datasets (requires TimeSeriesDatasetConfig)
    loaded = loader.load_timeseries(ts_config, external_dir)
"""

from __future__ import annotations

from pathlib import Path

from covenant_ml.datasets.loaders.arff_loader import ARFFLoader
from covenant_ml.datasets.loaders.csv_loader import CSVLoader
from covenant_ml.datasets.loaders.timeseries_csv_loader import TimeSeriesCSVLoader
from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from covenant_ml.datasets.types import (
    DatasetConfig,
    LoadedDataset,
    TimeSeriesDatasetConfig,
)


class DatasetLoader:
    """Unified dataset loader supporting multiple formats.

    Routes loading to format-specific loaders based on config type.
    Thread-safe for concurrent reads.

    Methods:
        load(): For standard DatasetConfig (CSV, ARFF files)
        load_timeseries(): For TimeSeriesDatasetConfig (time-series CSV)

    The separation ensures proper typing without runtime type checking.
    Users should use the method matching their config type.
    """

    def __init__(self) -> None:
        """Initialize with format-specific loaders."""
        self._csv_loader = CSVLoader()
        self._arff_loader = ARFFLoader()
        self._timeseries_csv_loader = TimeSeriesCSVLoader()

    def load(
        self,
        config: DatasetConfig,
        external_dir: Path,
        progress_callback: ProgressCallbackProtocol | None = None,
    ) -> LoadedDataset:
        """Load standard dataset based on format specified in config.

        For time-series datasets, use load_timeseries() instead.

        Args:
            config: Dataset configuration specifying format and location.
            external_dir: Root directory containing dataset folders.
            progress_callback: Optional callback for loading progress updates.

        Returns:
            LoadedDataset with features, labels, and metadata.

        Raises:
            FileNotFoundError: If dataset file doesn't exist.
            ValueError: If format unsupported or data invalid.
        """
        file_format = config["file_format"]

        if file_format == "csv":
            return self._csv_loader.load(config, external_dir, progress_callback)
        if file_format == "arff":
            return self._arff_loader.load(config, external_dir)
        # Excel format not yet implemented
        raise ValueError(f"Excel format not yet implemented for dataset '{config['name']}'")

    def load_timeseries(
        self,
        config: TimeSeriesDatasetConfig,
        external_dir: Path,
        progress_callback: ProgressCallbackProtocol | None = None,
    ) -> LoadedDataset:
        """Load time-series dataset with temporal aggregation.

        Handles datasets with multiple observations per entity over time.
        Aggregates features according to the strategy in config.time_series.

        Args:
            config: Time-series dataset configuration with aggregation spec.
            external_dir: Root directory containing dataset folders.
            progress_callback: Optional callback for loading progress updates.

        Returns:
            LoadedDataset with aggregated features ready for ML.

        Raises:
            FileNotFoundError: If dataset file doesn't exist.
            ValueError: If data invalid or parsing fails.
        """
        return self._timeseries_csv_loader.load(config, external_dir, progress_callback)


def create_dataset_loader() -> DatasetLoader:
    """Factory function for creating unified dataset loader.

    Returns:
        New DatasetLoader instance.
    """
    return DatasetLoader()


__all__ = [
    "DatasetLoader",
    "create_dataset_loader",
]
