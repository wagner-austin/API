"""Dataset registry for the pluggable dataset loading system.

Provides registries of known dataset configurations.
Immutable after construction, thread-safe for reads.

Registries:
    DatasetRegistry: For standard DatasetConfig (CSV, ARFF)
    TimeSeriesDatasetRegistry: For TimeSeriesDatasetConfig (time-series CSV)
"""

from __future__ import annotations

from covenant_ml.datasets.registry_configs import (
    VERIFIED_CONFIGS,
    VERIFIED_REGRESSION_CONFIGS,
    VERIFIED_TIMESERIES_CONFIGS,
)
from covenant_ml.datasets.types import (
    DatasetConfig,
    RegressionDatasetConfig,
    TimeSeriesDatasetConfig,
)


class DatasetRegistry:
    """Registry of known dataset configurations.

    Immutable after construction. Thread-safe for reads.
    Provides lookup by dataset name with strict validation.
    """

    def __init__(self, configs: tuple[DatasetConfig, ...]) -> None:
        """Initialize with a tuple of dataset configs.

        Args:
            configs: Immutable tuple of DatasetConfig entries.

        Raises:
            ValueError: If duplicate dataset names found.
        """
        self._configs: dict[str, DatasetConfig] = {}
        for cfg in configs:
            name = cfg["name"]
            if name in self._configs:
                raise ValueError(f"Duplicate dataset name: {name}")
            self._configs[name] = cfg

    def get(self, name: str) -> DatasetConfig:
        """Get configuration for a dataset by name.

        Args:
            name: Dataset name (e.g., "kaggle_company_bankruptcy").

        Returns:
            DatasetConfig for the requested dataset.

        Raises:
            KeyError: If dataset not found in registry.
        """
        if name not in self._configs:
            available = ", ".join(sorted(self._configs.keys()))
            raise KeyError(f"Dataset '{name}' not found. Available: {available}")
        return self._configs[name]

    def list_names(self) -> tuple[str, ...]:
        """List all registered dataset names.

        Returns:
            Sorted tuple of dataset names.
        """
        return tuple(sorted(self._configs.keys()))

    def __contains__(self, name: str) -> bool:
        """Check if dataset is registered.

        Args:
            name: Dataset name to check.

        Returns:
            True if dataset is in registry.
        """
        return name in self._configs

    def __len__(self) -> int:
        """Get number of registered datasets.

        Returns:
            Number of datasets in registry.
        """
        return len(self._configs)


class TimeSeriesDatasetRegistry:
    """Registry of time-series dataset configurations.

    Stores TimeSeriesDatasetConfig entries for datasets with
    multiple observations per entity over time.

    Immutable after construction. Thread-safe for reads.
    """

    def __init__(self, configs: tuple[TimeSeriesDatasetConfig, ...]) -> None:
        """Initialize with a tuple of time-series dataset configs.

        Args:
            configs: Immutable tuple of TimeSeriesDatasetConfig entries.

        Raises:
            ValueError: If duplicate dataset names found.
        """
        self._configs: dict[str, TimeSeriesDatasetConfig] = {}
        for cfg in configs:
            name = cfg["name"]
            if name in self._configs:
                raise ValueError(f"Duplicate dataset name: {name}")
            self._configs[name] = cfg

    def get(self, name: str) -> TimeSeriesDatasetConfig:
        """Get time-series configuration for a dataset by name.

        Args:
            name: Dataset name (e.g., "kaggle_amex_default").

        Returns:
            TimeSeriesDatasetConfig for the requested dataset.

        Raises:
            KeyError: If dataset not found in registry.
        """
        if name not in self._configs:
            available = ", ".join(sorted(self._configs.keys()))
            raise KeyError(f"Time-series dataset '{name}' not found. Available: {available}")
        return self._configs[name]

    def list_names(self) -> tuple[str, ...]:
        """List all registered time-series dataset names.

        Returns:
            Sorted tuple of dataset names.
        """
        return tuple(sorted(self._configs.keys()))

    def __contains__(self, name: str) -> bool:
        """Check if time-series dataset is registered.

        Args:
            name: Dataset name to check.

        Returns:
            True if dataset is in registry.
        """
        return name in self._configs

    def __len__(self) -> int:
        """Get number of registered time-series datasets.

        Returns:
            Number of datasets in registry.
        """
        return len(self._configs)


def make_default_timeseries_registry() -> TimeSeriesDatasetRegistry:
    """Create registry with verified time-series dataset configurations.

    Returns:
        TimeSeriesDatasetRegistry with production configs.
    """
    return TimeSeriesDatasetRegistry(VERIFIED_TIMESERIES_CONFIGS)


def make_default_registry() -> DatasetRegistry:
    """Create registry with all verified dataset configurations.

    Returns:
        DatasetRegistry with production dataset configs.
    """
    return DatasetRegistry(VERIFIED_CONFIGS)


class RegressionDatasetRegistry:
    """Registry of regression dataset configurations.

    Stores RegressionDatasetConfig entries for datasets with
    continuous target values.

    Immutable after construction. Thread-safe for reads.
    """

    def __init__(self, configs: tuple[RegressionDatasetConfig, ...]) -> None:
        """Initialize with a tuple of regression dataset configs.

        Args:
            configs: Immutable tuple of RegressionDatasetConfig entries.

        Raises:
            ValueError: If duplicate dataset names found.
        """
        self._configs: dict[str, RegressionDatasetConfig] = {}
        for cfg in configs:
            name = cfg["name"]
            if name in self._configs:
                raise ValueError(f"Duplicate dataset name: {name}")
            self._configs[name] = cfg

    def get(self, name: str) -> RegressionDatasetConfig:
        """Get configuration for a regression dataset by name.

        Args:
            name: Dataset name (e.g., "financial_distress").

        Returns:
            RegressionDatasetConfig for the requested dataset.

        Raises:
            KeyError: If dataset not found in registry.
        """
        if name not in self._configs:
            available = ", ".join(sorted(self._configs.keys()))
            raise KeyError(f"Regression dataset '{name}' not found. Available: {available}")
        return self._configs[name]

    def list_names(self) -> tuple[str, ...]:
        """List all registered regression dataset names.

        Returns:
            Sorted tuple of dataset names.
        """
        return tuple(sorted(self._configs.keys()))

    def __contains__(self, name: str) -> bool:
        """Check if regression dataset is registered.

        Args:
            name: Dataset name to check.

        Returns:
            True if dataset is in registry.
        """
        return name in self._configs

    def __len__(self) -> int:
        """Get number of registered regression datasets.

        Returns:
            Number of datasets in registry.
        """
        return len(self._configs)


def make_default_regression_registry() -> RegressionDatasetRegistry:
    """Create registry with verified regression dataset configurations.

    Returns:
        RegressionDatasetRegistry with production regression configs.
    """
    return RegressionDatasetRegistry(VERIFIED_REGRESSION_CONFIGS)


__all__ = [
    "DatasetRegistry",
    "RegressionDatasetRegistry",
    "TimeSeriesDatasetRegistry",
    "make_default_registry",
    "make_default_regression_registry",
    "make_default_timeseries_registry",
]
