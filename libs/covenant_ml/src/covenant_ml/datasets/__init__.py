"""Pluggable dataset loading system.

Provides a registry-based system for loading external datasets
with auto-detection of target columns and proper validation.

Example:
    >>> from covenant_ml.datasets import make_default_registry, create_dataset_loader
    >>> registry = make_default_registry()
    >>> config = registry.get("taiwan")
    >>> loader = create_dataset_loader()
    >>> dataset = loader.load(config, Path("data/external"))
"""

from covenant_ml.datasets.loader import (
    DatasetLoader,
    create_dataset_loader,
)
from covenant_ml.datasets.loaders._regression_csv import (
    RegressionCSVLoader,
    create_regression_csv_loader,
)
from covenant_ml.datasets.loaders.timeseries_csv_loader import (
    TimeSeriesCSVLoader,
    create_timeseries_csv_loader,
)
from covenant_ml.datasets.protocol import (
    DatasetLoaderProtocol,
    DatasetValidatorProtocol,
    RegressionDatasetLoaderProtocol,
    RegressionDatasetLoaderWithProgressProtocol,
)
from covenant_ml.datasets.registry import (
    DatasetRegistry,
    RegressionDatasetRegistry,
    TimeSeriesDatasetRegistry,
    make_default_registry,
    make_default_regression_registry,
    make_default_timeseries_registry,
)
from covenant_ml.datasets.testing import (
    FakeDatasetLoader,
    FakeRegressionDatasetLoader,
    create_fake_dataset_loader,
    create_fake_regression_dataset_loader,
)
from covenant_ml.datasets.types import (
    AggregationStrategy,
    DatasetConfig,
    DatasetMeta,
    DatasetValidationResult,
    FileEncoding,
    FileFormat,
    LabelType,
    LoadedDataset,
    RegressionDatasetConfig,
    RegressionDatasetMeta,
    RegressionLoadedDataset,
    RegressionTargetSpec,
    TargetColumnSpec,
    TimeSeriesDatasetConfig,
    TimeSeriesSpec,
)

__all__ = [
    "AggregationStrategy",
    "DatasetConfig",
    "DatasetLoader",
    "DatasetLoaderProtocol",
    "DatasetMeta",
    "DatasetRegistry",
    "DatasetValidationResult",
    "DatasetValidatorProtocol",
    "FakeDatasetLoader",
    "FakeRegressionDatasetLoader",
    "FileEncoding",
    "FileFormat",
    "LabelType",
    "LoadedDataset",
    "RegressionCSVLoader",
    "RegressionDatasetConfig",
    "RegressionDatasetLoaderProtocol",
    "RegressionDatasetLoaderWithProgressProtocol",
    "RegressionDatasetMeta",
    "RegressionDatasetRegistry",
    "RegressionLoadedDataset",
    "RegressionTargetSpec",
    "TargetColumnSpec",
    "TimeSeriesCSVLoader",
    "TimeSeriesDatasetConfig",
    "TimeSeriesDatasetRegistry",
    "TimeSeriesSpec",
    "create_dataset_loader",
    "create_fake_dataset_loader",
    "create_fake_regression_dataset_loader",
    "create_regression_csv_loader",
    "create_timeseries_csv_loader",
    "make_default_registry",
    "make_default_regression_registry",
    "make_default_timeseries_registry",
]
