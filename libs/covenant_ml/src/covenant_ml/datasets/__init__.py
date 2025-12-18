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
from covenant_ml.datasets.protocol import (
    DatasetLoaderProtocol,
    DatasetValidatorProtocol,
)
from covenant_ml.datasets.registry import (
    DatasetRegistry,
    make_default_registry,
)
from covenant_ml.datasets.testing import (
    FakeDatasetLoader,
    create_fake_dataset_loader,
)
from covenant_ml.datasets.types import (
    DatasetConfig,
    DatasetMeta,
    DatasetValidationResult,
    FileEncoding,
    FileFormat,
    LabelType,
    LoadedDataset,
    TargetColumnSpec,
)

__all__ = [
    # Types
    "DatasetConfig",
    "DatasetMeta",
    "DatasetValidationResult",
    "FileEncoding",
    "FileFormat",
    "LabelType",
    "LoadedDataset",
    "TargetColumnSpec",
    # Protocols
    "DatasetLoaderProtocol",
    "DatasetValidatorProtocol",
    # Registry
    "DatasetRegistry",
    "make_default_registry",
    # Loader
    "DatasetLoader",
    "create_dataset_loader",
    # Testing utilities
    "FakeDatasetLoader",
    "create_fake_dataset_loader",
]
