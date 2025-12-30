"""Test hooks for worker components, ML registry injection, and dataset loading.

Production code uses real implementations; tests can override these module-level
symbols to inject fakes without conditionals in core logic.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from covenant_ml.backends.registry import ClassifierRegistry, default_registry
from covenant_ml.datasets import (
    DatasetConfig,
    DatasetRegistry,
    LoadedDataset,
    TimeSeriesDatasetConfig,
    TimeSeriesDatasetRegistry,
    create_dataset_loader,
    create_timeseries_csv_loader,
    make_default_registry,
    make_default_timeseries_registry,
)
from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from covenant_ml.explainers.registry import ExplainerRegistry, default_explainer_registry
from covenant_ml.types import PredictorProtocol
from platform_workers.rq_harness import WorkerConfig

# =============================================================================
# Worker Runner Hook
# =============================================================================


class WorkerRunnerProtocol(Protocol):
    """Protocol for worker runner function."""

    def __call__(self, config: WorkerConfig) -> None: ...


test_runner: WorkerRunnerProtocol | None = None


# =============================================================================
# ML Registry Hook
# =============================================================================


class RegistryFactory(Protocol):
    """Protocol for classifier registry factory."""

    def __call__(self) -> ClassifierRegistry: ...


registry_factory: RegistryFactory = default_registry


# =============================================================================
# Dataset Registry Hook
# =============================================================================


class DatasetRegistryFactoryProtocol(Protocol):
    """Protocol for dataset registry factory function."""

    def __call__(self) -> DatasetRegistry:
        """Create a DatasetRegistry with dataset configurations.

        Returns:
            DatasetRegistry instance.
        """
        ...


def _real_dataset_registry() -> DatasetRegistry:
    """Real implementation returning production dataset registry.

    Returns:
        DatasetRegistry with all verified dataset configurations.
    """
    return make_default_registry()


dataset_registry_factory: DatasetRegistryFactoryProtocol = _real_dataset_registry


# =============================================================================
# Dataset Loader Hook
# =============================================================================


class DatasetLoaderCallable(Protocol):
    """Protocol for callable dataset loader function.

    Defines the signature of a callable function for loading datasets.
    Supports optional progress callback for granular loading progress.
    """

    def __call__(
        self,
        config: DatasetConfig,
        external_dir: Path,
        progress_callback: ProgressCallbackProtocol | None = None,
    ) -> LoadedDataset:
        """Load a dataset from disk.

        Args:
            config: Dataset configuration from registry.
            external_dir: Root directory containing dataset folders.
            progress_callback: Optional callback for loading progress updates.

        Returns:
            LoadedDataset with features, labels, and metadata.

        Raises:
            FileNotFoundError: If dataset file doesn't exist.
            ValueError: If data doesn't match expected format.
        """
        ...


def _real_dataset_loader(
    config: DatasetConfig,
    external_dir: Path,
    progress_callback: ProgressCallbackProtocol | None = None,
) -> LoadedDataset:
    """Real implementation using covenant_ml.datasets loader.

    Args:
        config: Dataset configuration from registry.
        external_dir: Root directory containing dataset folders.
        progress_callback: Optional callback for loading progress updates.

    Returns:
        LoadedDataset with features, labels, and metadata.

    Raises:
        FileNotFoundError: If dataset file doesn't exist.
        ValueError: If data doesn't match expected format.
    """
    loader = create_dataset_loader()
    return loader.load(config, external_dir, progress_callback)


dataset_loader: DatasetLoaderCallable = _real_dataset_loader


# =============================================================================
# Time-Series Dataset Registry Hook
# =============================================================================


class TimeSeriesRegistryFactoryProtocol(Protocol):
    """Protocol for time-series dataset registry factory function."""

    def __call__(self) -> TimeSeriesDatasetRegistry:
        """Create a TimeSeriesDatasetRegistry with dataset configurations.

        Returns:
            TimeSeriesDatasetRegistry instance.
        """
        ...


def _real_timeseries_registry() -> TimeSeriesDatasetRegistry:
    """Real implementation returning production time-series dataset registry.

    Returns:
        TimeSeriesDatasetRegistry with all verified time-series dataset configurations.
    """
    return make_default_timeseries_registry()


timeseries_registry_factory: TimeSeriesRegistryFactoryProtocol = _real_timeseries_registry


# =============================================================================
# Time-Series Dataset Loader Hook
# =============================================================================


class TimeSeriesLoaderCallable(Protocol):
    """Protocol for callable time-series dataset loader function.

    Defines the signature of a callable function for loading time-series datasets.
    Supports optional progress callback for granular loading progress.
    """

    def __call__(
        self,
        config: TimeSeriesDatasetConfig,
        external_dir: Path,
        progress_callback: ProgressCallbackProtocol | None = None,
    ) -> LoadedDataset:
        """Load a time-series dataset from disk.

        Args:
            config: Time-series dataset configuration from registry.
            external_dir: Root directory containing dataset folders.
            progress_callback: Optional callback for loading progress updates.

        Returns:
            LoadedDataset with aggregated features, labels, and metadata.

        Raises:
            FileNotFoundError: If dataset file doesn't exist.
            ValueError: If data doesn't match expected format.
        """
        ...


def _real_timeseries_loader(
    config: TimeSeriesDatasetConfig,
    external_dir: Path,
    progress_callback: ProgressCallbackProtocol | None = None,
) -> LoadedDataset:
    """Real implementation using covenant_ml.datasets time-series loader.

    Args:
        config: Time-series dataset configuration from registry.
        external_dir: Root directory containing dataset folders.
        progress_callback: Optional callback for loading progress updates.

    Returns:
        LoadedDataset with aggregated features, labels, and metadata.

    Raises:
        FileNotFoundError: If dataset file doesn't exist.
        ValueError: If data doesn't match expected format.
    """
    loader = create_timeseries_csv_loader()
    return loader.load(config, external_dir, progress_callback)


timeseries_loader: TimeSeriesLoaderCallable = _real_timeseries_loader


# =============================================================================
# Explainer Registry Hook
# =============================================================================


class ExplainerRegistryFactoryProtocol(Protocol):
    """Protocol for explainer registry factory function."""

    def __call__(self) -> ExplainerRegistry:
        """Create an ExplainerRegistry with explainer implementations.

        Returns:
            ExplainerRegistry instance.
        """
        ...


def _real_explainer_registry() -> ExplainerRegistry:
    """Real implementation returning production explainer registry.

    Returns:
        ExplainerRegistry with all supported explainers.
    """
    return default_explainer_registry()


explainer_registry_factory: ExplainerRegistryFactoryProtocol = _real_explainer_registry


# =============================================================================
# Model Loader Hooks
# =============================================================================


class MLPLoaderProtocol(Protocol):
    """Protocol for MLP model loader function."""

    def __call__(self, model_path: Path, meta_path: Path) -> PredictorProtocol:
        """Load MLP model from state dict using metadata.

        Args:
            model_path: Path to .pt state dict file.
            meta_path: Path to metadata JSON file.

        Returns:
            Prepared MLP model implementing PredictorProtocol.

        Raises:
            FileNotFoundError: If model or metadata file missing.
            JSONTypeError: If metadata is invalid.
        """
        ...


def _real_mlp_loader(model_path: Path, meta_path: Path) -> PredictorProtocol:
    """Real implementation loading MLP model from disk.

    Args:
        model_path: Path to .pt state dict file.
        meta_path: Path to metadata JSON file.

    Returns:
        Prepared MLP model implementing PredictorProtocol.

    Raises:
        FileNotFoundError: If model or metadata file missing.
        JSONTypeError: If metadata is invalid.
    """
    from covenant_radar_api.worker._model_loaders import load_mlp_model

    return load_mlp_model(model_path, meta_path)


mlp_loader: MLPLoaderProtocol = _real_mlp_loader


class LSTMLoaderProtocol(Protocol):
    """Protocol for LSTM model loader function."""

    def __call__(self, model_path: Path, meta_path: Path) -> PredictorProtocol:
        """Load LSTM model from state dict using metadata.

        Args:
            model_path: Path to .pt state dict file.
            meta_path: Path to metadata JSON file.

        Returns:
            Prepared LSTM model implementing PredictorProtocol.

        Raises:
            FileNotFoundError: If model or metadata file missing.
            JSONTypeError: If metadata is invalid.
        """
        ...


def _real_lstm_loader(model_path: Path, meta_path: Path) -> PredictorProtocol:
    """Real implementation loading LSTM model from disk.

    Args:
        model_path: Path to .pt state dict file.
        meta_path: Path to metadata JSON file.

    Returns:
        Prepared LSTM model implementing PredictorProtocol.

    Raises:
        FileNotFoundError: If model or metadata file missing.
        JSONTypeError: If metadata is invalid.
    """
    from covenant_radar_api.worker._model_loaders import load_lstm_model

    return load_lstm_model(model_path, meta_path)


lstm_loader: LSTMLoaderProtocol = _real_lstm_loader


class LightGBMLoaderProtocol(Protocol):
    """Protocol for LightGBM model loader function."""

    def __call__(self, model_path: Path) -> PredictorProtocol:
        """Load LightGBM model from .txt file.

        Args:
            model_path: Path to the saved model file (.txt format).

        Returns:
            Prepared LightGBM model implementing PredictorProtocol.

        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        ...


def _real_lightgbm_loader(model_path: Path) -> PredictorProtocol:
    """Real implementation loading LightGBM model from disk.

    Args:
        model_path: Path to the saved model file (.txt format).

    Returns:
        Prepared LightGBM model implementing PredictorProtocol.

    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    from covenant_radar_api.worker._model_loaders import load_lightgbm_model

    return load_lightgbm_model(model_path)


lightgbm_loader: LightGBMLoaderProtocol = _real_lightgbm_loader


class LogRegLoaderProtocol(Protocol):
    """Protocol for Logistic Regression model loader function."""

    def __call__(self, model_path: Path) -> PredictorProtocol:
        """Load LogReg model from .joblib file.

        Args:
            model_path: Path to the saved model file (.joblib format).

        Returns:
            Prepared LogReg model implementing PredictorProtocol.

        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        ...


def _real_logreg_loader(model_path: Path) -> PredictorProtocol:
    """Real implementation loading LogReg model from disk.

    Args:
        model_path: Path to the saved model file (.joblib format).

    Returns:
        Prepared LogReg model implementing PredictorProtocol.

    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    from covenant_radar_api.worker._model_loaders import load_logreg_model

    return load_logreg_model(model_path)


logreg_loader: LogRegLoaderProtocol = _real_logreg_loader


class RandomForestLoaderProtocol(Protocol):
    """Protocol for Random Forest model loader function."""

    def __call__(self, model_path: Path) -> PredictorProtocol:
        """Load Random Forest model from .joblib file.

        Args:
            model_path: Path to the saved model file (.joblib format).

        Returns:
            Prepared Random Forest model implementing PredictorProtocol.

        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        ...


def _real_random_forest_loader(model_path: Path) -> PredictorProtocol:
    """Real implementation loading Random Forest model from disk.

    Args:
        model_path: Path to the saved model file (.joblib format).

    Returns:
        Prepared Random Forest model implementing PredictorProtocol.

    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    from covenant_radar_api.worker._model_loaders import load_random_forest_model

    return load_random_forest_model(model_path)


random_forest_loader: RandomForestLoaderProtocol = _real_random_forest_loader


__all__ = [
    "DatasetConfig",
    "DatasetLoaderCallable",
    "DatasetRegistry",
    "DatasetRegistryFactoryProtocol",
    "ExplainerRegistry",
    "ExplainerRegistryFactoryProtocol",
    "LSTMLoaderProtocol",
    "LightGBMLoaderProtocol",
    "LoadedDataset",
    "LogRegLoaderProtocol",
    "MLPLoaderProtocol",
    "PredictorProtocol",
    "ProgressCallbackProtocol",
    "RandomForestLoaderProtocol",
    "RegistryFactory",
    "TimeSeriesDatasetConfig",
    "TimeSeriesDatasetRegistry",
    "TimeSeriesLoaderCallable",
    "TimeSeriesRegistryFactoryProtocol",
    "WorkerRunnerProtocol",
    "dataset_loader",
    "dataset_registry_factory",
    "explainer_registry_factory",
    "lightgbm_loader",
    "logreg_loader",
    "lstm_loader",
    "mlp_loader",
    "random_forest_loader",
    "registry_factory",
    "test_runner",
    "timeseries_loader",
    "timeseries_registry_factory",
]
