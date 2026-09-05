"""Hook protocols for covenant_radar_api.worker."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from covenant_ml.backends.registry import (
    ClassifierRegistry,
)
from covenant_ml.datasets import (
    DatasetConfig,
    DatasetRegistry,
    LoadedDataset,
    TimeSeriesDatasetConfig,
    TimeSeriesDatasetRegistry,
)
from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from covenant_ml.explainers.registry import ExplainerRegistry
from covenant_ml.features import FeaturePreset
from covenant_ml.optimizer import OptimizerStrategyRegistry
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)
from covenant_ml.types import BackendName, PredictorProtocol, TrainProgress
from numpy.typing import NDArray

from covenant_radar_api.worker.optimize_types import UnifiedOptimizeParseResult


class RegistryFactory(Protocol):
    """Protocol for classifier registry factory."""

    def __call__(self) -> ClassifierRegistry: ...


class DatasetRegistryFactoryProtocol(Protocol):
    """Protocol for dataset registry factory function."""

    def __call__(self) -> DatasetRegistry:
        """Create a DatasetRegistry with dataset configurations.

        Returns:
            DatasetRegistry instance.
        """
        ...


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


class TimeSeriesRegistryFactoryProtocol(Protocol):
    """Protocol for time-series dataset registry factory function."""

    def __call__(self) -> TimeSeriesDatasetRegistry:
        """Create a TimeSeriesDatasetRegistry with dataset configurations.

        Returns:
            TimeSeriesDatasetRegistry instance.
        """
        ...


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


class ExplainerRegistryFactoryProtocol(Protocol):
    """Protocol for explainer registry factory function."""

    def __call__(self) -> ExplainerRegistry:
        """Create an ExplainerRegistry with explainer implementations.

        Returns:
            ExplainerRegistry instance.
        """
        ...


class OptimizerRegistryFactoryProtocol(Protocol):
    """Protocol for optimizer strategy registry factory function."""

    def __call__(self) -> OptimizerStrategyRegistry:
        """Create an OptimizerStrategyRegistry with optimization strategies.

        Returns:
            OptimizerStrategyRegistry instance.
        """
        ...


class ObjectiveFactoryProtocol(Protocol):
    """Protocol for unified objective factory function.

    Creates per-backend objective functions using dynamic imports.
    The returned objective must have an n_features property.
    """

    def __call__(
        self,
        backend_name: BackendName,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
        feature_names: list[str],
        config: UnifiedOptimizeParseResult,
    ) -> ObjectiveWithFeatureCount:
        """Create an objective function for the specified backend.

        Args:
            backend_name: Backend to create objective for.
            x: Feature matrix.
            y: Binary labels.
            feature_names: Feature column names.
            config: Parsed optimization config with backend-specific fields.

        Returns:
            Objective callable with n_features property.
        """
        ...


class ObjectiveWithFeatureCount(Protocol):
    """Protocol for objective functions that track engineered feature count."""

    @property
    def n_features(self) -> int:
        """Return the actual feature count (after engineering)."""
        ...

    def __call__(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        int_params: SampledIntParams,
        float_params: SampledFloatParams,
        string_params: SampledStringParams,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_state: int,
    ) -> float:
        """Train model with given hyperparameters and return validation AUC.

        Args:
            x_features: Feature matrix.
            y_labels: Binary labels.
            feature_names: Feature column names.
            int_params: Integer hyperparameters.
            float_params: Float hyperparameters.
            string_params: String hyperparameters.
            train_ratio: Training data fraction.
            val_ratio: Validation data fraction.
            test_ratio: Test data fraction.
            random_state: Random seed.

        Returns:
            Validation AUC score.
        """
        ...


class _CreateMLPObjectiveProto(Protocol):
    """Protocol for covenant_nn.create_mlp_objective."""

    def __call__(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        device: Literal["cpu", "cuda", "auto"],
        precision: Literal["fp32", "fp16", "bf16", "auto"],
        feature_preset: FeaturePreset,
        n_epochs: int,
        early_stopping_patience: int,
        optimizer_name: Literal["adamw", "adam", "sgd"] = ...,
        epoch_callback: Callable[[TrainProgress], None] | None = ...,
    ) -> ObjectiveWithFeatureCount: ...


class _CreateLSTMObjectiveProto(Protocol):
    """Protocol for covenant_nn.create_lstm_objective."""

    def __call__(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        device: Literal["cpu", "cuda", "auto"],
        precision: Literal["fp32", "fp16", "bf16", "auto"],
        feature_preset: FeaturePreset,
        n_epochs: int,
        early_stopping_patience: int,
        sequence_length: int,
        bidirectional: bool = ...,
        epoch_callback: Callable[[TrainProgress], None] | None = ...,
    ) -> ObjectiveWithFeatureCount: ...


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


class DataBankUploaderProtocol(Protocol):
    """Protocol for data-bank model uploader function.

    Uploads trained model files to data-bank-api for centralized storage.
    """

    def __call__(
        self,
        model_path: Path,
        data_bank_url: str,
        data_bank_key: str,
    ) -> str:
        """Upload model to data-bank-api.

        Args:
            model_path: Path to the model file to upload.
            data_bank_url: Base URL for data-bank-api.
            data_bank_key: API key for authentication.

        Returns:
            file_id from data-bank-api.

        Raises:
            DataBankClientError: On upload failure.
        """
        ...
