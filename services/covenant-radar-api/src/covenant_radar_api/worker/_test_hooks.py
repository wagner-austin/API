"""Test hooks for worker components, ML registry injection, and dataset loading.

Production code uses real implementations; tests can override these module-level
symbols to inject fakes without conditionals in core logic.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from covenant_ml.backends.protocol import ProgressCallback
from covenant_ml.backends.registry import (
    BackendFactory,
    BackendRegistration,
    ClassifierRegistry,
    default_registry,
)
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
from covenant_ml.features import FeaturePreset
from covenant_ml.optimizer import OptimizerStrategyRegistry, default_optimizer_registry
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)
from covenant_ml.types import BackendName, PredictorProtocol
from numpy.typing import NDArray

from covenant_radar_api.worker.optimize_types import UnifiedOptimizeParseResult

# =============================================================================
# Worker Runner Hook
# =============================================================================


# =============================================================================
# ML Registry Hook
# =============================================================================


class RegistryFactory(Protocol):
    """Protocol for classifier registry factory."""

    def __call__(self) -> ClassifierRegistry: ...


def _full_registry() -> ClassifierRegistry:
    """Build registry with all backends including PyTorch (covenant_nn).

    Returns:
        ClassifierRegistry with tree-based backends from covenant_ml
        and neural backends (MLP, LSTM) from covenant_nn.
    """
    reg = default_registry()
    nn_mod = __import__("covenant_nn", fromlist=["create_mlp_backend", "create_lstm_backend"])
    create_mlp: BackendFactory = nn_mod.create_mlp_backend
    create_lstm: BackendFactory = nn_mod.create_lstm_backend
    reg.register("mlp", BackendRegistration(create_mlp))
    reg.register("lstm", BackendRegistration(create_lstm))
    return reg


registry_factory: RegistryFactory = _full_registry


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
# Optimizer Strategy Registry Hook
# =============================================================================


class OptimizerRegistryFactoryProtocol(Protocol):
    """Protocol for optimizer strategy registry factory function."""

    def __call__(self) -> OptimizerStrategyRegistry:
        """Create an OptimizerStrategyRegistry with optimization strategies.

        Returns:
            OptimizerStrategyRegistry instance.
        """
        ...


def _real_optimizer_registry() -> OptimizerStrategyRegistry:
    """Real implementation returning production optimizer strategy registry.

    Returns:
        OptimizerStrategyRegistry with all built-in strategies.
    """
    return default_optimizer_registry()


optimizer_registry_factory: OptimizerRegistryFactoryProtocol = _real_optimizer_registry


# =============================================================================
# Objective Factory Hook
# =============================================================================


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


def _real_objective_factory(
    backend_name: BackendName,
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    feature_names: list[str],
    config: UnifiedOptimizeParseResult,
) -> ObjectiveWithFeatureCount:
    """Create per-backend objective using dynamic imports.

    Dispatches to the appropriate create_*_objective factory based on
    backend_name. Tree-based backends are in covenant_ml.optimizer,
    neural backends are in covenant_nn.

    Args:
        backend_name: Backend to create objective for.
        x: Feature matrix.
        y: Binary labels.
        feature_names: Feature column names.
        config: Parsed optimization config.

    Returns:
        Objective callable with n_features property.

    Raises:
        ValueError: If backend_name is not recognized.
    """
    if backend_name == "xgboost":
        from covenant_ml.optimizer import create_xgboost_objective

        return create_xgboost_objective(
            x,
            y,
            feature_names,
            config["device"],
            config["feature_preset"],
        )

    if backend_name == "lightgbm":
        from covenant_ml.optimizer import create_lightgbm_objective

        return create_lightgbm_objective(
            x,
            y,
            feature_names,
            config["device"],
            config["feature_preset"],
            early_stopping_rounds=config["early_stopping_rounds"],
            n_jobs=config["n_jobs"],
        )

    if backend_name == "cleargbm":
        from covenant_ml.optimizer import create_cleargbm_objective

        return create_cleargbm_objective(
            x,
            y,
            feature_names,
            config["feature_preset"],
            early_stopping_rounds=config["early_stopping_rounds"],
        )

    if backend_name == "logreg":
        from covenant_ml.optimizer import create_logreg_objective

        return create_logreg_objective(
            x,
            y,
            feature_names,
            config["feature_preset"],
        )

    if backend_name == "random_forest":
        from covenant_ml.optimizer import create_random_forest_objective

        return create_random_forest_objective(
            x,
            y,
            feature_names,
            config["feature_preset"],
        )

    if backend_name == "mlp":
        nn_mod = __import__("covenant_nn", fromlist=["create_mlp_objective"])
        create_mlp: _CreateMLPObjectiveProto = nn_mod.create_mlp_objective
        return create_mlp(
            x,
            y,
            feature_names,
            config["device"],
            config["precision"],
            config["feature_preset"],
            config["n_epochs"],
            config["early_stopping_patience"],
            optimizer_name=config["nn_optimizer"],
        )

    # backend_name == "lstm"
    nn_mod = __import__("covenant_nn", fromlist=["create_lstm_objective"])
    create_lstm_obj: _CreateLSTMObjectiveProto = nn_mod.create_lstm_objective
    return create_lstm_obj(
        x,
        y,
        feature_names,
        config["device"],
        config["precision"],
        config["feature_preset"],
        config["n_epochs"],
        config["early_stopping_patience"],
        config["sequence_length"],
        bidirectional=config["bidirectional"],
    )


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
        epoch_callback: ProgressCallback | None = ...,
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
        epoch_callback: ProgressCallback | None = ...,
    ) -> ObjectiveWithFeatureCount: ...


objective_factory: ObjectiveFactoryProtocol = _real_objective_factory


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


# =============================================================================
# Data Bank Uploader Hook
# =============================================================================


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


def _real_data_bank_uploader(
    model_path: Path,
    data_bank_url: str,
    data_bank_key: str,
) -> str:
    """Real implementation uploading model to data-bank.

    Args:
        model_path: Path to the model file to upload.
        data_bank_url: Base URL for data-bank-api.
        data_bank_key: API key for authentication.

    Returns:
        file_id from data-bank-api.

    Raises:
        DataBankClientError: On upload failure.
    """
    from platform_core.data_bank_client import DataBankClient
    from platform_core.logging import get_logger

    log = get_logger(__name__)
    client = DataBankClient(data_bank_url, data_bank_key)
    file_id = model_path.name
    content_type = "application/octet-stream"
    with model_path.open("rb") as f:
        response = client.upload(file_id, f, content_type=content_type)
    log.info("Uploaded model to data-bank", extra={"file_id": response["file_id"]})
    return response["file_id"]


data_bank_uploader: DataBankUploaderProtocol = _real_data_bank_uploader


__all__ = [
    "DataBankUploaderProtocol",
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
    "ObjectiveFactoryProtocol",
    "ObjectiveWithFeatureCount",
    "OptimizerRegistryFactoryProtocol",
    "PredictorProtocol",
    "ProgressCallbackProtocol",
    "RandomForestLoaderProtocol",
    "RegistryFactory",
    "TimeSeriesDatasetConfig",
    "TimeSeriesDatasetRegistry",
    "TimeSeriesLoaderCallable",
    "TimeSeriesRegistryFactoryProtocol",
    "data_bank_uploader",
    "dataset_loader",
    "dataset_registry_factory",
    "explainer_registry_factory",
    "lightgbm_loader",
    "logreg_loader",
    "lstm_loader",
    "mlp_loader",
    "objective_factory",
    "optimizer_registry_factory",
    "random_forest_loader",
    "registry_factory",
    "timeseries_loader",
    "timeseries_registry_factory",
]
