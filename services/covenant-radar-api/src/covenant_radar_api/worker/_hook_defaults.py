"""Default (production) implementations for covenant_radar_api.worker hooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
from covenant_ml.optimizer import OptimizerStrategyRegistry, default_optimizer_registry
from covenant_ml.types import BackendName, PredictorProtocol
from numpy.typing import NDArray

from covenant_radar_api.worker._hook_protocols import (
    ObjectiveWithFeatureCount,
    _CreateLSTMObjectiveProto,
    _CreateMLPObjectiveProto,
)
from covenant_radar_api.worker.optimize_types import UnifiedOptimizeParseResult


def _real_dataset_registry() -> DatasetRegistry:
    """Real implementation returning production dataset registry.

    Returns:
        DatasetRegistry with all verified dataset configurations.
    """
    return make_default_registry()


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


def _real_timeseries_registry() -> TimeSeriesDatasetRegistry:
    """Real implementation returning production time-series dataset registry.

    Returns:
        TimeSeriesDatasetRegistry with all verified time-series dataset configurations.
    """
    return make_default_timeseries_registry()


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


def _real_explainer_registry() -> ExplainerRegistry:
    """Real implementation returning production explainer registry.

    Returns:
        ExplainerRegistry with all supported explainers.
    """
    return default_explainer_registry()


def _real_optimizer_registry() -> OptimizerStrategyRegistry:
    """Real implementation returning production optimizer strategy registry.

    Returns:
        OptimizerStrategyRegistry with all built-in strategies.
    """
    return default_optimizer_registry()


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
