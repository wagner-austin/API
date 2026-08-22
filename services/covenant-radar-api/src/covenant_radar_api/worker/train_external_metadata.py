"""Model metadata writing for external training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from covenant_ml.types import (
    BackendName,
    LightGBMModelMeta,
    LogRegConfig,
    LogRegModelMeta,
    LSTMConfig,
    LSTMModelMeta,
    MLPConfig,
    MLPModelMeta,
    ModelMeta,
    RandomForestConfig,
    RandomForestModelMeta,
)
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger

_log = get_logger(__name__)


def _get_active_filename(backend_name: str) -> str:
    """Get active model filename for backend.

    Args:
        backend_name: Name of the ML backend.

    Returns:
        Active model filename with appropriate extension.

    Raises:
        ValueError: If backend_name is not recognized.
    """
    if backend_name == "xgboost":
        return "active_xgb.ubj"
    if backend_name == "mlp":
        return "active_mlp.pt"
    if backend_name == "lstm":
        return "active_lstm.pt"
    if backend_name == "lightgbm":
        return "active_lgbm.txt"
    if backend_name == "cleargbm":
        return "active_cgbm.json"
    if backend_name == "logreg":
        return "active_logreg.joblib"
    if backend_name == "random_forest":
        return "active_rf.joblib"
    raise ValueError(f"Unknown backend: {backend_name}")


def _get_meta_filename(backend_name: BackendName) -> str:
    """Get metadata filename for backend.

    Args:
        backend_name: Name of the ML backend.

    Returns:
        Filename for the metadata JSON file, or empty string if
        the backend's format is self-describing.
    """
    if backend_name == "mlp":
        return "active_mlp_meta.json"
    if backend_name == "lstm":
        return "active_lstm_meta.json"
    if backend_name == "lightgbm":
        return "active_lgbm_meta.json"
    if backend_name == "logreg":
        return "active_logreg_meta.json"
    if backend_name == "random_forest":
        return "active_rf_meta.json"
    # xgboost and cleargbm don't need metadata (self-describing formats)
    return ""


# =============================================================================
# Metadata builders
# =============================================================================


def _build_mlp_metadata(
    config: MLPConfig,
    n_features: int,
) -> MLPModelMeta:
    """Build MLP model metadata.

    Args:
        config: MLP training configuration.
        n_features: Number of input features.

    Returns:
        MLPModelMeta TypedDict.
    """
    return {
        "backend": "mlp",
        "n_features": n_features,
        "hidden_sizes": list(config["hidden_sizes"]),
        "dropout": config["dropout"],
    }


def _build_lstm_metadata(
    config: LSTMConfig,
    n_features: int,
) -> LSTMModelMeta:
    """Build LSTM model metadata.

    Args:
        config: LSTM training configuration.
        n_features: Number of input features.

    Returns:
        LSTMModelMeta TypedDict.
    """
    return {
        "backend": "lstm",
        "n_features": n_features,
        "sequence_length": config["sequence_length"],
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
        "bidirectional": config["bidirectional"],
        "dropout": config["dropout"],
    }


class _MetadataBuilder(ABC):
    """Abstract base for metadata builders.

    Each builder captures the config in the narrowed type context,
    then can build metadata once n_features is known.
    """

    @abstractmethod
    def build(self, n_features: int) -> ModelMeta:
        """Build model metadata with the given feature count."""


class _MlpMetadataBuilder(_MetadataBuilder):
    """Builds MLP model metadata."""

    def __init__(self, config: MLPConfig) -> None:
        self._config = config

    def build(self, n_features: int) -> MLPModelMeta:
        return _build_mlp_metadata(self._config, n_features)


class _LstmMetadataBuilder(_MetadataBuilder):
    """Builds LSTM model metadata."""

    def __init__(self, config: LSTMConfig) -> None:
        self._config = config

    def build(self, n_features: int) -> LSTMModelMeta:
        return _build_lstm_metadata(self._config, n_features)


class _LightgbmMetadataBuilder(_MetadataBuilder):
    """Builds LightGBM model metadata."""

    def build(self, n_features: int) -> LightGBMModelMeta:
        return {"backend": "lightgbm"}


class _LogRegMetadataBuilder(_MetadataBuilder):
    """Builds LogReg model metadata."""

    def __init__(self, config: LogRegConfig) -> None:
        self._config = config

    def build(self, n_features: int) -> LogRegModelMeta:
        return {
            "backend": "logreg",
            "n_features": n_features,
            "penalty": self._config["penalty"],
            "solver": self._config["solver"],
        }


class _RandomForestMetadataBuilder(_MetadataBuilder):
    """Builds Random Forest model metadata."""

    def __init__(self, config: RandomForestConfig) -> None:
        self._config = config

    def build(self, n_features: int) -> RandomForestModelMeta:
        return {
            "backend": "random_forest",
            "n_features": n_features,
            "n_estimators": self._config["n_estimators"],
            "max_depth": self._config["max_depth"],
        }


def _write_model_metadata(
    backend_name: BackendName,
    meta: ModelMeta,
    output_dir: Path,
) -> Path:
    """Write model metadata JSON to disk.

    Args:
        backend_name: Name of the ML backend.
        meta: Model metadata to save.
        output_dir: Directory where model is saved.

    Returns:
        Path to the saved metadata file.
    """
    meta_filename = _get_meta_filename(backend_name)
    meta_path = output_dir / meta_filename
    json_str = dump_json_str(meta, compact=False, indent=2)
    meta_path.write_text(json_str, encoding="utf-8")

    _log.info(
        "Saved model metadata",
        extra={
            "backend": backend_name,
            "meta_path": str(meta_path),
        },
    )

    return meta_path


# =============================================================================
# Dispatch: build config log + metadata builder from ParseResult
# =============================================================================
