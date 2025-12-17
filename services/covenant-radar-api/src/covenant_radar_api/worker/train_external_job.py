"""Background job for training on external CSV data with automatic feature selection.

Trains XGBoost on ALL columns from external datasets (Taiwan, US, Polish).
The model automatically determines feature importance - no manual feature
engineering required.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, TypedDict

from covenant_ml.backends.registry import ClassifierRegistry
from covenant_ml.base_trainer import BaseTabularTrainer
from covenant_ml.datasets import LoadedDataset
from covenant_ml.types import (
    BackendName,
    ClassifierTrainConfig,
    EvalMetrics,
    FeatureImportance,
    LightGBMConfig,
    LightGBMModelMeta,
    LSTMConfig,
    LSTMModelMeta,
    MLPConfig,
    MLPModelMeta,
    ModelMeta,
    TrainConfig,
    TrainOutcome,
)
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_str,
)
from platform_core.logging import get_logger

_log = get_logger(__name__)


def _parse_device(raw: JSONValue | None) -> Literal["cpu", "cuda", "auto"]:
    """Parse device setting, defaulting to 'auto'."""
    if raw is None:
        return "auto"
    if not isinstance(raw, str):
        raise JSONTypeError("device must be a string")
    if raw == "cpu":
        return "cpu"
    if raw == "cuda":
        return "cuda"
    if raw == "auto":
        return "auto"
    raise ValueError("device must be one of: cpu, cuda, auto")


def _optional_float(data: JSONObject, key: str, default: float) -> float:
    """Extract optional float from dict."""
    raw = data.get(key)
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    raise JSONTypeError(f"Field '{key}' must be a number")


def _optional_int(data: JSONObject, key: str, default: int) -> int:
    """Extract optional int from dict."""
    raw = data.get(key)
    if raw is None:
        return default
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    raise JSONTypeError(f"Field '{key}' must be a number")


def _parse_mlp_precision(raw: JSONObject) -> Literal["fp32", "fp16", "bf16", "auto"]:
    """Parse and validate MLP precision field."""
    precision_val = raw.get("precision")
    if precision_val == "fp32":
        return "fp32"
    if precision_val == "fp16":
        return "fp16"
    if precision_val == "bf16":
        return "bf16"
    if precision_val == "auto":
        return "auto"
    raise JSONTypeError("precision must be fp32, fp16, bf16, or auto")


def _parse_mlp_optimizer(raw: JSONObject) -> Literal["adamw", "adam", "sgd"]:
    """Parse and validate MLP optimizer field."""
    optimizer_val = raw.get("optimizer")
    if optimizer_val == "adamw":
        return "adamw"
    if optimizer_val == "adam":
        return "adam"
    if optimizer_val == "sgd":
        return "sgd"
    raise JSONTypeError("optimizer must be adamw, adam, or sgd")


def _parse_mlp_hidden_sizes(raw: JSONObject) -> tuple[int, ...]:
    """Parse and validate hidden_sizes as tuple of ints."""
    hidden_sizes_val = raw.get("hidden_sizes")
    if not isinstance(hidden_sizes_val, list):
        raise JSONTypeError("hidden_sizes must be list of ints for mlp")
    result: list[int] = []
    for item in hidden_sizes_val:
        if not isinstance(item, int):
            raise JSONTypeError("hidden_sizes must be list of ints for mlp")
        result.append(item)
    return tuple(result)


def _parse_mlp_config(
    raw: JSONObject,
    device: Literal["cpu", "cuda", "auto"],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> MLPConfig:
    """Parse MLP backend config from JSON object."""
    from platform_core.json_utils import require_float, require_int

    return {
        "device": device,
        "precision": _parse_mlp_precision(raw),
        "optimizer": _parse_mlp_optimizer(raw),
        "hidden_sizes": _parse_mlp_hidden_sizes(raw),
        "learning_rate": require_float(raw, "learning_rate"),
        "batch_size": require_int(raw, "batch_size"),
        "n_epochs": require_int(raw, "n_epochs"),
        "dropout": require_float(raw, "dropout"),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_state": require_int(raw, "random_state"),
        "early_stopping_patience": require_int(raw, "early_stopping_patience"),
    }


def _parse_xgboost_config(
    raw: JSONObject,
    device: Literal["cpu", "cuda", "auto"],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> TrainConfig:
    """Parse XGBoost backend config from JSON object."""
    from platform_core.json_utils import require_float, require_int

    early_stopping_rounds = _optional_int(raw, "early_stopping_rounds", 10)
    reg_alpha = _optional_float(raw, "reg_alpha", 0.0)
    reg_lambda = _optional_float(raw, "reg_lambda", 1.0)
    xgb_cfg: TrainConfig = {
        "device": device,
        "learning_rate": require_float(raw, "learning_rate"),
        "max_depth": require_int(raw, "max_depth"),
        "n_estimators": require_int(raw, "n_estimators"),
        "subsample": require_float(raw, "subsample"),
        "colsample_bytree": require_float(raw, "colsample_bytree"),
        "random_state": require_int(raw, "random_state"),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "early_stopping_rounds": early_stopping_rounds,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
    }
    scale_pos_weight_raw = raw.get("scale_pos_weight")
    if isinstance(scale_pos_weight_raw, (int, float)):
        xgb_cfg["scale_pos_weight"] = float(scale_pos_weight_raw)
    elif scale_pos_weight_raw is not None:
        raise JSONTypeError("scale_pos_weight must be a number")
    return xgb_cfg


def _parse_lstm_precision(raw: JSONObject) -> Literal["fp32", "fp16", "bf16", "auto"]:
    """Parse and validate LSTM precision field."""
    precision_val = raw.get("precision")
    if precision_val == "fp32":
        return "fp32"
    if precision_val == "fp16":
        return "fp16"
    if precision_val == "bf16":
        return "bf16"
    if precision_val == "auto":
        return "auto"
    raise JSONTypeError("precision must be fp32, fp16, bf16, or auto")


def _parse_lstm_config(
    raw: JSONObject,
    device: Literal["cpu", "cuda", "auto"],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> LSTMConfig:
    """Parse LSTM backend config from JSON object."""
    from platform_core.json_utils import require_float, require_int

    bidirectional_val = raw.get("bidirectional")
    if not isinstance(bidirectional_val, bool):
        raise JSONTypeError("bidirectional must be a boolean")
    return {
        "device": device,
        "precision": _parse_lstm_precision(raw),
        "hidden_size": require_int(raw, "hidden_size"),
        "num_layers": require_int(raw, "num_layers"),
        "dropout": require_float(raw, "dropout"),
        "bidirectional": bidirectional_val,
        "sequence_length": require_int(raw, "sequence_length"),
        "learning_rate": require_float(raw, "learning_rate"),
        "batch_size": require_int(raw, "batch_size"),
        "n_epochs": require_int(raw, "n_epochs"),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_state": require_int(raw, "random_state"),
        "early_stopping_patience": require_int(raw, "early_stopping_patience"),
    }


def _parse_lightgbm_config(
    raw: JSONObject,
    device: Literal["cpu", "cuda", "auto"],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> LightGBMConfig:
    """Parse LightGBM backend config from JSON object."""
    from platform_core.json_utils import require_float, require_int

    early_stopping_rounds = _optional_int(raw, "early_stopping_rounds", 10)
    reg_alpha = _optional_float(raw, "reg_alpha", 0.0)
    reg_lambda = _optional_float(raw, "reg_lambda", 1.0)
    return {
        "device": device,
        "learning_rate": require_float(raw, "learning_rate"),
        "max_depth": require_int(raw, "max_depth"),
        "n_estimators": require_int(raw, "n_estimators"),
        "num_leaves": require_int(raw, "num_leaves"),
        "min_child_samples": require_int(raw, "min_child_samples"),
        "subsample": require_float(raw, "subsample"),
        "colsample_bytree": require_float(raw, "colsample_bytree"),
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_state": require_int(raw, "random_state"),
        "early_stopping_rounds": early_stopping_rounds,
    }


class XGBoostParseResult(TypedDict, total=True):
    """Result of parsing XGBoost config."""

    backend: Literal["xgboost"]
    config: TrainConfig
    dataset: str


class MLPParseResult(TypedDict, total=True):
    """Result of parsing MLP config."""

    backend: Literal["mlp"]
    config: MLPConfig
    dataset: str


class LSTMParseResult(TypedDict, total=True):
    """Result of parsing LSTM config."""

    backend: Literal["lstm"]
    config: LSTMConfig
    dataset: str


class LightGBMParseResult(TypedDict, total=True):
    """Result of parsing LightGBM config."""

    backend: Literal["lightgbm"]
    config: LightGBMConfig
    dataset: str


ParseResult = XGBoostParseResult | MLPParseResult | LSTMParseResult | LightGBMParseResult


def _parse_external_train_config(config_json: str) -> ParseResult:
    """Parse training config for external data.

    Returns:
        ParseResult with backend, config, and dataset_name
    """
    from platform_core.json_utils import require_str

    raw = load_json_str(config_json)
    if not isinstance(raw, dict):
        raise JSONTypeError("config must be a JSON object")

    # Dataset selection (required) - validate against registry
    from covenant_radar_api.worker import _test_hooks as hooks

    dataset = require_str(raw, "dataset")
    registry = hooks.dataset_registry_factory()
    if dataset not in registry:
        available = ", ".join(registry.list_names())
        raise ValueError(f"dataset must be one of: {available} (got {dataset})")
    dataset_name = dataset

    # Common split defaults
    train_ratio = _optional_float(raw, "train_ratio", 0.7)
    val_ratio = _optional_float(raw, "val_ratio", 0.15)
    test_ratio = _optional_float(raw, "test_ratio", 0.15)

    # Validate ratios sum to 1.0
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.3f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

    device = _parse_device(raw.get("device"))

    # Backend selection (optional; default xgboost)
    backend_val = raw.get("backend")
    if backend_val == "mlp":
        mlp_result: MLPParseResult = {
            "backend": "mlp",
            "config": _parse_mlp_config(raw, device, train_ratio, val_ratio, test_ratio),
            "dataset": dataset_name,
        }
        return mlp_result
    if backend_val == "lstm":
        lstm_result: LSTMParseResult = {
            "backend": "lstm",
            "config": _parse_lstm_config(raw, device, train_ratio, val_ratio, test_ratio),
            "dataset": dataset_name,
        }
        return lstm_result
    if backend_val == "lightgbm":
        lgbm_result: LightGBMParseResult = {
            "backend": "lightgbm",
            "config": _parse_lightgbm_config(raw, device, train_ratio, val_ratio, test_ratio),
            "dataset": dataset_name,
        }
        return lgbm_result
    xgb_result: XGBoostParseResult = {
        "backend": "xgboost",
        "config": _parse_xgboost_config(raw, device, train_ratio, val_ratio, test_ratio),
        "dataset": dataset_name,
    }
    return xgb_result


def _load_dataset(dataset_name: str, external_dir: Path) -> LoadedDataset:
    """Load the specified dataset using pluggable loader.

    Args:
        dataset_name: Name of dataset in registry.
        external_dir: Path to data/external directory.

    Returns:
        LoadedDataset with feature matrix, labels, and metadata.

    Raises:
        KeyError: If dataset not in registry.
        FileNotFoundError: If dataset file doesn't exist.
        ValueError: If data doesn't match expected format.
    """
    from covenant_radar_api.worker import _test_hooks as hooks

    registry = hooks.dataset_registry_factory()
    config = registry.get(dataset_name)
    return hooks.dataset_loader(config, external_dir)


def _metrics_to_json(metrics: EvalMetrics) -> dict[str, JSONValue]:
    """Convert EvalMetrics to JSON-serializable dict."""
    return {
        "loss": metrics["loss"],
        "ppl": metrics["ppl"],
        "auc": metrics["auc"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
    }


def _importance_to_json(imp: FeatureImportance) -> dict[str, JSONValue]:
    """Convert FeatureImportance to JSON-serializable dict."""
    return {
        "name": imp["name"],
        "importance": imp["importance"],
        "rank": imp["rank"],
    }


def _build_xgboost_log(config: TrainConfig) -> dict[str, JSONValue]:
    """Build log dict for XGBoost config."""
    return {
        "learning_rate": config["learning_rate"],
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "reg_alpha": config["reg_alpha"],
        "reg_lambda": config["reg_lambda"],
    }


def _build_mlp_log(config: MLPConfig) -> dict[str, JSONValue]:
    """Build log dict for MLP config."""
    return {
        "learning_rate": config["learning_rate"],
        "hidden_sizes": list(config["hidden_sizes"]),
        "n_epochs": config["n_epochs"],
        "dropout": config["dropout"],
    }


def _build_lstm_log(config: LSTMConfig) -> dict[str, JSONValue]:
    """Build log dict for LSTM config."""
    return {
        "learning_rate": config["learning_rate"],
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
        "n_epochs": config["n_epochs"],
        "bidirectional": config["bidirectional"],
        "sequence_length": config["sequence_length"],
    }


def _build_lightgbm_log(config: LightGBMConfig) -> dict[str, JSONValue]:
    """Build log dict for LightGBM config."""
    return {
        "learning_rate": config["learning_rate"],
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "num_leaves": config["num_leaves"],
        "reg_alpha": config["reg_alpha"],
        "reg_lambda": config["reg_lambda"],
    }


def _get_active_filename(backend_name: str) -> str:
    """Get active model filename for backend."""
    if backend_name == "xgboost":
        return "active_xgb.ubj"
    if backend_name == "mlp":
        return "active_mlp.pt"
    if backend_name == "lstm":
        return "active_lstm.pt"
    # lightgbm
    return "active_lgbm.txt"


def _get_meta_filename(backend_name: BackendName) -> str:
    """Get metadata filename for backend.

    Args:
        backend_name: Name of the ML backend.

    Returns:
        Filename for the metadata JSON file.
    """
    if backend_name == "mlp":
        return "active_mlp_meta.json"
    if backend_name == "lstm":
        return "active_lstm_meta.json"
    if backend_name == "lightgbm":
        return "active_lgbm_meta.json"
    # xgboost doesn't need metadata (self-describing format)
    return ""


def _build_mlp_metadata(config: MLPConfig, n_features: int) -> MLPModelMeta:
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


def _build_lstm_metadata(config: LSTMConfig, n_features: int) -> LSTMModelMeta:
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
        extra={"backend": backend_name, "meta_path": str(meta_path)},
    )

    return meta_path


def run_external_training(
    config_json: str,
    external_dir: Path,
    output_dir: Path,
) -> dict[str, JSONValue]:
    """Run training on external CSV data with automatic feature selection.

    XGBoost trains on ALL columns and determines which are most important.

    Args:
        config_json: JSON config with dataset name and hyperparameters
        external_dir: Path to data/external directory with datasets
        output_dir: Directory to save model artifacts

    Returns:
        Training result with model info, metrics, and feature importances
    """
    parse_result = _parse_external_train_config(config_json)
    dataset_name = parse_result["dataset"]
    backend_name = parse_result["backend"]

    # Load raw dataset with all columns
    dataset = _load_dataset(dataset_name, external_dir)

    # Build config log info based on backend (discriminated union narrowing)
    # Also prepare metadata builder for inference support (MLP/LSTM/LightGBM)
    config_log: dict[str, JSONValue]
    train_config: ClassifierTrainConfig
    metadata_builder: _MetadataBuilder | None = None
    if parse_result["backend"] == "xgboost":
        config_log = _build_xgboost_log(parse_result["config"])
        train_config = parse_result["config"]
        # XGBoost doesn't need metadata (self-describing format)
    elif parse_result["backend"] == "mlp":
        config_log = _build_mlp_log(parse_result["config"])
        train_config = parse_result["config"]
        metadata_builder = _MlpMetadataBuilder(parse_result["config"])
    elif parse_result["backend"] == "lstm":
        config_log = _build_lstm_log(parse_result["config"])
        train_config = parse_result["config"]
        metadata_builder = _LstmMetadataBuilder(parse_result["config"])
    else:
        # lightgbm
        config_log = _build_lightgbm_log(parse_result["config"])
        train_config = parse_result["config"]
        metadata_builder = _LightgbmMetadataBuilder()

    _log.info(
        "Starting external training",
        extra={
            "dataset": dataset_name,
            "n_samples": dataset["meta"]["n_samples"],
            "n_features": dataset["meta"]["n_features"],
            "n_positive": dataset["meta"]["n_positive"],
            "n_negative": dataset["meta"]["n_negative"],
            "backend": backend_name,
            "config": config_log,
        },
    )

    # Train via unified base trainer using default registry
    from covenant_radar_api.worker import _test_hooks as hooks

    reg_factory = hooks.registry_factory
    registry: ClassifierRegistry = reg_factory()
    trainer = BaseTabularTrainer(registry)

    outcome: TrainOutcome = trainer.train(
        backend=backend_name,
        x_features=dataset["x"],
        y_labels=dataset["y"],
        feature_names=list(dataset["meta"]["feature_names"]),
        config=train_config,
        output_dir=output_dir,
        progress=None,
    )

    # Copy to backend-specific active file (use copyfile to avoid permission issues)
    # XGBoost=.ubj, MLP/LSTM=.pt, LightGBM=.txt
    active_filename = _get_active_filename(backend_name)
    active_model_path = output_dir / active_filename
    shutil.copyfile(outcome["model_path"], active_model_path)

    # Save model metadata for inference loading (MLP/LSTM/LightGBM)
    meta_path: Path | None = None
    if metadata_builder is not None:
        meta = metadata_builder.build(dataset["meta"]["n_features"])
        meta_path = _write_model_metadata(backend_name, meta, output_dir)

    # Log top features
    top_features = outcome["feature_importances"][:10]
    _log.info(
        "Training complete - top features by importance",
        extra={
            "model_id": outcome["model_id"],
            "test_auc": outcome["test_metrics"]["auc"],
            "top_10_features": [
                {"rank": f["rank"], "name": f["name"], "importance": f"{f['importance']:.4f}"}
                for f in top_features
            ],
        },
    )

    # Build result
    result: dict[str, JSONValue] = {
        "status": "complete",
        "dataset": dataset_name,
        "model_id": outcome["model_id"],
        "model_path": outcome["model_path"],
        "active_model_path": str(active_model_path),
        "active_meta_path": str(meta_path) if meta_path is not None else None,
        "samples_total": outcome["samples_total"],
        "samples_train": outcome["samples_train"],
        "samples_val": outcome["samples_val"],
        "samples_test": outcome["samples_test"],
        "n_features": dataset["meta"]["n_features"],
        "scale_pos_weight": outcome["scale_pos_weight_computed"],
        "best_val_auc": outcome["best_val_auc"],
        "best_round": outcome["best_round"],
        "total_rounds": outcome["total_rounds"],
        "early_stopped": outcome["early_stopped"],
        "train_metrics": _metrics_to_json(outcome["train_metrics"]),
        "val_metrics": _metrics_to_json(outcome["val_metrics"]),
        "test_metrics": _metrics_to_json(outcome["test_metrics"]),
        "feature_importances": [_importance_to_json(f) for f in outcome["feature_importances"]],
    }

    return result


def process_external_train_job(config_json: str) -> dict[str, JSONValue]:
    """RQ job entry point for external data training.

    Args:
        config_json: JSON config with dataset name and hyperparameters

    Returns:
        Training result with model info and feature importances
    """
    from covenant_radar_api.core.config import settings_from_env

    settings = settings_from_env()

    # Get directories from settings
    data_root = Path(settings["app"]["data_root"])
    external_dir = data_root / "external"
    output_dir = Path(settings["app"]["models_root"])

    output_dir.mkdir(parents=True, exist_ok=True)

    return run_external_training(config_json, external_dir, output_dir)


__all__ = ["process_external_train_job", "run_external_training"]
