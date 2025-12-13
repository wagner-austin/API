"""Background job for training on external CSV data with automatic feature selection.

Trains XGBoost on ALL columns from external datasets (Taiwan, US, Polish).
The model automatically determines feature importance - no manual feature
engineering required.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal, TypedDict

from covenant_ml.backends.registry import ClassifierRegistry
from covenant_ml.base_trainer import BaseTabularTrainer
from covenant_ml.types import (
    ClassifierTrainConfig,
    EvalMetrics,
    FeatureImportance,
    MLPConfig,
    TrainConfig,
    TrainOutcome,
)
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue, load_json_str
from platform_core.logging import get_logger

from covenant_radar_api.seeding.real_data import (
    RawDataset,
    load_polish_raw,
    load_taiwan_raw,
    load_us_raw,
)

_log = get_logger(__name__)

DatasetName = Literal["taiwan", "us", "polish"]


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


class XGBoostParseResult(TypedDict, total=True):
    """Result of parsing XGBoost config."""

    backend: Literal["xgboost"]
    config: TrainConfig
    dataset: DatasetName


class MLPParseResult(TypedDict, total=True):
    """Result of parsing MLP config."""

    backend: Literal["mlp"]
    config: MLPConfig
    dataset: DatasetName


ParseResult = XGBoostParseResult | MLPParseResult


def _parse_external_train_config(config_json: str) -> ParseResult:
    """Parse training config for external data.

    Returns:
        ParseResult with backend, config, and dataset_name
    """
    from platform_core.json_utils import require_str

    raw = load_json_str(config_json)
    if not isinstance(raw, dict):
        raise JSONTypeError("config must be a JSON object")

    # Dataset selection (required)
    dataset = require_str(raw, "dataset")
    dataset_name: DatasetName
    if dataset == "taiwan":
        dataset_name = "taiwan"
    elif dataset == "us":
        dataset_name = "us"
    elif dataset == "polish":
        dataset_name = "polish"
    else:
        raise ValueError(f"dataset must be one of: taiwan, us, polish (got {dataset})")

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
    xgb_result: XGBoostParseResult = {
        "backend": "xgboost",
        "config": _parse_xgboost_config(raw, device, train_ratio, val_ratio, test_ratio),
        "dataset": dataset_name,
    }
    return xgb_result


def _load_dataset(dataset_name: DatasetName, external_dir: Path) -> RawDataset:
    """Load the specified dataset with all columns.

    Args:
        dataset_name: Which dataset to load ('taiwan', 'us', or 'polish')
        external_dir: Path to data/external directory

    Returns:
        RawDataset with feature matrix, labels, and column names

    Raises:
        FileNotFoundError: If dataset file doesn't exist
    """
    if dataset_name == "taiwan":
        data_path = external_dir / "taiwan_data" / "data.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Taiwan dataset not found at {data_path}")
        return load_taiwan_raw(data_path)
    if dataset_name == "us":
        data_path = external_dir / "us_data" / "american_bankruptcy.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"US dataset not found at {data_path}")
        return load_us_raw(data_path)
    # dataset_name == "polish"
    data_path = external_dir / "polish_data" / "1year.arff"
    if not data_path.exists():
        raise FileNotFoundError(f"Polish dataset not found at {data_path}")
    return load_polish_raw(data_path)


def _metrics_to_json(metrics: EvalMetrics) -> dict[str, JSONValue]:
    """Convert EvalMetrics to JSON-serializable dict."""
    return {
        "loss": metrics["loss"],
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
    if parse_result["backend"] == "xgboost":
        config_log = _build_xgboost_log(parse_result["config"])
        train_config: ClassifierTrainConfig = parse_result["config"]
    else:
        config_log = _build_mlp_log(parse_result["config"])
        train_config = parse_result["config"]

    _log.info(
        "Starting external training",
        extra={
            "dataset": dataset_name,
            "n_samples": dataset["n_samples"],
            "n_features": dataset["n_features"],
            "n_bankrupt": dataset["n_bankrupt"],
            "n_healthy": dataset["n_healthy"],
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
        feature_names=dataset["feature_names"],
        config=train_config,
        output_dir=output_dir,
        progress=None,
    )

    # Copy to active.ubj (use copyfile to avoid permission issues on Docker volumes)
    active_model_path = output_dir / "active.ubj"
    shutil.copyfile(outcome["model_path"], active_model_path)

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
        "samples_total": outcome["samples_total"],
        "samples_train": outcome["samples_train"],
        "samples_val": outcome["samples_val"],
        "samples_test": outcome["samples_test"],
        "n_features": dataset["n_features"],
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
