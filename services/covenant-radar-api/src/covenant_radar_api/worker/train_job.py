"""Background job for ML model training.

Enhanced trainer with:
- Train/validation/test splits
- Comprehensive metrics (AUC, accuracy, precision, recall, F1, log loss)
- Early stopping based on validation AUC
- Progress callbacks for monitoring
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from covenant_domain.features import LoanFeatures, extract_features
from covenant_domain.models import CovenantResult, Deal
from covenant_ml.trainer import train_model_with_validation
from covenant_ml.types import EvalMetrics, TrainConfig, TrainOutcome
from covenant_persistence import (
    CovenantResultRepository,
    DealRepository,
    MeasurementRepository,
)
from numpy.typing import NDArray
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue, load_json_str
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
    """Extract optional float from dict, raising on wrong type."""
    raw = data.get(key)
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    raise JSONTypeError(f"Field '{key}' must be a number")


def _optional_int(data: JSONObject, key: str, default: int) -> int:
    """Extract optional int from dict, raising on wrong type."""
    raw = data.get(key)
    if raw is None:
        return default
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    raise JSONTypeError(f"Field '{key}' must be a number")


class TrainingDataProvider(Protocol):
    """Protocol for providing training data and repositories."""

    def deal_repo(self) -> DealRepository: ...

    def measurement_repo(self) -> MeasurementRepository: ...

    def covenant_result_repo(self) -> CovenantResultRepository: ...

    def get_sector_encoder(self) -> dict[str, int]: ...

    def get_region_encoder(self) -> dict[str, int]: ...

    def get_model_output_dir(self) -> Path: ...


def _parse_train_config(config_json: str) -> TrainConfig:
    """Parse training config from JSON string.

    Includes optional parameters for train/val/test splits with defaults:
    - device: auto
    - train_ratio: 0.7
    - val_ratio: 0.15
    - test_ratio: 0.15
    - early_stopping_rounds: 10
    - reg_alpha: 0.0
    - reg_lambda: 1.0
    - scale_pos_weight: None

    Ratios must sum to 1.0.
    """
    from platform_core.json_utils import require_float, require_int

    raw = load_json_str(config_json)
    if not isinstance(raw, dict):
        raise JSONTypeError("config must be a JSON object")

    # Optional parameters with defaults (70/15/15 split)
    train_ratio = _optional_float(raw, "train_ratio", 0.7)
    val_ratio = _optional_float(raw, "val_ratio", 0.15)
    test_ratio = _optional_float(raw, "test_ratio", 0.15)
    early_stopping_rounds = _optional_int(raw, "early_stopping_rounds", 10)
    reg_alpha = _optional_float(raw, "reg_alpha", 0.0)
    reg_lambda = _optional_float(raw, "reg_lambda", 1.0)

    scale_pos_weight_raw = raw.get("scale_pos_weight")
    scale_pos_weight: float | None = None
    if isinstance(scale_pos_weight_raw, (int, float)):
        scale_pos_weight = float(scale_pos_weight_raw)
    elif scale_pos_weight_raw is not None:
        raise JSONTypeError("scale_pos_weight must be a number")

    device = _parse_device(raw.get("device"))

    # Validate ratios sum to 1.0
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.3f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

    config: TrainConfig = {
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
    if scale_pos_weight is not None:
        config["scale_pos_weight"] = scale_pos_weight
    return config


def _extract_features_for_deal(
    deal: Deal,
    measurements_by_period: dict[str, dict[str, int]],
    recent_results: list[CovenantResult],
    sector_encoder: dict[str, int],
    region_encoder: dict[str, int],
) -> LoanFeatures:
    """Extract features for a single deal."""
    sorted_periods = sorted(measurements_by_period.keys(), reverse=True)

    metrics_current = measurements_by_period[sorted_periods[0]] if len(sorted_periods) > 0 else {}
    metrics_1p = measurements_by_period[sorted_periods[1]] if len(sorted_periods) > 1 else {}
    metrics_4p = measurements_by_period[sorted_periods[4]] if len(sorted_periods) > 4 else {}

    return extract_features(
        deal=deal,
        metrics_current=metrics_current,
        metrics_1p_ago=metrics_1p,
        metrics_4p_ago=metrics_4p,
        recent_results=recent_results,
        sector_encoder=sector_encoder,
        region_encoder=region_encoder,
    )


def _build_training_data(
    provider: TrainingDataProvider,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Build training data from repository data.

    Returns:
        Tuple of (features_array, labels_array)
    """
    deal_repo = provider.deal_repo()
    measurement_repo = provider.measurement_repo()
    result_repo = provider.covenant_result_repo()

    sector_encoder = provider.get_sector_encoder()
    region_encoder = provider.get_region_encoder()

    deals = deal_repo.list_all()
    features_list: list[LoanFeatures] = []
    labels: list[int] = []

    for deal in deals:
        deal_id = deal["id"]
        measurements = measurement_repo.list_for_deal(deal_id)
        results = result_repo.list_for_deal(deal_id)

        # Group measurements by period
        periods: dict[str, dict[str, int]] = {}
        for m in measurements:
            period_key = f"{m['period_start_iso']}_{m['period_end_iso']}"
            if period_key not in periods:
                periods[period_key] = {}
            periods[period_key][m["metric_name"]] = m["metric_value_scaled"]

        if len(periods) == 0:
            continue

        # Check if any result was a breach (label = 1) or not (label = 0)
        has_breach = False
        for r in results:
            if r["status"] == "BREACH":
                has_breach = True
                break

        features = _extract_features_for_deal(
            deal=deal,
            measurements_by_period=periods,
            recent_results=list(results),
            sector_encoder=sector_encoder,
            region_encoder=region_encoder,
        )
        features_list.append(features)
        labels.append(1 if has_breach else 0)

    # Convert to numpy arrays
    n_samples = len(features_list)
    n_features = 8
    x_array = np.zeros((n_samples, n_features), dtype=np.float64)

    for i, feat in enumerate(features_list):
        x_array[i, 0] = feat["debt_to_ebitda"]
        x_array[i, 1] = feat["interest_cover"]
        x_array[i, 2] = feat["current_ratio"]
        x_array[i, 3] = feat["leverage_change_1p"]
        x_array[i, 4] = feat["leverage_change_4p"]
        x_array[i, 5] = float(feat["sector_encoded"])
        x_array[i, 6] = float(feat["region_encoded"])
        x_array[i, 7] = float(feat["near_breach_count_4p"])

    y_array = np.array(labels, dtype=np.int64)
    return x_array, y_array


# Feature names in the same order as _build_training_data populates x_array
FEATURE_NAMES: list[str] = [
    "debt_to_ebitda",
    "interest_cover",
    "current_ratio",
    "leverage_change_1p",
    "leverage_change_4p",
    "sector_encoded",
    "region_encoded",
    "near_breach_count_4p",
]


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


def run_training(
    config_json: str,
    provider: TrainingDataProvider,
) -> dict[str, JSONValue]:
    """Run model training job with train/val/test splits.

    This function is designed to be called from an RQ worker.

    Args:
        config_json: JSON string with training configuration
        provider: Provider for training data and model output

    Returns:
        Job result with model_id and comprehensive training metrics.

    Raises:
        ValueError: Invalid config or insufficient training data
    """
    config = _parse_train_config(config_json)

    x_data, y_data = _build_training_data(provider)

    if len(x_data) == 0:
        raise ValueError("No training data available")

    # Minimum samples needed for train/val/test split
    min_samples = 10
    if len(x_data) < min_samples:
        raise ValueError(
            f"Insufficient training data: {len(x_data)} samples (minimum {min_samples} required)"
        )

    output_dir = provider.get_model_output_dir()

    _log.info(
        "Starting model training",
        extra={
            "n_samples": len(x_data),
            "n_features": x_data.shape[1],
            "config": {
                "device": config["device"],
                "learning_rate": config["learning_rate"],
                "n_estimators": config["n_estimators"],
                "train_ratio": config["train_ratio"],
                "val_ratio": config["val_ratio"],
                "reg_alpha": config["reg_alpha"],
                "reg_lambda": config["reg_lambda"],
                "scale_pos_weight": config.get("scale_pos_weight"),
            },
        },
    )

    # Train with validation and test splits
    outcome: TrainOutcome = train_model_with_validation(
        x_features=x_data,
        y_labels=y_data,
        config=config,
        output_dir=output_dir,
        feature_names=FEATURE_NAMES,
    )

    # Copy to active.ubj for API to load
    active_model_path = output_dir / "active.ubj"
    shutil.copy(outcome["model_path"], active_model_path)

    _log.info(
        "Training complete",
        extra={
            "model_id": outcome["model_id"],
            "samples_train": outcome["samples_train"],
            "samples_val": outcome["samples_val"],
            "samples_test": outcome["samples_test"],
            "best_val_auc": outcome["best_val_auc"],
            "test_auc": outcome["test_metrics"]["auc"],
            "early_stopped": outcome["early_stopped"],
        },
    )

    result_config: dict[str, JSONValue] = {
        "status": "complete",
        "model_id": outcome["model_id"],
        "model_path": outcome["model_path"],
        "active_model_path": str(active_model_path),
        "samples_total": outcome["samples_total"],
        "samples_train": outcome["samples_train"],
        "samples_val": outcome["samples_val"],
        "samples_test": outcome["samples_test"],
        "best_val_auc": outcome["best_val_auc"],
        "best_round": outcome["best_round"],
        "total_rounds": outcome["total_rounds"],
        "early_stopped": outcome["early_stopped"],
        "train_metrics": _metrics_to_json(outcome["train_metrics"]),
        "val_metrics": _metrics_to_json(outcome["val_metrics"]),
        "test_metrics": _metrics_to_json(outcome["test_metrics"]),
    }

    config_payload: dict[str, JSONValue] = {
        "device": config["device"],
        "learning_rate": config["learning_rate"],
        "max_depth": config["max_depth"],
        "n_estimators": config["n_estimators"],
        "subsample": config["subsample"],
        "colsample_bytree": config["colsample_bytree"],
        "random_state": config["random_state"],
        "train_ratio": config["train_ratio"],
        "val_ratio": config["val_ratio"],
        "test_ratio": config["test_ratio"],
        "early_stopping_rounds": config["early_stopping_rounds"],
        "reg_alpha": config["reg_alpha"],
        "reg_lambda": config["reg_lambda"],
    }
    scale_pos_weight = config.get("scale_pos_weight")
    if scale_pos_weight is not None:
        config_payload["scale_pos_weight"] = scale_pos_weight

    result_config["config"] = config_payload
    return result_config


def process_train_job(config_json: str) -> dict[str, JSONValue]:
    """RQ job entry point for model training.

    This is the function that RQ calls. It loads the ServiceContainer from
    environment variables and passes it as the provider to run_training.

    Args:
        config_json: JSON string with training configuration

    Returns:
        Job result with model_id and training metrics.
    """
    from covenant_radar_api.core.config import settings_from_env
    from covenant_radar_api.core.container import ServiceContainer

    settings = settings_from_env()

    # Get model output directory from settings (defaults to /data/models)
    model_output_dir = Path(settings["app"]["models_root"])

    # Ensure output directory exists
    model_output_dir.mkdir(parents=True, exist_ok=True)

    container = ServiceContainer.from_settings(
        settings,
        model_output_dir=model_output_dir,
    )

    result = run_training(config_json, container)

    container.close()
    return result


__all__ = ["TrainingDataProvider", "process_train_job", "run_training"]
