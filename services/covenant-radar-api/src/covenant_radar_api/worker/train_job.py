"""Background job for ML model training."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
from covenant_domain.features import LoanFeatures, extract_features
from covenant_domain.models import CovenantResult, Deal
from covenant_ml.trainer import save_model, train_model
from covenant_ml.types import TrainConfig
from covenant_persistence import (
    CovenantResultRepository,
    DealRepository,
    MeasurementRepository,
)
from numpy.typing import NDArray
from platform_core.json_utils import JSONTypeError, JSONValue, load_json_str


class TrainingDataProvider(Protocol):
    """Protocol for providing training data and repositories."""

    def deal_repo(self) -> DealRepository: ...

    def measurement_repo(self) -> MeasurementRepository: ...

    def covenant_result_repo(self) -> CovenantResultRepository: ...

    def get_sector_encoder(self) -> dict[str, int]: ...

    def get_region_encoder(self) -> dict[str, int]: ...

    def get_model_output_dir(self) -> Path: ...


def _parse_train_config(config_json: str) -> TrainConfig:
    """Parse training config from JSON string."""
    from platform_core.json_utils import require_float, require_int

    raw = load_json_str(config_json)
    if not isinstance(raw, dict):
        raise JSONTypeError("config must be a JSON object")
    return TrainConfig(
        learning_rate=require_float(raw, "learning_rate"),
        max_depth=require_int(raw, "max_depth"),
        n_estimators=require_int(raw, "n_estimators"),
        subsample=require_float(raw, "subsample"),
        colsample_bytree=require_float(raw, "colsample_bytree"),
        random_state=require_int(raw, "random_state"),
    )


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


def run_training(
    config_json: str,
    provider: TrainingDataProvider,
) -> dict[str, JSONValue]:
    """Run model training job.

    This function is designed to be called from an RQ worker.

    Args:
        config_json: JSON string with training configuration
        provider: Provider for training data and model output

    Returns:
        Job result with model_id and training metrics.

    Raises:
        ValueError: Invalid config or insufficient training data
    """
    config = _parse_train_config(config_json)

    x_train, y_train = _build_training_data(provider)

    if len(x_train) == 0:
        raise ValueError("No training data available")

    model = train_model(x_train, y_train, config)

    model_id = str(uuid.uuid4())
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    model_filename = f"covenant_model_{timestamp}_{model_id[:8]}.ubj"
    output_dir = provider.get_model_output_dir()
    model_path = output_dir / model_filename

    save_model(model, str(model_path))

    return {
        "status": "complete",
        "model_id": model_id,
        "model_path": str(model_path),
        "samples_trained": len(x_train),
        "config": {
            "learning_rate": config["learning_rate"],
            "max_depth": config["max_depth"],
            "n_estimators": config["n_estimators"],
            "subsample": config["subsample"],
            "colsample_bytree": config["colsample_bytree"],
            "random_state": config["random_state"],
        },
    }


__all__ = ["TrainingDataProvider", "run_training"]
