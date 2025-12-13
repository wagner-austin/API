"""HTTP request body parsing for covenant-radar-api.

Parses raw request bytes into strictly-typed domain models.
Uses platform_core.json_utils and covenant_domain.decode functions.
No framework validation (e.g., Pydantic) - internal decoders only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypedDict

from covenant_domain import (
    Covenant,
    CovenantId,
    Deal,
    DealId,
    Measurement,
    decode_covenant,
    decode_covenant_id,
    decode_deal,
    decode_deal_id,
    decode_measurement,
)
from covenant_ml.types import (
    MLPConfig,
    TrainConfig,
)
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    load_json_str,
    require_float,
    require_int,
    require_list,
    require_str,
)


class CreateDealRequest(TypedDict, total=True):
    """Request body for creating a new deal."""

    id: DealId
    name: str
    borrower: str
    sector: str
    region: str
    commitment_amount_cents: int
    currency: str
    maturity_date_iso: str


class UpdateDealRequest(TypedDict, total=True):
    """Request body for updating an existing deal."""

    name: str
    borrower: str
    sector: str
    region: str
    commitment_amount_cents: int
    currency: str
    maturity_date_iso: str


class CreateCovenantRequest(TypedDict, total=True):
    """Request body for creating a new covenant."""

    id: CovenantId
    deal_id: DealId
    name: str
    formula: str
    threshold_value_scaled: int
    threshold_direction: str
    frequency: str


class AddMeasurementsRequest(TypedDict, total=True):
    """Request body for adding measurements."""

    measurements: Sequence[Measurement]


class EvaluateRequest(TypedDict, total=True):
    """Request body for evaluating covenants for a deal and period."""

    deal_id: str
    period_start_iso: str
    period_end_iso: str
    tolerance_ratio_scaled: int


class PredictRequest(TypedDict, total=True):
    """Request body for predicting breach risk for a deal."""

    deal_id: str


class PredictResponse(TypedDict, total=True):
    """Response body for breach risk prediction."""

    deal_id: str
    probability: float
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"]


class TrainResponse(TypedDict, total=True):
    """Response body for training job submission."""

    job_id: str
    status: Literal["queued"]


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
    raise JSONTypeError("device must be one of: cpu, cuda, auto")


def _optional_float(data: JSONObject, key: str, default: float) -> float:
    """Extract optional float from JSON, raising on wrong type."""
    raw = data.get(key)
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    raise JSONTypeError(f"Field '{key}' must be a number")


def _optional_int(data: JSONObject, key: str, default: int) -> int:
    """Extract optional int from JSON, raising on wrong type."""
    raw = data.get(key)
    if raw is None:
        return default
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    raise JSONTypeError(f"Field '{key}' must be a number")


def _parse_body_as_dict(body: bytes) -> JSONObject:
    """Parse request body as JSON dict. Raises on invalid JSON or non-dict."""
    raw = load_json_str(body.decode("utf-8"))
    if not isinstance(raw, dict):
        raise JSONTypeError("Request body must be a JSON object")
    return raw


def parse_deal_request(body: bytes) -> Deal:
    """Parse request body into Deal.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
    """
    data = _parse_body_as_dict(body)
    return decode_deal(data)


def parse_deal_id_request(body: bytes) -> DealId:
    """Parse request body into DealId.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
    """
    data = _parse_body_as_dict(body)
    return decode_deal_id(data)


def parse_update_deal_request(body: bytes, deal_id: DealId) -> Deal:
    """Parse update request body into Deal with provided ID.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
    """
    data = _parse_body_as_dict(body)
    return Deal(
        id=deal_id,
        name=require_str(data, "name"),
        borrower=require_str(data, "borrower"),
        sector=require_str(data, "sector"),
        region=require_str(data, "region"),
        commitment_amount_cents=require_int(data, "commitment_amount_cents"),
        currency=require_str(data, "currency"),
        maturity_date_iso=require_str(data, "maturity_date_iso"),
    )


def parse_covenant_request(body: bytes) -> Covenant:
    """Parse request body into Covenant.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
    """
    data = _parse_body_as_dict(body)
    return decode_covenant(data)


def parse_covenant_id_request(body: bytes) -> CovenantId:
    """Parse request body into CovenantId.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
    """
    data = _parse_body_as_dict(body)
    return decode_covenant_id(data)


def parse_measurements_request(body: bytes) -> list[Measurement]:
    """Parse request body into list of Measurements.

    Expects: {"measurements": [...]}

    Raises:
        JSONTypeError: Missing required field or invalid field type.
    """
    data = _parse_body_as_dict(body)
    raw_list = require_list(data, "measurements")
    result: list[Measurement] = []
    for item in raw_list:
        if not isinstance(item, dict):
            raise JSONTypeError("Each measurement must be a JSON object")
        result.append(decode_measurement(item))
    return result


def parse_evaluate_request(body: bytes) -> EvaluateRequest:
    """Parse request body into EvaluateRequest.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
    """
    data = _parse_body_as_dict(body)
    return EvaluateRequest(
        deal_id=require_str(data, "deal_id"),
        period_start_iso=require_str(data, "period_start_iso"),
        period_end_iso=require_str(data, "period_end_iso"),
        tolerance_ratio_scaled=require_int(data, "tolerance_ratio_scaled"),
    )


def parse_predict_request(body: bytes) -> PredictRequest:
    """Parse request body into PredictRequest.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
    """
    data = _parse_body_as_dict(body)
    return PredictRequest(deal_id=require_str(data, "deal_id"))


def parse_train_request(body: bytes) -> TrainConfig:
    """Parse request body into TrainConfig.

    Optional fields with defaults:
    - device: "auto"
    - train_ratio: 0.7
    - val_ratio: 0.15
    - test_ratio: 0.15
    - early_stopping_rounds: 10
    - reg_alpha: 0.0
    - reg_lambda: 1.0
    - scale_pos_weight: None

    Raises:
        JSONTypeError: Missing required field or invalid field type.
    """
    data = _parse_body_as_dict(body)

    # Required fields
    learning_rate = require_float(data, "learning_rate")
    max_depth = require_int(data, "max_depth")
    n_estimators = require_int(data, "n_estimators")
    subsample = require_float(data, "subsample")
    colsample_bytree = require_float(data, "colsample_bytree")
    random_state = require_int(data, "random_state")

    device = _parse_device(data.get("device"))

    # Optional fields with defaults
    train_ratio = _optional_float(data, "train_ratio", 0.7)
    val_ratio = _optional_float(data, "val_ratio", 0.15)
    test_ratio = _optional_float(data, "test_ratio", 0.15)
    early_stopping_rounds = _optional_int(data, "early_stopping_rounds", 10)
    reg_alpha = _optional_float(data, "reg_alpha", 0.0)
    reg_lambda = _optional_float(data, "reg_lambda", 1.0)

    scale_pos_weight_raw = data.get("scale_pos_weight")
    scale_pos_weight: float | None = None
    if isinstance(scale_pos_weight_raw, (int, float)):
        scale_pos_weight = float(scale_pos_weight_raw)
    elif scale_pos_weight_raw is not None:
        raise JSONTypeError("scale_pos_weight must be a number")

    train_config: TrainConfig = {
        "device": device,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "random_state": random_state,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "early_stopping_rounds": early_stopping_rounds,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
    }
    if scale_pos_weight is not None:
        train_config["scale_pos_weight"] = scale_pos_weight
    return train_config


# --- External Training Request Parsing ---


DatasetName = Literal["taiwan", "us", "polish"]


class XGBoostParseResult(TypedDict, total=True):
    """Result of parsing XGBoost config from external train request."""

    backend: Literal["xgboost"]
    config: TrainConfig
    dataset: DatasetName


class MLPParseResult(TypedDict, total=True):
    """Result of parsing MLP config from external train request."""

    backend: Literal["mlp"]
    config: MLPConfig
    dataset: DatasetName


ExternalTrainParseResult = XGBoostParseResult | MLPParseResult


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


def _parse_dataset_name(raw: JSONObject) -> DatasetName:
    """Parse and validate dataset name."""
    dataset = require_str(raw, "dataset")
    if dataset == "taiwan":
        return "taiwan"
    if dataset == "us":
        return "us"
    if dataset == "polish":
        return "polish"
    raise ValueError(f"dataset must be one of: taiwan, us, polish (got {dataset})")


def _parse_mlp_config(
    raw: JSONObject,
    device: Literal["cpu", "cuda", "auto"],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> MLPConfig:
    """Parse MLP backend config from JSON object."""
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


def _parse_xgboost_external_config(
    raw: JSONObject,
    device: Literal["cpu", "cuda", "auto"],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> TrainConfig:
    """Parse XGBoost backend config from JSON object for external training."""
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


def parse_external_train_request(body: bytes) -> ExternalTrainParseResult:
    """Parse request body for external training into backend-specific config.

    Supports both XGBoost and MLP backends via the 'backend' field.
    Default backend is 'xgboost' if not specified.

    Request format for XGBoost:
        {
            "dataset": "taiwan" | "us" | "polish",
            "backend": "xgboost",  // optional, default
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "device": "auto",  // optional
            "train_ratio": 0.7,  // optional
            "val_ratio": 0.15,  // optional
            "test_ratio": 0.15,  // optional
            "early_stopping_rounds": 10,  // optional
            "reg_alpha": 0.0,  // optional
            "reg_lambda": 1.0,  // optional
            "scale_pos_weight": 2.5  // optional
        }

    Request format for MLP:
        {
            "dataset": "taiwan" | "us" | "polish",
            "backend": "mlp",
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "hidden_sizes": [64, 32],
            "precision": "fp32" | "fp16" | "bf16" | "auto",
            "optimizer": "adamw" | "adam" | "sgd",
            "random_state": 42,
            "early_stopping_patience": 10,
            "device": "auto",  // optional
            "train_ratio": 0.7,  // optional
            "val_ratio": 0.15,  // optional
            "test_ratio": 0.15  // optional
        }

    Returns:
        ExternalTrainParseResult with backend type, config, and dataset name.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
        ValueError: Invalid dataset name or split ratios don't sum to 1.0.
    """
    raw = _parse_body_as_dict(body)

    # Dataset selection (required)
    dataset_name = _parse_dataset_name(raw)

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
        "config": _parse_xgboost_external_config(raw, device, train_ratio, val_ratio, test_ratio),
        "dataset": dataset_name,
    }
    return xgb_result


__all__ = [
    "AddMeasurementsRequest",
    "CreateCovenantRequest",
    "CreateDealRequest",
    "DatasetName",
    "EvaluateRequest",
    "ExternalTrainParseResult",
    "MLPParseResult",
    "PredictRequest",
    "PredictResponse",
    "TrainResponse",
    "UpdateDealRequest",
    "XGBoostParseResult",
    "parse_covenant_id_request",
    "parse_covenant_request",
    "parse_deal_id_request",
    "parse_deal_request",
    "parse_evaluate_request",
    "parse_external_train_request",
    "parse_measurements_request",
    "parse_predict_request",
    "parse_train_request",
    "parse_update_deal_request",
]
