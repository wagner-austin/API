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
from covenant_ml import FeaturePreset
from covenant_ml.explainers.types import SupportedExplainer
from covenant_ml.optimizer import (
    LightGBMSearchSpace,
    LSTMSearchSpace,
    MLPSearchSpace,
    OptimizationConfig,
    XGBoostSearchSpace,
    make_default_optimization_config,
    make_lightgbm_default_space,
    make_lstm_default_space,
    make_mlp_default_space,
    make_xgboost_categorical_space,
    make_xgboost_default_space,
)
from covenant_ml.types import (
    BackendName,
    LightGBMConfig,
    LSTMConfig,
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
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


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


class LSTMParseResult(TypedDict, total=True):
    """Result of parsing LSTM config from external train request."""

    backend: Literal["lstm"]
    config: LSTMConfig
    dataset: DatasetName


class LightGBMParseResult(TypedDict, total=True):
    """Result of parsing LightGBM config from external train request."""

    backend: Literal["lightgbm"]
    config: LightGBMConfig
    dataset: DatasetName


ExternalTrainParseResult = (
    XGBoostParseResult | MLPParseResult | LSTMParseResult | LightGBMParseResult
)


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

    Supports XGBoost, MLP, LSTM, and LightGBM backends via the 'backend' field.
    Default backend is 'xgboost' if not specified.

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
        "config": _parse_xgboost_external_config(raw, device, train_ratio, val_ratio, test_ratio),
        "dataset": dataset_name,
    }
    return xgb_result


# --- Optimization Request Parsing ---


XGBoostSpaceProfile = Literal["default", "categorical"]
MLPSpaceProfile = Literal["default"]
LightGBMSpaceProfile = Literal["default"]
LSTMSpaceProfile = Literal["default"]


class OptimizeRequest(TypedDict, total=True):
    """Request body for hyperparameter optimization."""

    dataset: DatasetName
    n_trials: int
    timeout_seconds: int | None
    device: Literal["cpu", "cuda", "auto"]
    space_profile: XGBoostSpaceProfile
    feature_preset: FeaturePreset
    random_state: int


class OptimizeResponse(TypedDict, total=True):
    """Response body for optimization job submission."""

    job_id: str
    status: Literal["queued"]


class XGBoostOptimizeParseResult(TypedDict, total=True):
    """Parsed XGBoost optimization request.

    Args:
        backend: Backend identifier literal "xgboost".
        dataset: Dataset name for optimization.
        config: Optimization configuration.
        search_space: XGBoost-specific search space.
        space_profile: Search space profile used.
        device: Compute device.
        feature_preset: Feature engineering preset.
    """

    backend: Literal["xgboost"]
    dataset: DatasetName
    config: OptimizationConfig
    search_space: XGBoostSearchSpace
    space_profile: XGBoostSpaceProfile
    device: Literal["cpu", "cuda", "auto"]
    feature_preset: FeaturePreset


class MLPOptimizeParseResult(TypedDict, total=True):
    """Parsed MLP optimization request.

    Args:
        backend: Backend identifier literal "mlp".
        dataset: Dataset name for optimization.
        config: Optimization configuration.
        search_space: MLP-specific search space.
        space_profile: Search space profile used.
        device: Compute device.
        feature_preset: Feature engineering preset.
        precision: Floating-point precision.
        optimizer: Optimizer name.
        n_epochs: Number of training epochs per trial.
        early_stopping_patience: Early stopping patience.
    """

    backend: Literal["mlp"]
    dataset: DatasetName
    config: OptimizationConfig
    search_space: MLPSearchSpace
    space_profile: MLPSpaceProfile
    device: Literal["cpu", "cuda", "auto"]
    feature_preset: FeaturePreset
    precision: Literal["fp32", "fp16", "bf16", "auto"]
    optimizer: Literal["adamw", "adam", "sgd"]
    n_epochs: int
    early_stopping_patience: int


class LightGBMOptimizeParseResult(TypedDict, total=True):
    """Parsed LightGBM optimization request.

    Args:
        backend: Backend identifier literal "lightgbm".
        dataset: Dataset name for optimization.
        config: Optimization configuration.
        search_space: LightGBM-specific search space.
        space_profile: Search space profile used.
        device: Compute device.
        feature_preset: Feature engineering preset.
        early_stopping_rounds: Early stopping rounds per trial.
    """

    backend: Literal["lightgbm"]
    dataset: DatasetName
    config: OptimizationConfig
    search_space: LightGBMSearchSpace
    space_profile: LightGBMSpaceProfile
    device: Literal["cpu", "cuda", "auto"]
    feature_preset: FeaturePreset
    early_stopping_rounds: int


class LSTMOptimizeParseResult(TypedDict, total=True):
    """Parsed LSTM optimization request.

    Args:
        backend: Backend identifier literal "lstm".
        dataset: Dataset name for optimization.
        config: Optimization configuration.
        search_space: LSTM-specific search space.
        space_profile: Search space profile used.
        device: Compute device.
        feature_preset: Feature engineering preset.
        precision: Floating-point precision.
        n_epochs: Number of training epochs per trial.
        early_stopping_patience: Early stopping patience.
        sequence_length: Sequence length for LSTM.
        bidirectional: Whether LSTM is bidirectional.
    """

    backend: Literal["lstm"]
    dataset: DatasetName
    config: OptimizationConfig
    search_space: LSTMSearchSpace
    space_profile: LSTMSpaceProfile
    device: Literal["cpu", "cuda", "auto"]
    feature_preset: FeaturePreset
    precision: Literal["fp32", "fp16", "bf16", "auto"]
    n_epochs: int
    early_stopping_patience: int
    sequence_length: int
    bidirectional: bool


OptimizeParseResult = (
    XGBoostOptimizeParseResult
    | MLPOptimizeParseResult
    | LightGBMOptimizeParseResult
    | LSTMOptimizeParseResult
)


def _parse_xgboost_space_profile(raw: JSONValue | None) -> XGBoostSpaceProfile:
    """Parse XGBoost space profile, defaulting to 'default'.

    Args:
        raw: Raw JSON value.

    Returns:
        XGBoostSpaceProfile literal.

    Raises:
        JSONTypeError: If value is not a valid profile.
    """
    if raw is None:
        return "default"
    if not isinstance(raw, str):
        raise JSONTypeError("space_profile must be a string")
    if raw == "default":
        return "default"
    if raw == "categorical":
        return "categorical"
    raise JSONTypeError("space_profile must be one of: default, categorical")


def _parse_mlp_space_profile(raw: JSONValue | None) -> MLPSpaceProfile:
    """Parse MLP space profile, defaulting to 'default'.

    Args:
        raw: Raw JSON value.

    Returns:
        MLPSpaceProfile literal.

    Raises:
        JSONTypeError: If value is not a valid profile.
    """
    if raw is None:
        return "default"
    if not isinstance(raw, str):
        raise JSONTypeError("space_profile must be a string")
    if raw == "default":
        return "default"
    raise JSONTypeError("space_profile must be: default (mlp backend)")


def _parse_lightgbm_space_profile(raw: JSONValue | None) -> LightGBMSpaceProfile:
    """Parse LightGBM space profile, defaulting to 'default'.

    Args:
        raw: Raw JSON value.

    Returns:
        LightGBMSpaceProfile literal.

    Raises:
        JSONTypeError: If value is not a valid profile.
    """
    if raw is None:
        return "default"
    if not isinstance(raw, str):
        raise JSONTypeError("space_profile must be a string")
    if raw == "default":
        return "default"
    raise JSONTypeError("space_profile must be: default (lightgbm backend)")


def _parse_lstm_space_profile(raw: JSONValue | None) -> LSTMSpaceProfile:
    """Parse LSTM space profile, defaulting to 'default'.

    Args:
        raw: Raw JSON value.

    Returns:
        LSTMSpaceProfile literal.

    Raises:
        JSONTypeError: If value is not a valid profile.
    """
    if raw is None:
        return "default"
    if not isinstance(raw, str):
        raise JSONTypeError("space_profile must be a string")
    if raw == "default":
        return "default"
    raise JSONTypeError("space_profile must be: default (lstm backend)")


def _parse_feature_preset(raw: JSONValue | None) -> FeaturePreset:
    """Parse feature preset, defaulting to 'none'.

    Args:
        raw: Raw JSON value.

    Returns:
        FeaturePreset literal.

    Raises:
        JSONTypeError: If value is not a valid preset.
    """
    if raw is None:
        return "none"
    if not isinstance(raw, str):
        raise JSONTypeError("feature_preset must be a string")
    if raw == "none":
        return "none"
    if raw == "log_only":
        return "log_only"
    if raw == "ratios_only":
        return "ratios_only"
    if raw == "full":
        return "full"
    raise JSONTypeError("feature_preset must be one of: none, log_only, ratios_only, full")


def _parse_optimize_precision(raw: JSONValue | None) -> Literal["fp32", "fp16", "bf16", "auto"]:
    """Parse precision for neural network optimization.

    Args:
        raw: Raw JSON value.

    Returns:
        Precision literal.

    Raises:
        JSONTypeError: If value is not a valid precision.
    """
    if raw is None:
        return "fp32"
    if not isinstance(raw, str):
        raise JSONTypeError("precision must be a string")
    if raw == "fp32":
        return "fp32"
    if raw == "fp16":
        return "fp16"
    if raw == "bf16":
        return "bf16"
    if raw == "auto":
        return "auto"
    raise JSONTypeError("precision must be one of: fp32, fp16, bf16, auto")


def _parse_optimize_nn_optimizer(raw: JSONValue | None) -> Literal["adamw", "adam", "sgd"]:
    """Parse optimizer for neural network optimization.

    Args:
        raw: Raw JSON value.

    Returns:
        Optimizer literal.

    Raises:
        JSONTypeError: If value is not a valid optimizer.
    """
    if raw is None:
        return "adamw"
    if not isinstance(raw, str):
        raise JSONTypeError("optimizer must be a string")
    if raw == "adamw":
        return "adamw"
    if raw == "adam":
        return "adam"
    if raw == "sgd":
        return "sgd"
    raise JSONTypeError("optimizer must be one of: adamw, adam, sgd")


def _parse_bidirectional(raw: JSONValue | None) -> bool:
    """Parse bidirectional flag for LSTM optimization.

    Args:
        raw: Raw JSON value.

    Returns:
        Boolean value.

    Raises:
        JSONTypeError: If value is not a boolean.
    """
    if raw is None:
        return False
    if not isinstance(raw, bool):
        raise JSONTypeError("bidirectional must be a boolean")
    return raw


def _get_xgboost_search_space(profile: XGBoostSpaceProfile) -> XGBoostSearchSpace:
    """Get XGBoost search space based on profile name.

    Args:
        profile: Search space profile name.

    Returns:
        XGBoostSearchSpace with appropriate ranges.
    """
    if profile == "default":
        return make_xgboost_default_space()
    return make_xgboost_categorical_space()


def _parse_common_optimize_fields(
    raw: JSONObject,
) -> tuple[DatasetName, OptimizationConfig, Literal["cpu", "cuda", "auto"], FeaturePreset]:
    """Parse common optimization fields shared by all backends.

    Args:
        raw: Parsed JSON object.

    Returns:
        Tuple of (dataset, config, device, feature_preset).

    Raises:
        JSONTypeError: If required fields are missing or invalid.
        ValueError: If dataset name is invalid.
    """
    dataset_name = _parse_dataset_name(raw)
    n_trials = require_int(raw, "n_trials")

    timeout_raw = raw.get("timeout_seconds")
    timeout_seconds: int | None = None
    if timeout_raw is not None:
        if not isinstance(timeout_raw, int):
            raise JSONTypeError("timeout_seconds must be an integer or null")
        timeout_seconds = timeout_raw

    device = _parse_device(raw.get("device"))
    feature_preset = _parse_feature_preset(raw.get("feature_preset"))
    random_state = _optional_int(raw, "random_state", 42)

    config = make_default_optimization_config(
        n_trials=n_trials,
        timeout_seconds=timeout_seconds,
        random_state=random_state,
    )

    return dataset_name, config, device, feature_preset


def _parse_xgboost_optimize(raw: JSONObject) -> XGBoostOptimizeParseResult:
    """Parse XGBoost-specific optimization request.

    Args:
        raw: Parsed JSON object.

    Returns:
        XGBoostOptimizeParseResult with all parameters.

    Raises:
        JSONTypeError: If fields are invalid.
        ValueError: If dataset name is invalid.
    """
    dataset_name, config, device, feature_preset = _parse_common_optimize_fields(raw)
    space_profile = _parse_xgboost_space_profile(raw.get("space_profile"))
    search_space = _get_xgboost_search_space(space_profile)

    return XGBoostOptimizeParseResult(
        backend="xgboost",
        dataset=dataset_name,
        config=config,
        search_space=search_space,
        space_profile=space_profile,
        device=device,
        feature_preset=feature_preset,
    )


def _parse_mlp_optimize(raw: JSONObject) -> MLPOptimizeParseResult:
    """Parse MLP-specific optimization request.

    Args:
        raw: Parsed JSON object.

    Returns:
        MLPOptimizeParseResult with all parameters.

    Raises:
        JSONTypeError: If fields are invalid.
        ValueError: If dataset name is invalid.
    """
    dataset_name, config, device, feature_preset = _parse_common_optimize_fields(raw)
    space_profile = _parse_mlp_space_profile(raw.get("space_profile"))
    precision = _parse_optimize_precision(raw.get("precision"))
    optimizer = _parse_optimize_nn_optimizer(raw.get("optimizer"))
    n_epochs = _optional_int(raw, "n_epochs", 50)
    early_stopping_patience = _optional_int(raw, "early_stopping_patience", 10)

    return MLPOptimizeParseResult(
        backend="mlp",
        dataset=dataset_name,
        config=config,
        search_space=make_mlp_default_space(),
        space_profile=space_profile,
        device=device,
        feature_preset=feature_preset,
        precision=precision,
        optimizer=optimizer,
        n_epochs=n_epochs,
        early_stopping_patience=early_stopping_patience,
    )


def _parse_lightgbm_optimize(raw: JSONObject) -> LightGBMOptimizeParseResult:
    """Parse LightGBM-specific optimization request.

    Args:
        raw: Parsed JSON object.

    Returns:
        LightGBMOptimizeParseResult with all parameters.

    Raises:
        JSONTypeError: If fields are invalid.
        ValueError: If dataset name is invalid.
    """
    dataset_name, config, device, feature_preset = _parse_common_optimize_fields(raw)
    space_profile = _parse_lightgbm_space_profile(raw.get("space_profile"))
    early_stopping_rounds = _optional_int(raw, "early_stopping_rounds", 10)

    return LightGBMOptimizeParseResult(
        backend="lightgbm",
        dataset=dataset_name,
        config=config,
        search_space=make_lightgbm_default_space(),
        space_profile=space_profile,
        device=device,
        feature_preset=feature_preset,
        early_stopping_rounds=early_stopping_rounds,
    )


def _parse_lstm_optimize(raw: JSONObject) -> LSTMOptimizeParseResult:
    """Parse LSTM-specific optimization request.

    Args:
        raw: Parsed JSON object.

    Returns:
        LSTMOptimizeParseResult with all parameters.

    Raises:
        JSONTypeError: If fields are invalid.
        ValueError: If dataset name is invalid.
    """
    dataset_name, config, device, feature_preset = _parse_common_optimize_fields(raw)
    space_profile = _parse_lstm_space_profile(raw.get("space_profile"))
    precision = _parse_optimize_precision(raw.get("precision"))
    n_epochs = _optional_int(raw, "n_epochs", 50)
    early_stopping_patience = _optional_int(raw, "early_stopping_patience", 10)
    sequence_length = _optional_int(raw, "sequence_length", 5)
    bidirectional = _parse_bidirectional(raw.get("bidirectional"))

    return LSTMOptimizeParseResult(
        backend="lstm",
        dataset=dataset_name,
        config=config,
        search_space=make_lstm_default_space(),
        space_profile=space_profile,
        device=device,
        feature_preset=feature_preset,
        precision=precision,
        n_epochs=n_epochs,
        early_stopping_patience=early_stopping_patience,
        sequence_length=sequence_length,
        bidirectional=bidirectional,
    )


def parse_optimize_request(body: bytes) -> OptimizeParseResult:
    """Parse request body for hyperparameter optimization.

    Supports XGBoost, MLP, LightGBM, and LSTM backends via the 'backend' field.
    Default backend is 'xgboost' if not specified.

    Request format (XGBoost):
        {
            "dataset": "taiwan" | "us" | "polish",
            "backend": "xgboost",  // optional, default "xgboost"
            "n_trials": 50,  // required
            "timeout_seconds": 3600,  // optional, null for no timeout
            "device": "auto",  // optional, default "auto"
            "space_profile": "default",  // optional: default, categorical
            "feature_preset": "none",  // optional: none, log_only, ratios_only, full
            "random_state": 42  // optional, default 42
        }

    Request format (MLP):
        {
            "dataset": "taiwan" | "us" | "polish",
            "backend": "mlp",
            "n_trials": 50,
            "precision": "fp32",  // optional: fp32, fp16, bf16, auto
            "optimizer": "adamw",  // optional: adamw, adam, sgd
            "n_epochs": 50,  // optional, default 50
            "early_stopping_patience": 10  // optional, default 10
        }

    Request format (LightGBM):
        {
            "dataset": "taiwan" | "us" | "polish",
            "backend": "lightgbm",
            "n_trials": 50,
            "early_stopping_rounds": 10  // optional, default 10
        }

    Request format (LSTM):
        {
            "dataset": "taiwan" | "us" | "polish",
            "backend": "lstm",
            "n_trials": 50,
            "precision": "fp32",
            "n_epochs": 50,
            "early_stopping_patience": 10,
            "sequence_length": 5,  // optional, default 5
            "bidirectional": false  // optional, default false
        }

    Args:
        body: Raw request body bytes.

    Returns:
        OptimizeParseResult union with backend-specific parameters.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
        ValueError: Invalid dataset name.
    """
    raw = _parse_body_as_dict(body)

    # Backend selection (optional; default xgboost)
    backend_val = raw.get("backend")
    if backend_val == "mlp":
        return _parse_mlp_optimize(raw)
    if backend_val == "lightgbm":
        return _parse_lightgbm_optimize(raw)
    if backend_val == "lstm":
        return _parse_lstm_optimize(raw)
    # Default to xgboost (including explicit "xgboost" or None)
    if backend_val is not None and backend_val != "xgboost":
        raise JSONTypeError("backend must be one of: xgboost, mlp, lightgbm, lstm")
    return _parse_xgboost_optimize(raw)


# --- Explain Request Parsing ---


class ExplainRequest(TypedDict, total=True):
    """Request body for feature importance explanation.

    Args:
        dataset: Dataset name for loading data.
        backend: ML backend used for the model.
        model_path: Path to the trained model file.
        explainer: Which explainer to use.
        target_class: Class index for importance computation.
        n_samples: Number of samples to use for explanation.
        random_state: Random seed for reproducibility.
    """

    dataset: DatasetName
    backend: BackendName
    model_path: str
    explainer: SupportedExplainer
    target_class: int
    n_samples: int
    random_state: int


class ExplainResponse(TypedDict, total=True):
    """Response body for explain job submission."""

    job_id: str
    status: Literal["queued"]


class ExplainParseResult(TypedDict, total=True):
    """Parsed explanation request for the worker job.

    Args:
        dataset: Dataset name for loading data.
        backend: ML backend used for the model.
        model_path: Path to the trained model file.
        explainer: Which explainer to use.
        target_class: Class index for importance computation.
        n_samples: Number of samples to use for explanation.
        random_state: Random seed for reproducibility.
    """

    dataset: DatasetName
    backend: BackendName
    model_path: str
    explainer: SupportedExplainer
    target_class: int
    n_samples: int
    random_state: int


def _parse_explainer(raw: JSONValue) -> SupportedExplainer:
    """Parse and validate explainer name.

    Args:
        raw: Raw JSON value.

    Returns:
        Validated SupportedExplainer literal.

    Raises:
        JSONTypeError: If value is not a valid explainer name.
    """
    if not isinstance(raw, str):
        raise JSONTypeError("explainer must be a string")
    if raw == "permutation":
        return "permutation"
    if raw == "gradient":
        return "gradient"
    if raw == "integrated_gradients":
        return "integrated_gradients"
    if raw == "shap_tree":
        return "shap_tree"
    raise JSONTypeError(
        "explainer must be one of: permutation, gradient, integrated_gradients, shap_tree"
    )


def _parse_backend_name(raw: JSONValue) -> BackendName:
    """Parse and validate backend name.

    Args:
        raw: Raw JSON value.

    Returns:
        Validated BackendName literal.

    Raises:
        JSONTypeError: If value is not a valid backend name.
    """
    if not isinstance(raw, str):
        raise JSONTypeError("backend must be a string")
    if raw == "xgboost":
        return "xgboost"
    if raw == "mlp":
        return "mlp"
    if raw == "lstm":
        return "lstm"
    if raw == "lightgbm":
        return "lightgbm"
    raise JSONTypeError("backend must be one of: xgboost, mlp, lstm, lightgbm")


def parse_explain_request(body: bytes) -> ExplainParseResult:
    """Parse request body for feature importance explanation.

    Request format:
        {
            "dataset": "taiwan" | "us" | "polish",  // required
            "backend": "xgboost" | "mlp" | "lstm" | "lightgbm",  // required
            "model_path": "/path/to/model.ubj",  // required
            "explainer": "permutation" | "gradient" | ...,  // required
            "target_class": 1,  // optional, default 1
            "n_samples": 1000,  // optional, default 1000
            "random_state": 42  // optional, default 42
        }

    Returns:
        ExplainParseResult with all explanation parameters.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
        ValueError: Invalid dataset name.
    """
    raw = _parse_body_as_dict(body)

    # Required fields
    dataset_name = _parse_dataset_name(raw)

    backend_raw = raw.get("backend")
    if backend_raw is None:
        raise JSONTypeError("Missing required field 'backend'")
    backend = _parse_backend_name(backend_raw)

    model_path = require_str(raw, "model_path")

    explainer_raw = raw.get("explainer")
    if explainer_raw is None:
        raise JSONTypeError("Missing required field 'explainer'")
    explainer = _parse_explainer(explainer_raw)

    # Optional fields with defaults
    target_class = _optional_int(raw, "target_class", 1)
    n_samples = _optional_int(raw, "n_samples", 1000)
    random_state = _optional_int(raw, "random_state", 42)

    return ExplainParseResult(
        dataset=dataset_name,
        backend=backend,
        model_path=model_path,
        explainer=explainer,
        target_class=target_class,
        n_samples=n_samples,
        random_state=random_state,
    )


__all__ = [
    "AddMeasurementsRequest",
    "CreateCovenantRequest",
    "CreateDealRequest",
    "DatasetName",
    "EvaluateRequest",
    "ExplainParseResult",
    "ExplainRequest",
    "ExplainResponse",
    "ExternalTrainParseResult",
    "LSTMOptimizeParseResult",
    "LSTMParseResult",
    "LSTMSpaceProfile",
    "LightGBMOptimizeParseResult",
    "LightGBMParseResult",
    "LightGBMSpaceProfile",
    "MLPOptimizeParseResult",
    "MLPParseResult",
    "MLPSpaceProfile",
    "OptimizeParseResult",
    "OptimizeRequest",
    "OptimizeResponse",
    "PredictRequest",
    "PredictResponse",
    "TrainResponse",
    "UpdateDealRequest",
    "XGBoostOptimizeParseResult",
    "XGBoostParseResult",
    "XGBoostSpaceProfile",
    "parse_covenant_id_request",
    "parse_covenant_request",
    "parse_deal_id_request",
    "parse_deal_request",
    "parse_evaluate_request",
    "parse_explain_request",
    "parse_external_train_request",
    "parse_measurements_request",
    "parse_optimize_request",
    "parse_predict_request",
    "parse_train_request",
    "parse_update_deal_request",
]
