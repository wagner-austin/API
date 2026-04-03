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
from covenant_ml.types import (
    BackendName,
    RegressorBackendName,
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

from covenant_radar_api.worker._train_external_parsers import (
    ParseResult as ExternalTrainParseResult,
)
from covenant_radar_api.worker._train_external_parsers import (
    parse_external_train_config,
)
from covenant_radar_api.worker._train_external_regression_parsers import (
    RegressionParseResult as ExternalRegressionTrainParseResult,
)
from covenant_radar_api.worker._train_external_regression_parsers import (
    parse_external_regression_train_config,
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
# Delegates to shared parsers in worker/_train_external_parsers.py.
# ExternalTrainParseResult is imported from there (no duplication).


DatasetName = Literal["taiwan", "us", "polish"]


def _parse_dataset_name(raw: JSONObject) -> DatasetName:
    """Parse and validate dataset name.

    Args:
        raw: JSON object containing the dataset field.

    Returns:
        Validated DatasetName literal.

    Raises:
        JSONTypeError: If dataset field is missing.
        ValueError: If dataset is not a valid name.
    """
    dataset = require_str(raw, "dataset")
    if dataset == "taiwan":
        return "taiwan"
    if dataset == "us":
        return "us"
    if dataset == "polish":
        return "polish"
    raise ValueError(f"dataset must be one of: taiwan, us, polish (got {dataset})")


def parse_external_train_request(
    body: bytes,
) -> ExternalTrainParseResult:
    """Parse request body for external training into backend-specific config.

    Delegates to shared parsers in worker/_train_external_parsers.py.
    Supports all 7 classifier backends via the 'backend' field.
    Default backend is 'xgboost' if not specified.

    Args:
        body: Raw HTTP request body bytes.

    Returns:
        ExternalTrainParseResult with backend type, config, and dataset.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
        ValueError: Invalid dataset name or split ratios don't sum to 1.0.
    """
    config_json = body.decode("utf-8")
    return parse_external_train_config(config_json)


def parse_external_regression_train_request(
    body: bytes,
) -> ExternalRegressionTrainParseResult:
    """Parse request body for external regression training.

    Delegates to shared parsers in worker/_train_external_regression_parsers.py.
    Supports xgboost_reg and lightgbm_reg backends.
    Default backend is 'xgboost_reg' if not specified.

    Args:
        body: Raw HTTP request body bytes.

    Returns:
        ExternalRegressionTrainParseResult with backend type, config,
        and regression dataset name.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
        ValueError: Invalid dataset name or split ratios don't sum to 1.0.
    """
    config_json = body.decode("utf-8")
    return parse_external_regression_train_config(config_json)


# --- Optimization Request Parsing ---


class OptimizeRequest(TypedDict, total=True):
    """Request body for hyperparameter optimization.

    Args:
        backend: ML backend name (all 7 backends supported).
        dataset: Dataset name for optimization.
        n_trials: Number of Optuna trials.
        timeout_seconds: Optional timeout in seconds.
        device: Compute device.
        feature_preset: Feature engineering preset.
        random_state: Random seed for reproducibility.
    """

    backend: BackendName
    dataset: DatasetName
    n_trials: int
    timeout_seconds: int | None
    device: Literal["cpu", "cuda", "auto"]
    feature_preset: FeaturePreset
    random_state: int


class OptimizeResponse(TypedDict, total=True):
    """Response body for optimization job submission."""

    job_id: str
    status: Literal["queued"]


class UnifiedOptimizeApiParseResult(TypedDict, total=True):
    """Parsed optimization request at the API edge.

    Only validates common fields. Backend-specific fields (precision,
    optimizer, n_epochs, etc.) are parsed by the worker job from the
    raw JSON body.

    Args:
        backend: ML backend name.
        dataset: Dataset name for optimization.
        n_trials: Number of Optuna trials.
        timeout_seconds: Optional timeout in seconds.
        device: Compute device.
        feature_preset: Feature engineering preset.
        random_state: Random seed for reproducibility.
    """

    backend: BackendName
    dataset: DatasetName
    n_trials: int
    timeout_seconds: int | None
    device: Literal["cpu", "cuda", "auto"]
    feature_preset: FeaturePreset
    random_state: int


def _parse_optimize_backend(raw: JSONValue | None) -> BackendName:
    """Parse optimize backend name, defaulting to 'xgboost'.

    Args:
        raw: Raw JSON value.

    Returns:
        BackendName literal.

    Raises:
        JSONTypeError: If value is not a string.
        ValueError: If value is not a valid backend.
    """
    if raw is None:
        return "xgboost"
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
    if raw == "cleargbm":
        return "cleargbm"
    if raw == "logreg":
        return "logreg"
    if raw == "random_forest":
        return "random_forest"
    raise ValueError(
        "backend must be one of: xgboost, mlp, lstm, lightgbm, cleargbm, logreg, random_forest"
    )


def _parse_optimize_feature_preset(raw: JSONValue | None) -> FeaturePreset:
    """Parse feature preset for optimization, defaulting to 'none'.

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


def parse_optimize_request(body: bytes) -> UnifiedOptimizeApiParseResult:
    """Parse request body for hyperparameter optimization.

    Validates common fields at the API edge. Backend-specific fields
    (precision, optimizer, n_epochs, etc.) are parsed by the worker job.
    All 7 classifier backends are supported.

    Request format:
        {
            "dataset": "taiwan" | "us" | "polish",  // required
            "backend": "xgboost",  // optional, default "xgboost"
            "n_trials": 50,  // required
            "timeout_seconds": 3600,  // optional, null for no timeout
            "device": "auto",  // optional, default "auto"
            "feature_preset": "none",  // optional: none, log_only, ratios_only, full
            "random_state": 42  // optional, default 42
        }

    Backend-specific fields (forwarded to worker, not validated here):
        XGBoost/ClearGBM/LightGBM: early_stopping_rounds, n_jobs
        MLP: precision, optimizer, n_epochs, early_stopping_patience
        LSTM: precision, n_epochs, early_stopping_patience, sequence_length, bidirectional
        LogReg/RandomForest: (no extra fields)

    Args:
        body: Raw request body bytes.

    Returns:
        UnifiedOptimizeApiParseResult with common parameters.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
        ValueError: Invalid dataset or backend name.
    """
    raw = _parse_body_as_dict(body)

    backend = _parse_optimize_backend(raw.get("backend"))
    dataset_name = _parse_dataset_name(raw)
    n_trials = require_int(raw, "n_trials")

    timeout_raw = raw.get("timeout_seconds")
    timeout_seconds: int | None = None
    if timeout_raw is not None:
        if not isinstance(timeout_raw, int):
            raise JSONTypeError("timeout_seconds must be an integer or null")
        timeout_seconds = timeout_raw

    device = _parse_device(raw.get("device"))
    feature_preset = _parse_optimize_feature_preset(raw.get("feature_preset"))
    random_state = _optional_int(raw, "random_state", 42)

    return UnifiedOptimizeApiParseResult(
        backend=backend,
        dataset=dataset_name,
        n_trials=n_trials,
        timeout_seconds=timeout_seconds,
        device=device,
        feature_preset=feature_preset,
        random_state=random_state,
    )


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
    if raw == "cleargbm":
        return "cleargbm"
    if raw == "logreg":
        return "logreg"
    if raw == "random_forest":
        return "random_forest"
    raise JSONTypeError(
        "backend must be one of: xgboost, mlp, lstm, lightgbm, cleargbm, logreg, random_forest"
    )


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


# --- Regression Optimization Request Parsing ---


class RegressionOptimizeApiParseResult(TypedDict, total=True):
    """Parsed regression optimization request at the API edge.

    Only validates common fields. Backend-specific fields are parsed
    by the worker job from the raw JSON body.

    Args:
        backend: Regressor backend name.
        dataset: Regression dataset name.
        n_trials: Number of Optuna trials.
        timeout_seconds: Optional timeout in seconds.
        device: Compute device.
        feature_preset: Feature engineering preset.
        random_state: Random seed for reproducibility.
    """

    backend: RegressorBackendName
    dataset: str
    n_trials: int
    timeout_seconds: int | None
    device: Literal["cpu", "cuda", "auto"]
    feature_preset: FeaturePreset
    random_state: int


def _parse_regressor_backend(raw: JSONValue | None) -> RegressorBackendName:
    """Parse regressor backend name, defaulting to 'xgboost_reg'.

    Args:
        raw: Raw JSON value.

    Returns:
        RegressorBackendName literal.

    Raises:
        JSONTypeError: If value is not a string.
        ValueError: If value is not a valid regressor backend.
    """
    if raw is None:
        return "xgboost_reg"
    if not isinstance(raw, str):
        raise JSONTypeError("backend must be a string")
    if raw == "xgboost_reg":
        return "xgboost_reg"
    if raw == "lightgbm_reg":
        return "lightgbm_reg"
    if raw == "mlp_reg":
        return "mlp_reg"
    if raw == "lstm_reg":
        return "lstm_reg"
    raise ValueError("backend must be one of: xgboost_reg, lightgbm_reg, mlp_reg, lstm_reg")


def _parse_regression_dataset_name(raw: JSONObject) -> str:
    """Parse and validate regression dataset name.

    Args:
        raw: JSON object containing the dataset field.

    Returns:
        Validated regression dataset name.

    Raises:
        JSONTypeError: If dataset field is missing.
        ValueError: If dataset is not in the regression registry.
    """
    from covenant_radar_api.worker._optimize_regression_common import (
        parse_regression_dataset_name,
    )

    dataset = require_str(raw, "dataset")
    return parse_regression_dataset_name(dataset)


def parse_regression_optimize_request(body: bytes) -> RegressionOptimizeApiParseResult:
    """Parse request body for regression hyperparameter optimization.

    Validates common fields at the API edge. Backend-specific fields
    (early_stopping_rounds, n_jobs) are parsed by the worker job.

    Request format:
        {
            "dataset": "financial_distress",  // required, regression dataset
            "backend": "xgboost_reg",  // optional, default "xgboost_reg"
            "n_trials": 50,  // required
            "timeout_seconds": 3600,  // optional, null for no timeout
            "device": "auto",  // optional, default "auto"
            "feature_preset": "none",  // optional: none, log_only, ratios_only, full
            "random_state": 42  // optional, default 42
        }

    Args:
        body: Raw request body bytes.

    Returns:
        RegressionOptimizeApiParseResult with common parameters.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
        ValueError: Invalid dataset or backend name.
    """
    raw = _parse_body_as_dict(body)

    backend = _parse_regressor_backend(raw.get("backend"))
    dataset_name = _parse_regression_dataset_name(raw)
    n_trials = require_int(raw, "n_trials")

    timeout_raw = raw.get("timeout_seconds")
    timeout_seconds: int | None = None
    if timeout_raw is not None:
        if not isinstance(timeout_raw, int):
            raise JSONTypeError("timeout_seconds must be an integer or null")
        timeout_seconds = timeout_raw

    device = _parse_device(raw.get("device"))
    feature_preset = _parse_optimize_feature_preset(raw.get("feature_preset"))
    random_state = _optional_int(raw, "random_state", 42)

    return RegressionOptimizeApiParseResult(
        backend=backend,
        dataset=dataset_name,
        n_trials=n_trials,
        timeout_seconds=timeout_seconds,
        device=device,
        feature_preset=feature_preset,
        random_state=random_state,
    )


class RegressionPredictRequest(TypedDict, total=True):
    """Parsed request for regression prediction.

    Attributes:
        backend: Regressor backend used for the model.
        model_path: Path to the trained regressor model file.
        features: 2D list of feature values (each inner list is one sample).
    """

    backend: RegressorBackendName
    model_path: str
    features: list[list[float]]


class RegressionPredictResponse(TypedDict, total=True):
    """Response body for regression prediction.

    Attributes:
        backend: Regressor backend used.
        predictions: Predicted continuous values (one per sample).
        n_samples: Number of samples predicted.
    """

    backend: RegressorBackendName
    predictions: list[float]
    n_samples: int


def parse_regression_predict_request(body: bytes) -> RegressionPredictRequest:
    """Parse request body for regression prediction.

    Request format:
        {
            "backend": "xgboost_reg" | "lightgbm_reg" | "mlp_reg" | "lstm_reg",
            "model_path": "/path/to/model",
            "features": [[1.0, 2.0, ...], [3.0, 4.0, ...]]
        }

    Args:
        body: Raw request bytes.

    Returns:
        RegressionPredictRequest with validated fields.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
        ValueError: Invalid backend name.
    """
    data = _parse_body_as_dict(body)

    backend = _parse_regressor_backend(data.get("backend"))
    model_path = require_str(data, "model_path")

    features_raw = require_list(data, "features")
    if len(features_raw) == 0:
        raise JSONTypeError("'features' must be a non-empty list of sample arrays")

    features: list[list[float]] = []
    for i, sample in enumerate(features_raw):
        if not isinstance(sample, list):
            raise JSONTypeError(f"features[{i}] must be a list of numbers")
        row: list[float] = []
        for j, val in enumerate(sample):
            if not isinstance(val, (int, float)):
                raise JSONTypeError(f"features[{i}][{j}] must be a number")
            row.append(float(val))
        features.append(row)

    return RegressionPredictRequest(
        backend=backend,
        model_path=model_path,
        features=features,
    )


# --- Regression Explain Request Parsing ---


class RegressionExplainRequest(TypedDict, total=True):
    """Request body for regression feature importance explanation.

    Args:
        dataset: Regression dataset name for loading data.
        backend: Regressor backend used for the model.
        model_path: Path to the trained regressor model file.
        explainer: Which explainer to use.
        n_samples: Number of samples to use for explanation.
        random_state: Random seed for reproducibility.
    """

    dataset: str
    backend: RegressorBackendName
    model_path: str
    explainer: SupportedExplainer
    n_samples: int
    random_state: int


class RegressionExplainResponse(TypedDict, total=True):
    """Response body for regression explain job submission."""

    job_id: str
    status: Literal["queued"]


class RegressionExplainParseResult(TypedDict, total=True):
    """Parsed regression explanation request for the worker job.

    Args:
        dataset: Regression dataset name for loading data.
        backend: Regressor backend used for the model.
        model_path: Path to the trained regressor model file.
        explainer: Which explainer to use.
        n_samples: Number of samples to use for explanation.
        random_state: Random seed for reproducibility.
    """

    dataset: str
    backend: RegressorBackendName
    model_path: str
    explainer: SupportedExplainer
    n_samples: int
    random_state: int


def parse_regression_explain_request(body: bytes) -> RegressionExplainParseResult:
    """Parse request body for regression feature importance explanation.

    Request format:
        {
            "dataset": "financial_distress",  // required, regression dataset
            "backend": "xgboost_reg" | ...,  // required
            "model_path": "/path/to/model.ubj",  // required
            "explainer": "permutation" | ...,  // required
            "n_samples": 1000,  // optional, default 1000
            "random_state": 42  // optional, default 42
        }

    Args:
        body: Raw request body bytes.

    Returns:
        RegressionExplainParseResult with all explanation parameters.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
        ValueError: Invalid dataset or backend name.
    """
    raw = _parse_body_as_dict(body)

    dataset_name = _parse_regression_dataset_name(raw)

    backend_raw = raw.get("backend")
    if backend_raw is None:
        raise JSONTypeError("Missing required field 'backend'")
    backend = _parse_regressor_backend(backend_raw)

    model_path = require_str(raw, "model_path")

    explainer_raw = raw.get("explainer")
    if explainer_raw is None:
        raise JSONTypeError("Missing required field 'explainer'")
    explainer = _parse_explainer(explainer_raw)

    n_samples = _optional_int(raw, "n_samples", 1000)
    random_state = _optional_int(raw, "random_state", 42)

    return RegressionExplainParseResult(
        dataset=dataset_name,
        backend=backend,
        model_path=model_path,
        explainer=explainer,
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
    "ExternalRegressionTrainParseResult",
    "ExternalTrainParseResult",
    "OptimizeRequest",
    "OptimizeResponse",
    "PredictRequest",
    "PredictResponse",
    "RegressionExplainParseResult",
    "RegressionExplainRequest",
    "RegressionExplainResponse",
    "RegressionOptimizeApiParseResult",
    "RegressionPredictRequest",
    "RegressionPredictResponse",
    "TrainResponse",
    "UnifiedOptimizeApiParseResult",
    "UpdateDealRequest",
    "parse_covenant_id_request",
    "parse_covenant_request",
    "parse_deal_id_request",
    "parse_deal_request",
    "parse_evaluate_request",
    "parse_explain_request",
    "parse_external_regression_train_request",
    "parse_external_train_request",
    "parse_measurements_request",
    "parse_optimize_request",
    "parse_predict_request",
    "parse_regression_explain_request",
    "parse_regression_optimize_request",
    "parse_regression_predict_request",
    "parse_train_request",
    "parse_update_deal_request",
]
