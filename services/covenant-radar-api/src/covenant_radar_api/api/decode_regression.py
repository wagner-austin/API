"""Request decoding for the regression ML endpoints."""

from __future__ import annotations

from typing import Literal, TypedDict

from covenant_ml import FeaturePreset
from covenant_ml.explainers.types import SupportedExplainer
from covenant_ml.types_regression import RegressorBackendName
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    load_json_str,
    require_int,
    require_list,
    require_str,
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


def _parse_body_as_dict(body: bytes) -> JSONObject:
    """Parse request body as JSON dict. Raises on invalid JSON or non-dict."""
    raw = load_json_str(body.decode("utf-8"))
    if not isinstance(raw, dict):
        raise JSONTypeError("Request body must be a JSON object")
    return raw


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
    "RegressionExplainParseResult",
    "RegressionExplainRequest",
    "RegressionExplainResponse",
    "RegressionOptimizeApiParseResult",
    "RegressionPredictRequest",
    "RegressionPredictResponse",
    "parse_regression_explain_request",
    "parse_regression_optimize_request",
    "parse_regression_predict_request",
]
