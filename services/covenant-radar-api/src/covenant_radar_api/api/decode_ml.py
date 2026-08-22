"""Request decoding for the ML endpoints: optimize and explain."""

from __future__ import annotations

from typing import Literal, TypedDict

from covenant_ml import FeaturePreset
from covenant_ml.explainers.types import SupportedExplainer
from covenant_ml.types import (
    BackendName,
)
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_int,
    require_str,
)

from covenant_radar_api.api.decode_regression import (
    _optional_int,
    _parse_body_as_dict,
    _parse_device,
    _parse_explainer,
    _parse_optimize_feature_preset,
)

DatasetName = Literal["taiwan", "us", "polish"]


def _optional_float(data: JSONObject, key: str, default: float) -> float:
    """Extract optional float from JSON, raising on wrong type."""
    raw = data.get(key)
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    raise JSONTypeError(f"Field '{key}' must be a number")


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
