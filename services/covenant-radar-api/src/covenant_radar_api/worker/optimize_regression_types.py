"""Unified types for regression hyperparameter optimization jobs.

Regression-specific TypedDicts for config parsing and result serialization.
Separated from optimize_types.py (classifier) for clear separation of concerns.

Reuses shared encode/decode helpers from optimize_types for sampled params.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from covenant_ml.features import FeaturePreset
from covenant_ml.types_regression import RegressorBackendName
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
)

from covenant_radar_api.worker.optimize_field_decoders import (
    _require_feature_preset,
    _require_int,
    _require_str,
)
from covenant_radar_api.worker.optimize_regression_results import _require_regressor_backend_name

# =============================================================================
# Shared Validation Helpers (private, regression-specific)
# =============================================================================


def _require_device(raw: JSONObject) -> Literal["cpu", "cuda", "auto"]:
    """Extract and validate device from JSON object.

    Args:
        raw: JSON object.

    Returns:
        Device literal.

    Raises:
        JSONTypeError: If device field is invalid.
    """
    val = raw.get("device")
    if val is None:
        raise JSONTypeError("Missing required field 'device'")
    if val == "cpu":
        return "cpu"
    if val == "cuda":
        return "cuda"
    if val == "auto":
        return "auto"
    raise JSONTypeError("Field 'device' must be one of: cpu, cuda, auto")


def _require_precision(
    raw: JSONObject,
) -> Literal["fp32", "fp16", "bf16", "auto"]:
    """Extract and validate precision from JSON object.

    Args:
        raw: JSON object.

    Returns:
        Precision literal.

    Raises:
        JSONTypeError: If precision field is invalid.
    """
    val = raw.get("precision")
    if val is None:
        raise JSONTypeError("Missing required field 'precision'")
    if val == "fp32":
        return "fp32"
    if val == "fp16":
        return "fp16"
    if val == "bf16":
        return "bf16"
    if val == "auto":
        return "auto"
    raise JSONTypeError("Field 'precision' must be one of: fp32, fp16, bf16, auto")


def _require_nn_optimizer(
    raw: JSONObject,
) -> Literal["adamw", "adam", "sgd"]:
    """Extract and validate nn_optimizer from JSON object.

    Args:
        raw: JSON object.

    Returns:
        NN optimizer literal.

    Raises:
        JSONTypeError: If nn_optimizer field is invalid.
    """
    val = raw.get("nn_optimizer")
    if val is None:
        raise JSONTypeError("Missing required field 'nn_optimizer'")
    if val == "adamw":
        return "adamw"
    if val == "adam":
        return "adam"
    if val == "sgd":
        return "sgd"
    raise JSONTypeError("Field 'nn_optimizer' must be one of: adamw, adam, sgd")


def _require_bool(raw: JSONObject, key: str) -> bool:
    """Extract required bool from JSON object.

    Args:
        raw: JSON object.
        key: Field name.

    Returns:
        Boolean value.

    Raises:
        JSONTypeError: If field is missing or not a bool.
    """
    val = raw.get(key)
    if val is None:
        raise JSONTypeError(f"Missing required field '{key}'")
    if not isinstance(val, bool):
        raise JSONTypeError(f"Field '{key}' must be a boolean")
    return val


# =============================================================================
# Regression Config ParseResult
# =============================================================================


class UnifiedRegressionOptimizeParseResult(TypedDict, total=True):
    """Parsed optimization config for regression jobs.

    Parallel to UnifiedOptimizeParseResult but uses RegressorBackendName.
    Common fields are required for all backends. Backend-specific fields
    are parsed for all backends with sensible defaults and only consumed
    by the relevant backend's objective factory.

    Attributes:
        backend: Regressor backend to optimize.
        dataset: Regression dataset name.
        n_trials: Number of optimization trials.
        timeout_seconds: Optional timeout in seconds (None for no timeout).
        device: Compute device.
        feature_preset: Feature engineering preset.
        random_state: Random seed for reproducibility.
        early_stopping_rounds: Early stopping rounds (LightGBM: 10).
        n_jobs: Number of parallel jobs (LightGBM: -1).
        precision: Float precision (MLP/LSTM: "fp32").
        nn_optimizer: Neural network optimizer (MLP: "adamw").
        n_epochs: Training epochs per trial (MLP/LSTM: 50).
        early_stopping_patience: Early stopping patience (MLP/LSTM: 10).
        sequence_length: LSTM sequence length (LSTM: 5).
        bidirectional: Whether LSTM is bidirectional (LSTM: False).
    """

    backend: RegressorBackendName
    dataset: str
    n_trials: int
    timeout_seconds: int | None
    device: Literal["cpu", "cuda", "auto"]
    feature_preset: FeaturePreset
    random_state: int
    early_stopping_rounds: int
    n_jobs: int
    precision: Literal["fp32", "fp16", "bf16", "auto"]
    nn_optimizer: Literal["adamw", "adam", "sgd"]
    n_epochs: int
    early_stopping_patience: int
    sequence_length: int
    bidirectional: bool


def encode_unified_regression_optimize_parse_result(
    result: UnifiedRegressionOptimizeParseResult,
) -> JSONObject:
    """Encode UnifiedRegressionOptimizeParseResult to JSON-serializable dict.

    Args:
        result: Parsed regression optimization config.

    Returns:
        JSON-serializable dict.
    """
    encoded: JSONObject = {
        "backend": result["backend"],
        "dataset": result["dataset"],
        "n_trials": result["n_trials"],
        "timeout_seconds": result["timeout_seconds"],
        "device": result["device"],
        "feature_preset": result["feature_preset"],
        "random_state": result["random_state"],
        "early_stopping_rounds": result["early_stopping_rounds"],
        "n_jobs": result["n_jobs"],
        "precision": result["precision"],
        "nn_optimizer": result["nn_optimizer"],
        "n_epochs": result["n_epochs"],
        "early_stopping_patience": result["early_stopping_patience"],
        "sequence_length": result["sequence_length"],
        "bidirectional": result["bidirectional"],
    }
    return encoded


def decode_unified_regression_optimize_parse_result(
    raw: JSONObject,
) -> UnifiedRegressionOptimizeParseResult:
    """Decode a JSON object into UnifiedRegressionOptimizeParseResult.

    Args:
        raw: JSON object to decode.

    Returns:
        Validated UnifiedRegressionOptimizeParseResult.

    Raises:
        JSONTypeError: If any required field is missing or has wrong type.
    """
    timeout_val = raw.get("timeout_seconds")
    timeout_seconds: int | None = None
    if timeout_val is not None:
        if not isinstance(timeout_val, int):
            raise JSONTypeError("Field 'timeout_seconds' must be an integer or null")
        timeout_seconds = timeout_val

    return UnifiedRegressionOptimizeParseResult(
        backend=_require_regressor_backend_name(raw),
        dataset=_require_str(raw, "dataset"),
        n_trials=_require_int(raw, "n_trials"),
        timeout_seconds=timeout_seconds,
        device=_require_device(raw),
        feature_preset=_require_feature_preset(raw),
        random_state=_require_int(raw, "random_state"),
        early_stopping_rounds=_require_int(raw, "early_stopping_rounds"),
        n_jobs=_require_int(raw, "n_jobs"),
        precision=_require_precision(raw),
        nn_optimizer=_require_nn_optimizer(raw),
        n_epochs=_require_int(raw, "n_epochs"),
        early_stopping_patience=_require_int(raw, "early_stopping_patience"),
        sequence_length=_require_int(raw, "sequence_length"),
        bidirectional=_require_bool(raw, "bidirectional"),
    )


def require_unified_regression_optimize_parse_result(
    raw: JSONValue,
) -> UnifiedRegressionOptimizeParseResult:
    """Validate a JSONValue as UnifiedRegressionOptimizeParseResult.

    Args:
        raw: JSON value to validate.

    Returns:
        Validated UnifiedRegressionOptimizeParseResult.

    Raises:
        JSONTypeError: If value is not valid.
    """
    if not isinstance(raw, dict):
        raise JSONTypeError("Expected a JSON object for UnifiedRegressionOptimizeParseResult")
    return decode_unified_regression_optimize_parse_result(raw)


# =============================================================================
# Regression Optimization Result
# =============================================================================
