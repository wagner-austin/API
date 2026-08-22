"""Unified regression optimization result: shape, codec, progress."""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

from covenant_ml.datasets.types import LoadPhase
from covenant_ml.features import FeaturePreset
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)
from covenant_ml.types_regression import RegressorBackendName
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
)

from covenant_radar_api.worker._optimize_param_codec import (
    encode_sampled_float_params,
    encode_sampled_int_params,
    encode_sampled_string_params,
)
from covenant_radar_api.worker.optimize_field_decoders import (
    _require_float,
    _require_int,
    _require_str,
)


def _require_regressor_backend_name(raw: JSONObject) -> RegressorBackendName:
    """Extract and validate regressor backend name from JSON object.

    Args:
        raw: JSON object.

    Returns:
        Validated RegressorBackendName.

    Raises:
        JSONTypeError: If backend field is missing or invalid.
    """
    val = raw.get("backend")
    if val is None:
        raise JSONTypeError("Missing required field 'backend'")
    if not isinstance(val, str):
        raise JSONTypeError("Field 'backend' must be a string")
    valid: tuple[str, ...] = ("xgboost_reg", "lightgbm_reg", "mlp_reg", "lstm_reg")
    if val not in valid:
        raise JSONTypeError(f"Field 'backend' must be one of: {', '.join(valid)} (got {val})")
    if val == "xgboost_reg":
        return "xgboost_reg"
    if val == "lightgbm_reg":
        return "lightgbm_reg"
    if val == "mlp_reg":
        return "mlp_reg"
    return "lstm_reg"


def _require_json_object(raw: JSONObject, key: str) -> JSONObject:
    """Extract required JSON object from parent object.

    Args:
        raw: Parent JSON object.
        key: Field name.

    Returns:
        Nested JSON object.

    Raises:
        JSONTypeError: If field is missing or not an object.
    """
    val = raw.get(key)
    if val is None:
        raise JSONTypeError(f"Missing required field '{key}'")
    if not isinstance(val, dict):
        raise JSONTypeError(f"Field '{key}' must be a JSON object")
    return val


def _require_feature_preset(raw: JSONObject) -> FeaturePreset:
    """Extract and validate feature_preset from JSON object.

    Args:
        raw: JSON object.

    Returns:
        FeaturePreset literal.

    Raises:
        JSONTypeError: If feature_preset field is invalid.
    """
    val = raw.get("feature_preset")
    if val is None:
        raise JSONTypeError("Missing required field 'feature_preset'")
    if val == "none":
        return "none"
    if val == "log_only":
        return "log_only"
    if val == "ratios_only":
        return "ratios_only"
    if val == "full":
        return "full"
    raise JSONTypeError("Field 'feature_preset' must be one of: none, log_only, ratios_only, full")


class UnifiedRegressionOptimizationResult(TypedDict, total=True):
    """Result of a regression hyperparameter optimization run.

    Parallel to UnifiedOptimizationResult but uses RegressorBackendName.
    best_value is negative RMSE (higher = better, since Optuna maximizes).

    Attributes:
        backend: Regressor backend that was optimized.
        status: Always "complete".
        dataset: Regression dataset name used.
        n_samples: Number of samples in dataset.
        n_features: Number of features (after engineering).
        feature_preset: Feature preset used.
        n_trials_complete: Number of completed trials.
        n_trials_pruned: Number of pruned trials.
        n_trials_failed: Number of failed trials.
        best_trial_number: Trial number of best result.
        best_value: Best objective value (negative RMSE).
        best_int_params: Best integer hyperparameters.
        best_float_params: Best float hyperparameters.
        best_string_params: Best string hyperparameters.
        duration_seconds: Total optimization duration.
    """

    backend: RegressorBackendName
    status: Literal["complete"]
    dataset: str
    n_samples: int
    n_features: int
    feature_preset: FeaturePreset
    n_trials_complete: int
    n_trials_pruned: int
    n_trials_failed: int
    best_trial_number: int
    best_value: float
    best_int_params: SampledIntParams
    best_float_params: SampledFloatParams
    best_string_params: SampledStringParams
    duration_seconds: float


# Param encoders live in _optimize_common, shared with the classifier path.
# This module previously kept its own copies which omitted every neural-net
# key (n_layers, hidden_size, num_layers, batch_size, dropout, ...), so tuned
# mlp_reg/lstm_reg hyperparameters were silently dropped from results.


def _require_int_field(raw: JSONObject, field: str) -> int:
    """Require an integer value from a nested JSON object.

    Args:
        raw: JSON object.
        field: Field name.

    Returns:
        Integer value.

    Raises:
        JSONTypeError: If field value is not an integer.
    """
    val = raw[field]
    if not isinstance(val, int):
        raise JSONTypeError(f"Field '{field}' must be an integer")
    return val


def _require_float_field(raw: JSONObject, field: str) -> float:
    """Require a float value from a nested JSON object.

    Args:
        raw: JSON object.
        field: Field name.

    Returns:
        Float value.

    Raises:
        JSONTypeError: If field value is not a number.
    """
    val = raw[field]
    if not isinstance(val, (int, float)):
        raise JSONTypeError(f"Field '{field}' must be a number")
    return float(val)


def _require_str_field(raw: JSONObject, field: str) -> str:
    """Require a string value from a nested JSON object.

    Args:
        raw: JSON object.
        field: Field name.

    Returns:
        String value.

    Raises:
        JSONTypeError: If field value is not a string.
    """
    val = raw[field]
    if not isinstance(val, str):
        raise JSONTypeError(f"Field '{field}' must be a string")
    return val


def _decode_sampled_int_params(raw: JSONObject) -> SampledIntParams:
    """Decode a JSON object into SampledIntParams (regression-relevant keys).

    Args:
        raw: JSON object with integer parameter values.

    Returns:
        SampledIntParams with validated values.

    Raises:
        JSONTypeError: If any value is not an integer.
    """
    result = SampledIntParams()
    if "max_depth" in raw:
        result["max_depth"] = _require_int_field(raw, "max_depth")
    if "n_estimators" in raw:
        result["n_estimators"] = _require_int_field(raw, "n_estimators")
    if "num_leaves" in raw:
        result["num_leaves"] = _require_int_field(raw, "num_leaves")
    if "min_child_samples" in raw:
        result["min_child_samples"] = _require_int_field(raw, "min_child_samples")
    if "min_samples_split" in raw:
        result["min_samples_split"] = _require_int_field(raw, "min_samples_split")
    if "min_samples_leaf" in raw:
        result["min_samples_leaf"] = _require_int_field(raw, "min_samples_leaf")
    return result


def _decode_sampled_float_params(raw: JSONObject) -> SampledFloatParams:
    """Decode a JSON object into SampledFloatParams (regression-relevant keys).

    Args:
        raw: JSON object with float parameter values.

    Returns:
        SampledFloatParams with validated values.

    Raises:
        JSONTypeError: If any value is not a number.
    """
    result = SampledFloatParams()
    if "learning_rate" in raw:
        result["learning_rate"] = _require_float_field(raw, "learning_rate")
    if "reg_alpha" in raw:
        result["reg_alpha"] = _require_float_field(raw, "reg_alpha")
    if "reg_lambda" in raw:
        result["reg_lambda"] = _require_float_field(raw, "reg_lambda")
    if "subsample" in raw:
        result["subsample"] = _require_float_field(raw, "subsample")
    if "colsample_bytree" in raw:
        result["colsample_bytree"] = _require_float_field(raw, "colsample_bytree")
    if "drop_rate" in raw:
        result["drop_rate"] = _require_float_field(raw, "drop_rate")
    if "skip_drop" in raw:
        result["skip_drop"] = _require_float_field(raw, "skip_drop")
    if "rate_drop" in raw:
        result["rate_drop"] = _require_float_field(raw, "rate_drop")
    if "feature_fraction" in raw:
        result["feature_fraction"] = _require_float_field(raw, "feature_fraction")
    return result


def _decode_sampled_string_params(raw: JSONObject) -> SampledStringParams:
    """Decode a JSON object into SampledStringParams (regression-relevant keys).

    Args:
        raw: JSON object with string parameter values.

    Returns:
        SampledStringParams with validated values.

    Raises:
        JSONTypeError: If any value is not a string.
    """
    result = SampledStringParams()
    if "boosting_type" in raw:
        result["boosting_type"] = _require_str_field(raw, "boosting_type")
    if "booster" in raw:
        result["booster"] = _require_str_field(raw, "booster")
    return result


def encode_unified_regression_optimization_result(
    result: UnifiedRegressionOptimizationResult,
) -> JSONObject:
    """Encode UnifiedRegressionOptimizationResult to JSON-serializable dict.

    Args:
        result: Regression optimization result.

    Returns:
        JSON-serializable dict.
    """
    return {
        "backend": result["backend"],
        "status": result["status"],
        "dataset": result["dataset"],
        "n_samples": result["n_samples"],
        "n_features": result["n_features"],
        "feature_preset": result["feature_preset"],
        "n_trials_complete": result["n_trials_complete"],
        "n_trials_pruned": result["n_trials_pruned"],
        "n_trials_failed": result["n_trials_failed"],
        "best_trial_number": result["best_trial_number"],
        "best_value": result["best_value"],
        "best_int_params": encode_sampled_int_params(result["best_int_params"]),
        "best_float_params": encode_sampled_float_params(result["best_float_params"]),
        "best_string_params": encode_sampled_string_params(result["best_string_params"]),
        "duration_seconds": result["duration_seconds"],
    }


def decode_unified_regression_optimization_result(
    raw: JSONObject,
) -> UnifiedRegressionOptimizationResult:
    """Decode a JSON object into UnifiedRegressionOptimizationResult.

    Args:
        raw: JSON object to decode.

    Returns:
        Validated UnifiedRegressionOptimizationResult.

    Raises:
        JSONTypeError: If any required field is missing or has wrong type.
    """
    status_val = _require_str(raw, "status")
    if status_val != "complete":
        raise JSONTypeError(f"Field 'status' must be 'complete' (got {status_val})")
    status: Literal["complete"] = "complete"

    return UnifiedRegressionOptimizationResult(
        backend=_require_regressor_backend_name(raw),
        status=status,
        dataset=_require_str(raw, "dataset"),
        n_samples=_require_int(raw, "n_samples"),
        n_features=_require_int(raw, "n_features"),
        feature_preset=_require_feature_preset(raw),
        n_trials_complete=_require_int(raw, "n_trials_complete"),
        n_trials_pruned=_require_int(raw, "n_trials_pruned"),
        n_trials_failed=_require_int(raw, "n_trials_failed"),
        best_trial_number=_require_int(raw, "best_trial_number"),
        best_value=_require_float(raw, "best_value"),
        best_int_params=_decode_sampled_int_params(
            _require_json_object(raw, "best_int_params"),
        ),
        best_float_params=_decode_sampled_float_params(
            _require_json_object(raw, "best_float_params"),
        ),
        best_string_params=_decode_sampled_string_params(
            _require_json_object(raw, "best_string_params"),
        ),
        duration_seconds=_require_float(raw, "duration_seconds"),
    )


def require_unified_regression_optimization_result(
    raw: JSONValue,
) -> UnifiedRegressionOptimizationResult:
    """Validate a JSONValue as UnifiedRegressionOptimizationResult.

    Args:
        raw: JSON value to validate.

    Returns:
        Validated UnifiedRegressionOptimizationResult.

    Raises:
        JSONTypeError: If value is not valid.
    """
    if not isinstance(raw, dict):
        raise JSONTypeError("Expected a JSON object for UnifiedRegressionOptimizationResult")
    return decode_unified_regression_optimization_result(raw)


# =============================================================================
# Regression Progress TypedDicts
# =============================================================================


RegressionOptimizePhase = Literal["loading_data", "feature_engineering", "optimizing", "saving"]


class RegressionPhaseProgressInfo(TypedDict, total=True):
    """Information about regression optimization phase transitions.

    Attributes:
        phase: Current optimization phase.
        backend: Regressor backend being optimized.
        dataset: Dataset name.
        n_samples: Number of samples (0 during loading).
        n_features: Number of features (0 during loading).
    """

    phase: RegressionOptimizePhase
    backend: RegressorBackendName
    dataset: str
    n_samples: int
    n_features: int


class RegressionTrialProgressInfo(TypedDict, total=True):
    """Backend-agnostic trial progress for regression optimization.

    Attributes:
        backend: Regressor backend being optimized.
        trial_number: Current trial number (0-indexed).
        n_trials_total: Total number of trials requested.
        current_value: Objective value of current trial (negative RMSE).
        best_value: Best objective value seen so far.
        best_trial: Trial number of the best result.
        is_best: Whether this trial is the new best.
    """

    backend: RegressorBackendName
    trial_number: int
    n_trials_total: int
    current_value: float
    best_value: float
    best_trial: int
    is_best: bool


class RegressionLoadingProgressInfo(TypedDict, total=True):
    """Granular progress during regression dataset loading.

    Attributes:
        dataset: Dataset name being loaded.
        phase: Loading sub-phase (reading, parsing, encoding).
        percent_complete: Percentage complete (0.0-100.0).
        rows_processed: Number of rows processed so far.
        rows_total: Total number of rows.
        message: Human-readable progress message.
    """

    dataset: str
    phase: LoadPhase
    percent_complete: float
    rows_processed: int
    rows_total: int
    message: str


# =============================================================================
# Regression Progress Callback Protocols
# =============================================================================


class RegressionPhaseProgressCallbackProtocol(Protocol):
    """Protocol for regression phase progress callback."""

    def __call__(self, info: RegressionPhaseProgressInfo) -> None:
        """Called when entering a new regression optimization phase.

        Args:
            info: Regression phase transition information.
        """
        ...


class RegressionLoadingProgressCallbackProtocol(Protocol):
    """Protocol for regression loading progress callback."""

    def __call__(self, info: RegressionLoadingProgressInfo) -> None:
        """Called with progress updates during regression dataset loading.

        Args:
            info: Regression loading progress information.
        """
        ...


class RegressionTrialProgressCallbackProtocol(Protocol):
    """Protocol for regression trial progress callback."""

    def __call__(self, info: RegressionTrialProgressInfo) -> None:
        """Called after each regression optimization trial with progress info.

        Args:
            info: Regression trial progress information.
        """
        ...


__all__ = [
    "RegressionLoadingProgressCallbackProtocol",
    "RegressionLoadingProgressInfo",
    "RegressionOptimizePhase",
    "RegressionPhaseProgressCallbackProtocol",
    "RegressionPhaseProgressInfo",
    "RegressionTrialProgressCallbackProtocol",
    "RegressionTrialProgressInfo",
    "UnifiedRegressionOptimizationResult",
    "decode_unified_regression_optimization_result",
    "encode_unified_regression_optimization_result",
    "require_unified_regression_optimization_result",
]
