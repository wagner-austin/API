"""Unified optimization result: shape and codec."""

from __future__ import annotations

from typing import Literal, TypedDict

from covenant_ml.features import FeaturePreset
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)
from covenant_ml.types import BackendName
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
    _require_backend_name,
    _require_feature_preset,
    _require_float,
    _require_int,
    _require_str,
)


class UnifiedOptimizationResult(TypedDict, total=True):
    """Result of a unified hyperparameter optimization run.

    Uses generic sampled param dicts instead of backend-specific best_* fields.

    Args:
        backend: Backend that was optimized.
        status: Always "complete".
        dataset: Dataset name used.
        n_samples: Number of samples in dataset.
        n_features: Number of features (after engineering).
        feature_preset: Feature preset used.
        n_trials_complete: Number of completed trials.
        n_trials_pruned: Number of pruned trials.
        n_trials_failed: Number of failed trials.
        best_trial_number: Trial number of best result.
        best_value: Best objective value (validation AUC).
        best_int_params: Best integer hyperparameters.
        best_float_params: Best float hyperparameters.
        best_string_params: Best string hyperparameters.
        duration_seconds: Total optimization duration.
    """

    backend: BackendName
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


# Param encoders live in _optimize_common so the classifier and regressor
# optimize paths share one implementation. They were previously duplicated,
# and the regressor copy silently omitted every neural-net key.


def encode_unified_optimization_result(
    result: UnifiedOptimizationResult,
) -> JSONObject:
    """Encode UnifiedOptimizationResult to a JSON-serializable dict.

    Args:
        result: Optimization result.

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


def _require_int_field(raw: JSONObject, field: str) -> int:
    """Require an integer value from a JSON object.

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
    """Require a float value from a JSON object.

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
    """Require a string value from a JSON object.

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


def _decode_int_tree_params(raw: JSONObject, result: SampledIntParams) -> None:
    """Decode tree/ensemble integer params from JSON into result dict.

    Args:
        raw: JSON object with integer parameter values.
        result: Mutable output dict to populate.

    Raises:
        JSONTypeError: If any value is not an integer.
    """
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


def _decode_int_nn_params(raw: JSONObject, result: SampledIntParams) -> None:
    """Decode neural-net and other integer params from JSON into result dict.

    Args:
        raw: JSON object with integer parameter values.
        result: Mutable output dict to populate.

    Raises:
        JSONTypeError: If any value is not an integer.
    """
    if "n_layers" in raw:
        result["n_layers"] = _require_int_field(raw, "n_layers")
    if "hidden_size" in raw:
        result["hidden_size"] = _require_int_field(raw, "hidden_size")
    if "num_layers" in raw:
        result["num_layers"] = _require_int_field(raw, "num_layers")
    if "batch_size" in raw:
        result["batch_size"] = _require_int_field(raw, "batch_size")
    if "max_bins" in raw:
        result["max_bins"] = _require_int_field(raw, "max_bins")
    if "max_iter" in raw:
        result["max_iter"] = _require_int_field(raw, "max_iter")


def _decode_sampled_int_params(raw: JSONObject) -> SampledIntParams:
    """Decode a JSON object into SampledIntParams.

    Args:
        raw: JSON object with integer parameter values.

    Returns:
        SampledIntParams with validated values.

    Raises:
        JSONTypeError: If any value is not an integer.
    """
    result = SampledIntParams()
    _decode_int_tree_params(raw, result)
    _decode_int_nn_params(raw, result)
    return result


def _decode_float_core_params(raw: JSONObject, result: SampledFloatParams) -> None:
    """Decode core float params from JSON into result dict.

    Args:
        raw: JSON object with float parameter values.
        result: Mutable output dict to populate.

    Raises:
        JSONTypeError: If any value is not a number.
    """
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
    if "dropout" in raw:
        result["dropout"] = _require_float_field(raw, "dropout")
    if "drop_rate" in raw:
        result["drop_rate"] = _require_float_field(raw, "drop_rate")


def _decode_float_extra_params(raw: JSONObject, result: SampledFloatParams) -> None:
    """Decode extra float params from JSON into result dict.

    Args:
        raw: JSON object with float parameter values.
        result: Mutable output dict to populate.

    Raises:
        JSONTypeError: If any value is not a number.
    """
    if "skip_drop" in raw:
        result["skip_drop"] = _require_float_field(raw, "skip_drop")
    if "rate_drop" in raw:
        result["rate_drop"] = _require_float_field(raw, "rate_drop")
    if "feature_fraction" in raw:
        result["feature_fraction"] = _require_float_field(raw, "feature_fraction")
    if "C" in raw:
        result["C"] = _require_float_field(raw, "C")
    if "tol" in raw:
        result["tol"] = _require_float_field(raw, "tol")
    if "l1_ratio" in raw:
        result["l1_ratio"] = _require_float_field(raw, "l1_ratio")
    if "max_features_float" in raw:
        result["max_features_float"] = _require_float_field(raw, "max_features_float")


def _decode_sampled_float_params(raw: JSONObject) -> SampledFloatParams:
    """Decode a JSON object into SampledFloatParams.

    Args:
        raw: JSON object with float parameter values.

    Returns:
        SampledFloatParams with validated values.

    Raises:
        JSONTypeError: If any value is not a number.
    """
    result = SampledFloatParams()
    _decode_float_core_params(raw, result)
    _decode_float_extra_params(raw, result)
    return result


def _decode_sampled_string_params(raw: JSONObject) -> SampledStringParams:
    """Decode a JSON object into SampledStringParams.

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
    if "penalty" in raw:
        result["penalty"] = _require_str_field(raw, "penalty")
    if "solver" in raw:
        result["solver"] = _require_str_field(raw, "solver")
    if "max_features" in raw:
        result["max_features"] = _require_str_field(raw, "max_features")
    return result


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


def decode_unified_optimization_result(
    raw: JSONObject,
) -> UnifiedOptimizationResult:
    """Decode a JSON object into UnifiedOptimizationResult.

    Args:
        raw: JSON object to decode.

    Returns:
        Validated UnifiedOptimizationResult.

    Raises:
        JSONTypeError: If any required field is missing or has wrong type.
    """
    status_val = _require_str(raw, "status")
    if status_val != "complete":
        raise JSONTypeError(f"Field 'status' must be 'complete' (got {status_val})")
    status: Literal["complete"] = "complete"

    return UnifiedOptimizationResult(
        backend=_require_backend_name(raw),
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


def require_unified_optimization_result(
    raw: JSONValue,
) -> UnifiedOptimizationResult:
    """Validate a JSONValue as UnifiedOptimizationResult.

    Args:
        raw: JSON value to validate.

    Returns:
        Validated UnifiedOptimizationResult.

    Raises:
        JSONTypeError: If value is not a valid UnifiedOptimizationResult.
    """
    if not isinstance(raw, dict):
        raise JSONTypeError("Expected a JSON object for UnifiedOptimizationResult")
    return decode_unified_optimization_result(raw)


# =============================================================================
# Callback Protocols
# =============================================================================
