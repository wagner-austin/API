"""Unified types for hyperparameter optimization jobs.

Backend-agnostic TypedDicts for progress reporting, config parsing,
and result serialization. Used by the unified optimize_job.py.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

from covenant_ml.datasets.types import LoadPhase
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

from ._optimize_common import (
    encode_sampled_float_params,
    encode_sampled_int_params,
    encode_sampled_string_params,
)

# =============================================================================
# Phase Literals
# =============================================================================

OptimizePhase = Literal["loading_data", "feature_engineering", "optimizing", "saving"]

# =============================================================================
# Progress TypedDicts
# =============================================================================


class PhaseProgressInfo(TypedDict, total=True):
    """Information about optimization phase transitions.

    Args:
        phase: Current optimization phase.
        backend: Backend being optimized.
        dataset: Dataset name.
        n_samples: Number of samples (0 during loading).
        n_features: Number of features (0 during loading).
    """

    phase: OptimizePhase
    backend: BackendName
    dataset: str
    n_samples: int
    n_features: int


class LoadingProgressInfo(TypedDict, total=True):
    """Granular progress during dataset loading.

    Args:
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


class TrialProgressInfo(TypedDict, total=True):
    """Backend-agnostic trial progress information.

    Args:
        backend: Backend being optimized.
        trial_number: Current trial number (0-indexed).
        n_trials_total: Total number of trials requested.
        current_value: Objective value of current trial.
        best_value: Best objective value seen so far.
        best_trial: Trial number of the best result.
        is_best: Whether this trial is the new best.
    """

    backend: BackendName
    trial_number: int
    n_trials_total: int
    current_value: float
    best_value: float
    best_trial: int
    is_best: bool


# =============================================================================
# Config ParseResult
# =============================================================================


class UnifiedOptimizeParseResult(TypedDict, total=True):
    """Parsed optimization config for the unified job.

    Common fields are required for all backends. Backend-specific fields
    are parsed for all backends with sensible defaults and only consumed
    by the relevant backend's objective factory.

    Args:
        backend: Backend to optimize.
        dataset: Dataset name.
        n_trials: Number of optimization trials.
        timeout_seconds: Optional timeout in seconds (None for no timeout).
        device: Compute device.
        feature_preset: Feature engineering preset.
        random_state: Random seed for reproducibility.
        early_stopping_rounds: Early stopping rounds (LightGBM/ClearGBM: 10).
        n_jobs: Number of parallel jobs (LightGBM: -1).
        precision: Float precision (MLP/LSTM: "fp32").
        nn_optimizer: Neural network optimizer (MLP: "adamw").
        n_epochs: Training epochs per trial (MLP/LSTM: 50).
        early_stopping_patience: Early stopping patience (MLP/LSTM: 10).
        sequence_length: LSTM sequence length (LSTM: 5).
        bidirectional: Whether LSTM is bidirectional (LSTM: False).
    """

    backend: BackendName
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


def encode_unified_optimize_parse_result(
    result: UnifiedOptimizeParseResult,
) -> JSONObject:
    """Encode UnifiedOptimizeParseResult to a JSON-serializable dict.

    Args:
        result: Parsed optimization config.

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


def _require_backend_name(raw: JSONObject) -> BackendName:
    """Extract and validate backend name from JSON object.

    Args:
        raw: JSON object.

    Returns:
        Validated BackendName.

    Raises:
        JSONTypeError: If backend field is missing or invalid.
    """
    val = raw.get("backend")
    if val is None:
        raise JSONTypeError("Missing required field 'backend'")
    if not isinstance(val, str):
        raise JSONTypeError("Field 'backend' must be a string")
    valid: tuple[str, ...] = (
        "xgboost",
        "mlp",
        "lstm",
        "lightgbm",
        "cleargbm",
        "logreg",
        "random_forest",
    )
    if val not in valid:
        raise JSONTypeError(f"Field 'backend' must be one of: {', '.join(valid)} (got {val})")
    # Narrow to BackendName via explicit matching
    if val == "xgboost":
        return "xgboost"
    if val == "mlp":
        return "mlp"
    if val == "lstm":
        return "lstm"
    if val == "lightgbm":
        return "lightgbm"
    if val == "cleargbm":
        return "cleargbm"
    if val == "logreg":
        return "logreg"
    return "random_forest"


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
    if val == "temporal":
        return "temporal"
    raise JSONTypeError(
        "Field 'feature_preset' must be one of: none, log_only, ratios_only, full, temporal"
    )


def _require_precision(raw: JSONObject) -> Literal["fp32", "fp16", "bf16", "auto"]:
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


def _require_nn_optimizer(raw: JSONObject) -> Literal["adamw", "adam", "sgd"]:
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


def _require_int(raw: JSONObject, key: str) -> int:
    """Extract required int from JSON object.

    Args:
        raw: JSON object.
        key: Field name.

    Returns:
        Integer value.

    Raises:
        JSONTypeError: If field is missing or not an integer.
    """
    val = raw.get(key)
    if val is None:
        raise JSONTypeError(f"Missing required field '{key}'")
    if not isinstance(val, int):
        raise JSONTypeError(f"Field '{key}' must be an integer")
    return val


def _require_bool(raw: JSONObject, key: str) -> bool:
    """Extract required bool from JSON object.

    Args:
        raw: JSON object.
        key: Field name.

    Returns:
        Boolean value.

    Raises:
        JSONTypeError: If field is missing or not a boolean.
    """
    val = raw.get(key)
    if val is None:
        raise JSONTypeError(f"Missing required field '{key}'")
    if not isinstance(val, bool):
        raise JSONTypeError(f"Field '{key}' must be a boolean")
    return val


def _require_str(raw: JSONObject, key: str) -> str:
    """Extract required string from JSON object.

    Args:
        raw: JSON object.
        key: Field name.

    Returns:
        String value.

    Raises:
        JSONTypeError: If field is missing or not a string.
    """
    val = raw.get(key)
    if val is None:
        raise JSONTypeError(f"Missing required field '{key}'")
    if not isinstance(val, str):
        raise JSONTypeError(f"Field '{key}' must be a string")
    return val


def decode_unified_optimize_parse_result(
    raw: JSONObject,
) -> UnifiedOptimizeParseResult:
    """Decode a JSON object into UnifiedOptimizeParseResult.

    Args:
        raw: JSON object to decode.

    Returns:
        Validated UnifiedOptimizeParseResult.

    Raises:
        JSONTypeError: If any required field is missing or has wrong type.
    """
    timeout_val = raw.get("timeout_seconds")
    timeout_seconds: int | None = None
    if timeout_val is not None:
        if not isinstance(timeout_val, int):
            raise JSONTypeError("Field 'timeout_seconds' must be an integer or null")
        timeout_seconds = timeout_val

    return UnifiedOptimizeParseResult(
        backend=_require_backend_name(raw),
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


def require_unified_optimize_parse_result(
    raw: JSONValue,
) -> UnifiedOptimizeParseResult:
    """Validate a JSONValue as UnifiedOptimizeParseResult.

    Args:
        raw: JSON value to validate.

    Returns:
        Validated UnifiedOptimizeParseResult.

    Raises:
        JSONTypeError: If value is not a valid UnifiedOptimizeParseResult.
    """
    if not isinstance(raw, dict):
        raise JSONTypeError("Expected a JSON object for UnifiedOptimizeParseResult")
    return decode_unified_optimize_parse_result(raw)


# =============================================================================
# Optimization Result
# =============================================================================


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


def _require_float(raw: JSONObject, key: str) -> float:
    """Extract required float from JSON object.

    Args:
        raw: JSON object.
        key: Field name.

    Returns:
        Float value.

    Raises:
        JSONTypeError: If field is missing or not a number.
    """
    val = raw.get(key)
    if val is None:
        raise JSONTypeError(f"Missing required field '{key}'")
    if not isinstance(val, (int, float)):
        raise JSONTypeError(f"Field '{key}' must be a number")
    return float(val)


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


class PhaseProgressCallbackProtocol(Protocol):
    """Protocol for phase progress callback."""

    def __call__(self, info: PhaseProgressInfo) -> None:
        """Called when entering a new optimization phase.

        Args:
            info: Phase transition information.
        """
        ...


class LoadingProgressCallbackProtocol(Protocol):
    """Protocol for loading progress callback."""

    def __call__(self, info: LoadingProgressInfo) -> None:
        """Called with progress updates during dataset loading.

        Args:
            info: Loading progress information.
        """
        ...


class TrialProgressCallbackProtocol(Protocol):
    """Protocol for trial progress callback."""

    def __call__(self, info: TrialProgressInfo) -> None:
        """Called after each optimization trial with progress info.

        Args:
            info: Trial progress information.
        """
        ...


# =============================================================================
# Encode/Decode for Progress Types
# =============================================================================


def encode_phase_progress_info(info: PhaseProgressInfo) -> JSONObject:
    """Encode PhaseProgressInfo to JSON-serializable dict.

    Args:
        info: Phase progress information.

    Returns:
        JSON-serializable dict.
    """
    return {
        "phase": info["phase"],
        "backend": info["backend"],
        "dataset": info["dataset"],
        "n_samples": info["n_samples"],
        "n_features": info["n_features"],
    }


def decode_phase_progress_info(raw: JSONObject) -> PhaseProgressInfo:
    """Decode a JSON object into PhaseProgressInfo.

    Args:
        raw: JSON object to decode.

    Returns:
        Validated PhaseProgressInfo.

    Raises:
        JSONTypeError: If any required field is missing or has wrong type.
    """
    phase_val = _require_str(raw, "phase")
    if phase_val not in ("loading_data", "feature_engineering", "optimizing", "saving"):
        raise JSONTypeError(
            f"Field 'phase' must be one of: loading_data, feature_engineering, "
            f"optimizing, saving (got {phase_val})"
        )
    phase: OptimizePhase
    if phase_val == "loading_data":
        phase = "loading_data"
    elif phase_val == "feature_engineering":
        phase = "feature_engineering"
    elif phase_val == "optimizing":
        phase = "optimizing"
    else:
        phase = "saving"

    return PhaseProgressInfo(
        phase=phase,
        backend=_require_backend_name(raw),
        dataset=_require_str(raw, "dataset"),
        n_samples=_require_int(raw, "n_samples"),
        n_features=_require_int(raw, "n_features"),
    )


def encode_loading_progress_info(info: LoadingProgressInfo) -> JSONObject:
    """Encode LoadingProgressInfo to JSON-serializable dict.

    Args:
        info: Loading progress information.

    Returns:
        JSON-serializable dict.
    """
    return {
        "dataset": info["dataset"],
        "phase": info["phase"],
        "percent_complete": info["percent_complete"],
        "rows_processed": info["rows_processed"],
        "rows_total": info["rows_total"],
        "message": info["message"],
    }


def decode_loading_progress_info(raw: JSONObject) -> LoadingProgressInfo:
    """Decode a JSON object into LoadingProgressInfo.

    Args:
        raw: JSON object to decode.

    Returns:
        Validated LoadingProgressInfo.

    Raises:
        JSONTypeError: If any required field is missing or has wrong type.
    """
    phase_val = _require_str(raw, "phase")
    if phase_val not in ("reading", "parsing", "encoding"):
        raise JSONTypeError(
            f"Field 'phase' must be one of: reading, parsing, encoding (got {phase_val})"
        )
    load_phase: LoadPhase
    if phase_val == "reading":
        load_phase = "reading"
    elif phase_val == "parsing":
        load_phase = "parsing"
    else:
        load_phase = "encoding"

    return LoadingProgressInfo(
        dataset=_require_str(raw, "dataset"),
        phase=load_phase,
        percent_complete=_require_float(raw, "percent_complete"),
        rows_processed=_require_int(raw, "rows_processed"),
        rows_total=_require_int(raw, "rows_total"),
        message=_require_str(raw, "message"),
    )


def encode_trial_progress_info(info: TrialProgressInfo) -> JSONObject:
    """Encode TrialProgressInfo to JSON-serializable dict.

    Args:
        info: Trial progress information.

    Returns:
        JSON-serializable dict.
    """
    return {
        "backend": info["backend"],
        "trial_number": info["trial_number"],
        "n_trials_total": info["n_trials_total"],
        "current_value": info["current_value"],
        "best_value": info["best_value"],
        "best_trial": info["best_trial"],
        "is_best": info["is_best"],
    }


def decode_trial_progress_info(raw: JSONObject) -> TrialProgressInfo:
    """Decode a JSON object into TrialProgressInfo.

    Args:
        raw: JSON object to decode.

    Returns:
        Validated TrialProgressInfo.

    Raises:
        JSONTypeError: If any required field is missing or has wrong type.
    """
    return TrialProgressInfo(
        backend=_require_backend_name(raw),
        trial_number=_require_int(raw, "trial_number"),
        n_trials_total=_require_int(raw, "n_trials_total"),
        current_value=_require_float(raw, "current_value"),
        best_value=_require_float(raw, "best_value"),
        best_trial=_require_int(raw, "best_trial"),
        is_best=_require_bool(raw, "is_best"),
    )


__all__ = [
    "LoadingProgressCallbackProtocol",
    "LoadingProgressInfo",
    "OptimizePhase",
    "PhaseProgressCallbackProtocol",
    "PhaseProgressInfo",
    "TrialProgressCallbackProtocol",
    "TrialProgressInfo",
    "UnifiedOptimizationResult",
    "UnifiedOptimizeParseResult",
    "decode_loading_progress_info",
    "decode_phase_progress_info",
    "decode_trial_progress_info",
    "decode_unified_optimization_result",
    "decode_unified_optimize_parse_result",
    "encode_loading_progress_info",
    "encode_phase_progress_info",
    "encode_trial_progress_info",
    "encode_unified_optimization_result",
    "encode_unified_optimize_parse_result",
    "require_unified_optimization_result",
    "require_unified_optimize_parse_result",
]
