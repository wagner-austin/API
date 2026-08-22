"""Unified types for hyperparameter optimization jobs.

Backend-agnostic TypedDicts for progress reporting, config parsing,
and result serialization. Used by the unified optimize_job.py.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

from covenant_ml.datasets.types import LoadPhase
from covenant_ml.features import FeaturePreset
from covenant_ml.types import BackendName
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
)

from covenant_radar_api.worker.optimize_field_decoders import (
    _require_backend_name,
    _require_bool,
    _require_device,
    _require_feature_preset,
    _require_float,
    _require_int,
    _require_nn_optimizer,
    _require_precision,
    _require_str,
)
from covenant_radar_api.worker.optimize_result_types import (
    UnifiedOptimizationResult,
    decode_unified_optimization_result,
    encode_unified_optimization_result,
    require_unified_optimization_result,
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
