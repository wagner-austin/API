"""Tuning type definitions for ClearGBM.

Provides TimingResult and TuningReport TypedDicts with their encode/decode
functions.

This module is private (underscore prefix) — not for external use.
"""

from __future__ import annotations

from typing import TypedDict

from cleargbm._types_config import (
    GradientBoostingConfig,
    decode_gradient_boosting_config,
    encode_gradient_boosting_config,
)
from cleargbm._types_json import (
    JSONDict,
    JSONTypeError,
    _as_json_dict,
    _require_float,
    _require_int,
    require_n_jobs,
    require_non_negative_float,
    require_positive_float,
    require_positive_int,
)

# =============================================================================
# Tuning Types
# =============================================================================


class TimingResult(TypedDict):
    """Timing result for a single configuration.

    Args:
        n_jobs: Number of parallel workers used.
        max_bins: Number of histogram bins used.
        max_depth: Maximum tree depth used.
        learning_rate: Learning rate used.
        elapsed_seconds: Time taken in seconds.
        trees_per_second: Training throughput.
    """

    n_jobs: int
    max_bins: int
    max_depth: int
    learning_rate: float
    elapsed_seconds: float
    trees_per_second: float


def encode_timing_result(result: TimingResult) -> JSONDict:
    """Encode TimingResult to JSON-serializable dict.

    Args:
        result: Result to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "n_jobs": result["n_jobs"],
        "max_bins": result["max_bins"],
        "max_depth": result["max_depth"],
        "learning_rate": result["learning_rate"],
        "elapsed_seconds": result["elapsed_seconds"],
        "trees_per_second": result["trees_per_second"],
    }


def decode_timing_result(raw: JSONDict) -> TimingResult:
    """Decode raw dict to TimingResult.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated TimingResult.

    Raises:
        KeyError: If required key is missing.
        JSONTypeError: If value has wrong type.
        ValueError: If value fails validation.
    """
    n_jobs = require_n_jobs(_require_int(raw, "n_jobs"), "n_jobs")
    max_bins = require_positive_int(_require_int(raw, "max_bins"), "max_bins")
    max_depth = require_positive_int(_require_int(raw, "max_depth"), "max_depth")
    learning_rate = require_positive_float(_require_float(raw, "learning_rate"), "learning_rate")
    elapsed_seconds = require_non_negative_float(
        _require_float(raw, "elapsed_seconds"), "elapsed_seconds"
    )
    trees_per_second = require_non_negative_float(
        _require_float(raw, "trees_per_second"), "trees_per_second"
    )

    return TimingResult(
        n_jobs=n_jobs,
        max_bins=max_bins,
        max_depth=max_depth,
        learning_rate=learning_rate,
        elapsed_seconds=elapsed_seconds,
        trees_per_second=trees_per_second,
    )


class TuningReport(TypedDict):
    """Complete autotuning report with recommendations.

    Args:
        best_config: Recommended configuration based on tuning.
        timing_results: All timing results from the grid search.
        sample_size: Number of samples used for tuning.
        n_features: Number of features in the dataset.
        recommended_n_jobs: Best n_jobs value found.
        recommended_max_bins: Best max_bins value found.
        parallel_speedup: Speedup ratio vs sequential (1.0 = no speedup).
        total_tune_time_seconds: Total time spent tuning.
    """

    best_config: GradientBoostingConfig
    timing_results: tuple[TimingResult, ...]
    sample_size: int
    n_features: int
    recommended_n_jobs: int
    recommended_max_bins: int
    parallel_speedup: float
    total_tune_time_seconds: float


def encode_tuning_report(report: TuningReport) -> JSONDict:
    """Encode TuningReport to JSON-serializable dict.

    Args:
        report: Report to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "best_config": encode_gradient_boosting_config(report["best_config"]),
        "timing_results": [encode_timing_result(r) for r in report["timing_results"]],
        "sample_size": report["sample_size"],
        "n_features": report["n_features"],
        "recommended_n_jobs": report["recommended_n_jobs"],
        "recommended_max_bins": report["recommended_max_bins"],
        "parallel_speedup": report["parallel_speedup"],
        "total_tune_time_seconds": report["total_tune_time_seconds"],
    }


def decode_tuning_report(raw: JSONDict) -> TuningReport:
    """Decode raw dict to TuningReport.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated TuningReport.

    Raises:
        KeyError: If required key is missing.
        JSONTypeError: If value has wrong type.
        ValueError: If value fails validation.
    """
    config_dict = _as_json_dict(raw["best_config"], "best_config")
    best_config = decode_gradient_boosting_config(config_dict)

    timing_results_raw = raw["timing_results"]
    if not isinstance(timing_results_raw, list):
        raise JSONTypeError(f"timing_results must be list, got {type(timing_results_raw).__name__}")
    timing_results: list[TimingResult] = []
    for i, tr_raw in enumerate(timing_results_raw):
        tr_dict = _as_json_dict(tr_raw, f"timing_results[{i}]")
        timing_results.append(decode_timing_result(tr_dict))

    sample_size = require_positive_int(_require_int(raw, "sample_size"), "sample_size")
    n_features = require_positive_int(_require_int(raw, "n_features"), "n_features")
    recommended_n_jobs = require_n_jobs(
        _require_int(raw, "recommended_n_jobs"), "recommended_n_jobs"
    )
    recommended_max_bins = require_positive_int(
        _require_int(raw, "recommended_max_bins"), "recommended_max_bins"
    )
    parallel_speedup = require_non_negative_float(
        _require_float(raw, "parallel_speedup"), "parallel_speedup"
    )
    total_tune_time_seconds = require_non_negative_float(
        _require_float(raw, "total_tune_time_seconds"),
        "total_tune_time_seconds",
    )

    return TuningReport(
        best_config=best_config,
        timing_results=tuple(timing_results),
        sample_size=sample_size,
        n_features=n_features,
        recommended_n_jobs=recommended_n_jobs,
        recommended_max_bins=recommended_max_bins,
        parallel_speedup=parallel_speedup,
        total_tune_time_seconds=total_tune_time_seconds,
    )


__all__ = [
    "TimingResult",
    "TuningReport",
    "decode_timing_result",
    "decode_tuning_report",
    "encode_timing_result",
    "encode_tuning_report",
]
