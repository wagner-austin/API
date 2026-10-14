"""Integrate chronological selection, calibration diagnostics and repeated alerts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from platform_core.json_utils import JSONObject, require_dict
from typing_extensions import ReadOnly, TypedDict

from .warning_alerts import (
    WarningLongitudinal,
    decode_warning_longitudinal,
    encode_warning_longitudinal,
    evaluate_warning_alerts,
)
from .warning_calibration import (
    WarningCalibration,
    decode_warning_calibration,
    encode_warning_calibration,
    summarize_warning_calibration,
)
from .warning_folds import WarningFold
from .warning_learning import WarningLearnerFactory
from .warning_study import (
    WarningForecast,
    decode_warning_forecast,
    encode_warning_forecast,
    run_warning_study,
)


class WarningEvaluation(TypedDict):
    """One frozen arm evaluated against the same final opportunities and policy."""

    forecast: ReadOnly[WarningForecast]
    raw_reliability: ReadOnly[WarningCalibration]
    calibrated_reliability: ReadOnly[WarningCalibration]
    raw_alerts: ReadOnly[WarningLongitudinal]
    calibrated_alerts: ReadOnly[WarningLongitudinal]


def decode_warning_evaluation(document: JSONObject) -> WarningEvaluation:
    """Decode nested predictions, reliability statistics and alert outcomes.

    Args:
        document: Archived arm report.

    Returns:
        WarningEvaluation: Validated immutable component records.

    Raises:
        TypeError: Nested required fields have invalid types.
        ValueError: Component validation errors or WARN-029 for mismatched row counts.
    """
    forecast = decode_warning_forecast(require_dict(document, "forecast"))
    raw = decode_warning_calibration(require_dict(document, "raw_reliability"))
    calibrated = decode_warning_calibration(require_dict(document, "calibrated_reliability"))
    raw_alerts = decode_warning_longitudinal(require_dict(document, "raw_alerts"))
    calibrated_alerts = decode_warning_longitudinal(require_dict(document, "calibrated_alerts"))
    counts = {
        len(forecast["raw"]),
        sum(raw["counts"]),
        sum(calibrated["counts"]),
        raw_alerts["opportunities"],
        calibrated_alerts["opportunities"],
    }
    if len(counts) != 1:
        raise ValueError(
            "[WARN-029] evaluation components must describe the same opportunity count"
        )
    if (
        raw_alerts["incidents"] != calibrated_alerts["incidents"]
        or raw_alerts["exposure_ms"] != calibrated_alerts["exposure_ms"]
        or sum(raw["positives"]) != sum(calibrated["positives"])
        or len(raw["counts"]) != len(calibrated["counts"])
    ):
        raise ValueError("[WARN-029] calibration must not change cohort outcomes or exposure")
    return WarningEvaluation(
        forecast=forecast,
        raw_reliability=raw,
        calibrated_reliability=calibrated,
        raw_alerts=raw_alerts,
        calibrated_alerts=calibrated_alerts,
    )


def encode_warning_evaluation(result: WarningEvaluation) -> JSONObject:
    """Encode the complete raw and calibrated comparison for one arm.

    Args:
        result: Validated evaluation record.

    Returns:
        JSONObject: Portable result without references to live model objects.
    """
    return {
        "forecast": encode_warning_forecast(result["forecast"]),
        "raw_reliability": encode_warning_calibration(result["raw_reliability"]),
        "calibrated_reliability": encode_warning_calibration(result["calibrated_reliability"]),
        "raw_alerts": encode_warning_longitudinal(result["raw_alerts"]),
        "calibrated_alerts": encode_warning_longitudinal(result["calibrated_alerts"]),
    }


def evaluate_warning_experiment(
    candidates: Mapping[str, WarningLearnerFactory],
    folds: Sequence[WarningFold],
    final: WarningFold,
    as_of_ms: int,
    budget: int,
    cooldown_ms: int,
    bins: int,
) -> tuple[str, dict[str, float], dict[str, WarningEvaluation]]:
    """Run the frozen comparison and report every arm without selecting on holdout scores.

    Args:
        candidates: Named predeclared learner factories including simple baselines.
        folds: Earlier borrower-disjoint fit/calibrate/validate development folds.
        final: Final fit/calibrate/holdout evidence; preserve this alongside results.
        as_of_ms: Final analysis cutoff for label availability.
        budget: Common fixed per-batch review capacity.
        cooldown_ms: Common frozen suppression duration.
        bins: Common fixed reliability bin count.

    Returns:
        tuple[str, dict[str, float], dict[str, WarningEvaluation]]: Development-selected
        arm, pooled development Brier scores and each arm's later evaluation.

    Raises:
        ValueError: Study, calibration, outcome or policy validation failures propagate.
            Nothing retries, drops a row or selects a replacement policy.
    """
    selected, scores, forecasts = run_warning_study(candidates, folds, final, as_of_ms)
    rows = final["validation"]
    results: dict[str, WarningEvaluation] = {}
    for name, forecast in forecasts.items():
        results[name] = WarningEvaluation(
            forecast=forecast,
            raw_reliability=summarize_warning_calibration(rows, forecast["raw"], bins),
            calibrated_reliability=summarize_warning_calibration(
                rows, forecast["calibrated"], bins
            ),
            raw_alerts=evaluate_warning_alerts(
                rows, forecast["raw"], budget, cooldown_ms, as_of_ms
            ),
            calibrated_alerts=evaluate_warning_alerts(
                rows, forecast["calibrated"], budget, cooldown_ms, as_of_ms
            ),
        )
    return selected, scores, results


__all__ = [
    "WarningEvaluation",
    "decode_warning_evaluation",
    "encode_warning_evaluation",
    "evaluate_warning_experiment",
]
