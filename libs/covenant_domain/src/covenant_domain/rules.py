from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from .formula_parser import evaluate_formula
from .models import Covenant, CovenantResult, Measurement


def classify_status(
    threshold_value_scaled: int,
    threshold_direction: Literal["<=", ">="],
    calculated_value_scaled: int,
    tolerance_ratio_scaled: int,
) -> Literal["OK", "NEAR_BREACH", "BREACH"]:
    """
    Classify covenant status based on calculated value vs threshold.

    tolerance_ratio_scaled: The tolerance band width (e.g., 100_000 for 10%).
    All values are scaled integers (multiply by 1_000_000).

    For "<=": BREACH if calculated > threshold
              NEAR_BREACH if calculated > threshold * (1 - tolerance)
              OK otherwise

    For ">=": BREACH if calculated < threshold
              NEAR_BREACH if calculated < threshold * (1 + tolerance)
              OK otherwise
    """
    # Calculate tolerance band
    tolerance_amount = (threshold_value_scaled * tolerance_ratio_scaled) // 1_000_000

    if threshold_direction == "<=":
        if calculated_value_scaled > threshold_value_scaled:
            return "BREACH"
        if calculated_value_scaled > threshold_value_scaled - tolerance_amount:
            return "NEAR_BREACH"
        return "OK"

    # threshold_direction == ">="
    if calculated_value_scaled < threshold_value_scaled:
        return "BREACH"
    if calculated_value_scaled < threshold_value_scaled + tolerance_amount:
        return "NEAR_BREACH"
    return "OK"


def _build_metrics_for_period(
    measurements: Sequence[Measurement],
    period_start_iso: str,
    period_end_iso: str,
) -> dict[str, int]:
    """
    Build metric name -> scaled value mapping for a specific period.

    Raises ValueError if duplicate metric names for same period.
    """
    metrics: dict[str, int] = {}

    for m in measurements:
        if m["period_start_iso"] == period_start_iso and m["period_end_iso"] == period_end_iso:
            name = m["metric_name"]
            if name in metrics:
                raise ValueError(f"Duplicate metric {name} for period {period_start_iso}")
            metrics[name] = m["metric_value_scaled"]

    return metrics


def evaluate_covenant_for_period(
    covenant: Covenant,
    period_start_iso: str,
    period_end_iso: str,
    measurements: Sequence[Measurement],
    tolerance_ratio_scaled: int,
) -> CovenantResult:
    """
    Evaluate a covenant for a specific period.

    Raises:
        KeyError: Required metric missing from measurements
        FormulaParseError: Invalid formula
        FormulaEvalError: Division by zero
        ValueError: Duplicate metrics
    """
    metrics = _build_metrics_for_period(measurements, period_start_iso, period_end_iso)

    calculated_value_scaled = evaluate_formula(covenant["formula"], metrics)

    status = classify_status(
        threshold_value_scaled=covenant["threshold_value_scaled"],
        threshold_direction=covenant["threshold_direction"],
        calculated_value_scaled=calculated_value_scaled,
        tolerance_ratio_scaled=tolerance_ratio_scaled,
    )

    return CovenantResult(
        covenant_id=covenant["id"],
        period_start_iso=period_start_iso,
        period_end_iso=period_end_iso,
        calculated_value_scaled=calculated_value_scaled,
        status=status,
    )


def evaluate_all_covenants_for_period(
    covenants: Sequence[Covenant],
    period_start_iso: str,
    period_end_iso: str,
    measurements: Sequence[Measurement],
    tolerance_ratio_scaled: int,
) -> list[CovenantResult]:
    """
    Evaluate all covenants for a period. Returns results in same order as covenants.

    Raises on first failure - no partial results.
    """
    results: list[CovenantResult] = []

    for covenant in covenants:
        result = evaluate_covenant_for_period(
            covenant=covenant,
            period_start_iso=period_start_iso,
            period_end_iso=period_end_iso,
            measurements=measurements,
            tolerance_ratio_scaled=tolerance_ratio_scaled,
        )
        results.append(result)

    return results


__all__ = [
    "classify_status",
    "evaluate_all_covenants_for_period",
    "evaluate_covenant_for_period",
]
