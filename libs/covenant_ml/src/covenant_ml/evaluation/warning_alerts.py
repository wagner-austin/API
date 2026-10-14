"""Outcome-independent cooldown policies and unique incident warning accounting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from platform_core.json_utils import JSONObject, narrow_json_to_int, require_int, require_list
from typing_extensions import ReadOnly, TypedDict

from .warning_history import require_warning_history
from .warning_panel import WarningRow, warning_label
from .warning_records import require_warning_time
from .warning_review import require_warning_batch, select_warning_reviews


class WarningLongitudinal(TypedDict):
    """Alert-level precision counts, unique-event recall counts and at-risk exposure."""

    opportunities: ReadOnly[int]
    alerts: ReadOnly[int]
    true_alerts: ReadOnly[int]
    incidents: ReadOnly[int]
    detected: ReadOnly[int]
    exposure_ms: ReadOnly[int]
    lead_times_ms: ReadOnly[tuple[int, ...]]


def decode_warning_longitudinal(document: JSONObject) -> WarningLongitudinal:
    """Require internally coherent alert counts, incident counts and exposure.

    Args:
        document: Archived longitudinal summary.

    Returns:
        WarningLongitudinal: Immutable summary; one lead time per detected incident.

    Raises:
        TypeError: Required fields or lead times are incorrectly typed.
        ValueError: WARN-022 for contradictory counts or nonpositive exposure/leads.
    """
    opportunities = require_int(document, "opportunities")
    alerts = require_int(document, "alerts")
    true_alerts = require_int(document, "true_alerts")
    incidents = require_int(document, "incidents")
    detected = require_int(document, "detected")
    exposure = require_int(document, "exposure_ms")
    leads = tuple(narrow_json_to_int(v) for v in require_list(document, "lead_times_ms"))
    if (
        not 0 <= detected <= incidents <= opportunities
        or not 0 <= true_alerts <= alerts <= opportunities
    ):
        raise ValueError("[WARN-022] longitudinal counts are inconsistent")
    if detected > true_alerts or (detected == 0 and true_alerts != 0):
        raise ValueError("[WARN-022] true alerts must identify detected incidents")
    if opportunities <= 0 or exposure <= 0 or len(leads) != detected or any(v <= 0 for v in leads):
        raise ValueError("[WARN-022] exposure and one positive lead per detection are required")
    return WarningLongitudinal(
        opportunities=opportunities,
        alerts=alerts,
        true_alerts=true_alerts,
        incidents=incidents,
        detected=detected,
        exposure_ms=exposure,
        lead_times_ms=leads,
    )


def encode_warning_longitudinal(result: WarningLongitudinal) -> JSONObject:
    """Encode longitudinal counts with no duplicate derived rates.

    Args:
        result: Validated summary.

    Returns:
        JSONObject: Independent JSON representation.
    """
    return {
        "opportunities": result["opportunities"],
        "alerts": result["alerts"],
        "true_alerts": result["true_alerts"],
        "incidents": result["incidents"],
        "detected": result["detected"],
        "exposure_ms": result["exposure_ms"],
        "lead_times_ms": list(result["lead_times_ms"]),
    }


def select_warning_alerts(
    scores: Mapping[int, Mapping[str, float]], budget: int, cooldown_ms: int
) -> tuple[tuple[int, str], ...]:
    """Rank each batch after suppressing recently alerted borrowers, without outcomes.

    Args:
        scores: Available scores by decision timestamp and borrower.
        budget: Fixed maximum reviews per decision batch.
        cooldown_ms: Nonnegative suppression duration from the previous alert;
            a borrower becomes eligible again exactly at that duration.

    Returns:
        tuple[tuple[int, str], ...]: Chronological alerts in within-batch rank order.

    Raises:
        ValueError: WARN-023 for invalid policy or empty history. Score validation
            propagates WARN-010/011; suppressed scores are also validated.
    """
    if not scores or isinstance(cooldown_ms, bool) or cooldown_ms < 0:
        raise ValueError("[WARN-023] require a nonempty history and nonnegative integer cooldown")
    if isinstance(budget, bool) or budget <= 0:
        raise ValueError("[WARN-023] review budget must be a positive integer")
    previous: dict[str, int] = {}
    alerts: list[tuple[int, str]] = []
    for cutoff in sorted(scores):
        require_warning_time({"cutoff_ms": cutoff}, "cutoff_ms")
        ranked = select_warning_reviews(scores[cutoff], max(1, len(scores[cutoff])))
        eligible = [
            name
            for name in ranked
            if name not in previous or cutoff - previous[name] >= cooldown_ms
        ]
        for name in eligible[:budget]:
            alerts.append((cutoff, name))
            previous[name] = cutoff
    return tuple(alerts)


def _exposure_and_events(rows: Sequence[WarningRow]) -> tuple[int, dict[str, int]]:
    """Union overlapping at-risk windows and require coherent first-breach histories.

    Args:
        rows: Validated mature opportunities with one fixed horizon.

    Returns:
        tuple[int, dict[str, int]]: At-risk borrower milliseconds and incident times.

    Raises:
        ValueError: WARN-024 for conflicting or contradictory breach evidence.
    """
    incidents = require_warning_history(rows)
    histories: dict[str, list[WarningRow]] = {}
    for row in rows:
        borrower = row["disclosure"]["borrower"]
        histories.setdefault(borrower, []).append(row)
    exposure = 0
    for borrower, history in histories.items():
        breach = incidents.get(borrower)
        windows: list[tuple[int, int]] = []
        for row in history:
            end = (
                min(row["horizon_end_ms"], breach) if breach is not None else row["horizon_end_ms"]
            )
            windows.append((row["cutoff_ms"], end))
        covered_until = -1
        for start, end in sorted(windows):
            exposure += max(0, end - max(start, covered_until))
            covered_until = max(covered_until, end)
    return exposure, incidents


def evaluate_warning_alerts(
    rows: Sequence[WarningRow],
    probabilities: Sequence[float],
    budget: int,
    cooldown_ms: int,
    as_of_ms: int,
) -> WarningLongitudinal:
    """Score repeated alerts without counting one incident as multiple successes.

    Args:
        rows: Same frozen opportunity population supplied to every model.
        probabilities: Scores aligned to rows, recorded before outcomes.
        budget: Common maximum reviews per batch; cooldown can leave slots unused.
        cooldown_ms: Frozen model-independent suppression duration.
        as_of_ms: Analysis cutoff for mature labels.

    Returns:
        WarningLongitudinal: Alert precision counts, unique incident recall counts,
        earliest qualifying alert lead times, and unioned at-risk exposure.
        Incident denominator includes only breaches inside a supplied opportunity.

    Raises:
        ValueError: WARN-025 for empty/misaligned panels or varying horizons;
            batch, selection and history consistency errors propagate.
    """
    if not rows or len(rows) != len(probabilities):
        raise ValueError("[WARN-025] scores must align with a nonempty opportunity panel")
    if len({row["horizon_end_ms"] - row["cutoff_ms"] for row in rows}) != 1:
        raise ValueError("[WARN-025] longitudinal evaluation requires a fixed horizon")
    batches: dict[int, list[WarningRow]] = {}
    scores: dict[int, dict[str, float]] = {}
    evidence: dict[tuple[int, str], WarningRow] = {}
    for row, score in zip(rows, probabilities, strict=True):
        cutoff, borrower = row["cutoff_ms"], row["disclosure"]["borrower"]
        batches.setdefault(cutoff, []).append(row)
        scores.setdefault(cutoff, {})[borrower] = score
        evidence[cutoff, borrower] = row
    for batch in batches.values():
        require_warning_batch(batch, as_of_ms)
    exposure, _ = _exposure_and_events(rows)
    incidents = {row["disclosure"]["borrower"] for row in rows if warning_label(row) == 1}
    alerts = select_warning_alerts(scores, budget, cooldown_ms)
    detected: dict[str, int] = {}
    true_alerts = 0
    for key in alerts:
        row = evidence[key]
        breach = row["followup"]["first_breach_ms"]
        if warning_label(row) == 1 and breach is not None:
            true_alerts += 1
            if key[1] not in detected:
                detected[key[1]] = breach - key[0]
    return WarningLongitudinal(
        opportunities=len(rows),
        alerts=len(alerts),
        true_alerts=true_alerts,
        incidents=len(incidents),
        detected=len(detected),
        exposure_ms=exposure,
        lead_times_ms=tuple(detected.values()),
    )


__all__ = [
    "WarningLongitudinal",
    "decode_warning_longitudinal",
    "encode_warning_longitudinal",
    "evaluate_warning_alerts",
    "select_warning_alerts",
]
