"""Fixed-budget warning comparisons on a shared, fully observed borrower batch."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from platform_core.json_utils import (
    JSONObject,
    narrow_json_to_int,
    narrow_json_to_str,
    require_int,
    require_list,
)
from typing_extensions import ReadOnly, TypedDict

from .warning_panel import WarningRow, warning_label


class WarningReview(TypedDict):
    """Immutable batch counts and ranked alerts; rates are derived, never duplicated."""

    eligible: ReadOnly[int]
    budget: ReadOnly[int]
    breaches: ReadOnly[int]
    selected: ReadOnly[tuple[str, ...]]
    lead_times_ms: ReadOnly[tuple[int, ...]]


def select_warning_reviews(scores: Mapping[str, float], budget: int) -> tuple[str, ...]:
    """Select up to budget borrowers without receiving outcomes.

    Args:
        scores: Finite risk scores keyed by unique borrower; larger means riskier.
        budget: Positive integer number of review slots, fixed before evaluation.

    Returns:
        tuple[str, ...]: Descending score order, ties broken by borrower identifier.

    Raises:
        ValueError: WARN-010 for invalid budget; WARN-011 for empty or invalid scores.
    """
    if isinstance(budget, bool) or budget <= 0:
        raise ValueError("[WARN-010] review budget must be a positive integer")
    if not scores:
        raise ValueError("[WARN-011] a review batch must contain borrowers")
    for borrower, score in scores.items():
        if (
            not borrower
            or borrower != borrower.strip()
            or isinstance(score, bool)
            or not math.isfinite(score)
        ):
            raise ValueError("[WARN-011] borrower identifiers and finite risk scores are required")
    ordered = sorted((-score, borrower) for borrower, score in scores.items())
    return tuple(borrower for _, borrower in ordered[:budget])


def decode_warning_review(document: JSONObject) -> WarningReview:
    """Require internally consistent review counts and positive detection lead times.

    Args:
        document: Serialized batch result.

    Returns:
        WarningReview: Validated immutable result; lead times follow detected alert order.

    Raises:
        TypeError: Required fields or list elements have invalid types.
        ValueError: WARN-012 for impossible counts, duplicate alerts or invalid lead times.
    """
    eligible = require_int(document, "eligible")
    budget = require_int(document, "budget")
    breaches = require_int(document, "breaches")
    selected = tuple(narrow_json_to_str(v) for v in require_list(document, "selected"))
    leads = tuple(narrow_json_to_int(v) for v in require_list(document, "lead_times_ms"))
    if eligible <= 0 or budget <= 0 or not 0 <= breaches <= eligible:
        raise ValueError("[WARN-012] invalid eligible, budget or breach count")
    if len(selected) != min(eligible, budget) or len(set(selected)) != len(selected):
        raise ValueError("[WARN-012] alerts must be unique and fill the available review slots")
    if any(not name or name != name.strip() for name in selected):
        raise ValueError("[WARN-012] selected borrower identifiers must be nonblank and trimmed")
    if (
        not max(0, breaches + len(selected) - eligible)
        <= len(leads)
        <= min(breaches, len(selected))
    ):
        raise ValueError("[WARN-012] detected breaches contradict batch and alert counts")
    if any(lead <= 0 or lead > 2**53 - 1 for lead in leads):
        raise ValueError("[WARN-012] lead times must be positive exact milliseconds")
    return WarningReview(
        eligible=eligible, budget=budget, breaches=breaches, selected=selected, lead_times_ms=leads
    )


def encode_warning_review(result: WarningReview) -> JSONObject:
    """Encode a complete batch result for replay.

    Args:
        result: Validated review counts and alerts.

    Returns:
        JSONObject: Independent JSON representation.
    """
    return {
        "eligible": result["eligible"],
        "budget": result["budget"],
        "breaches": result["breaches"],
        "selected": list(result["selected"]),
        "lead_times_ms": list(result["lead_times_ms"]),
    }


def warning_review_rates(result: WarningReview) -> tuple[float, float | None, int]:
    """Derive precision, recall and false-alert count without inventing undefined recall.

    Args:
        result: Validated review result.

    Returns:
        tuple[float, float | None, int]: Precision among actual alerts, recall among
        breaches (None if none occurred), and false alerts. This is one batch,
        not event-level recall across repeated reviews or a borrower-time rate.
    """
    detected = len(result["lead_times_ms"])
    alerts = len(result["selected"])
    return (
        detected / alerts,
        detected / result["breaches"] if result["breaches"] else None,
        alerts - detected,
    )


def require_warning_batch(rows: Sequence[WarningRow], as_of_ms: int) -> dict[str, WarningRow]:
    """Require a complete common decision batch for batch or longitudinal scoring.

    Args:
        rows: Decoder-validated at-risk evidence, one row per borrower at one cutoff
            and common horizon. Eligibility must have been frozen without outcomes.
        as_of_ms: Analysis time by which horizons and label evidence must be known.

    Returns:
        dict[str, WarningRow]: Unique borrower-indexed evidence for one decision window.

    Raises:
        ValueError: WARN-013 for mismatched batches; WARN-014 for immature or
            censored evidence.
    """
    if not rows:
        raise ValueError("[WARN-013] a nonempty borrower batch is required")
    borrowers = {row["disclosure"]["borrower"]: row for row in rows}
    windows = {(row["cutoff_ms"], row["horizon_end_ms"]) for row in rows}
    if len(borrowers) != len(rows) or len(windows) != 1:
        raise ValueError("[WARN-013] require unique borrowers and one common prediction window")
    for row in borrowers.values():
        label = warning_label(row)
        if (
            label is None
            or row["horizon_end_ms"] > as_of_ms
            or row["followup"]["available_ms"] > as_of_ms
        ):
            raise ValueError("[WARN-014] every horizon and outcome must be mature and known")
    return borrowers


def compare_warning_reviews(
    rows: Sequence[WarningRow],
    model_scores: Mapping[str, Mapping[str, float]],
    budget: int,
    as_of_ms: int,
) -> dict[str, WarningReview]:
    """Score models on exactly the same mature batch and fixed review budget.

    Args:
        rows: Decoder-validated at-risk evidence with outcome-independent eligibility.
        model_scores: Named models scoring exactly the same borrowers at the cutoff.
        budget: Positive fixed integer number of review slots.
        as_of_ms: Analysis cutoff for all labels and horizons.

    Returns:
        dict[str, WarningReview]: Counts and lead times for each model.

    Raises:
        ValueError: WARN-013 for mismatched models; batch WARN-013/014 and selection
            WARN-010/011 errors propagate.
    """
    borrowers = require_warning_batch(rows, as_of_ms)
    if len(model_scores) < 2:
        raise ValueError("[WARN-013] at least two models are required")
    labels = {borrower: warning_label(row) for borrower, row in borrowers.items()}
    results: dict[str, WarningReview] = {}
    for model, scores in model_scores.items():
        if not model or model != model.strip() or set(scores) != set(borrowers):
            raise ValueError("[WARN-013] named models must score exactly the same borrowers")
        selected = select_warning_reviews(scores, budget)
        leads: list[int] = []
        for borrower in selected:
            breach = borrowers[borrower]["followup"]["first_breach_ms"]
            if labels[borrower] == 1 and breach is not None:
                leads.append(breach - borrowers[borrower]["cutoff_ms"])
        results[model] = WarningReview(
            eligible=len(rows),
            budget=budget,
            breaches=sum(label == 1 for label in labels.values()),
            selected=selected,
            lead_times_ms=tuple(leads),
        )
    return results


__all__ = [
    "WarningReview",
    "compare_warning_reviews",
    "decode_warning_review",
    "encode_warning_review",
    "require_warning_batch",
    "select_warning_reviews",
    "warning_review_rates",
]
