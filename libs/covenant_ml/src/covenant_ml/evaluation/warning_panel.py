"""As-of feature selection and auditable, future-only incident warning labels."""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.json_utils import JSONObject, require_dict
from typing_extensions import ReadOnly, TypedDict

from .warning_records import (
    WarningDisclosure,
    WarningFollowup,
    decode_warning_disclosure,
    decode_warning_followup,
    encode_warning_disclosure,
    encode_warning_followup,
    require_warning_time,
)


class WarningRow(TypedDict):
    """One fixed prediction opportunity with separately retained outcome evidence."""

    cutoff_ms: ReadOnly[int]
    horizon_end_ms: ReadOnly[int]
    disclosure: ReadOnly[WarningDisclosure]
    followup: ReadOnly[WarningFollowup]


def select_warning_disclosure(
    disclosures: Sequence[WarningDisclosure],
    borrower: str,
    cutoff_ms: int,
) -> WarningDisclosure:
    """Select the latest available period and its latest available revision.

    Args:
        disclosures: Dated disclosures validated by the evidence codec.
        borrower: Exact borrower identifier.
        cutoff_ms: Prediction timestamp; availability at this instant is allowed.

    Returns:
        WarningDisclosure: Selected vintage, independent of later disclosures.

    Raises:
        ValueError: WARN-006 for no available disclosure or ambiguous vintages.
    """
    eligible = [
        row
        for row in disclosures
        if row["borrower"] == borrower and row["available_ms"] <= cutoff_ms
    ]
    if not eligible:
        raise ValueError("[WARN-006] no disclosure available for borrower at prediction cutoff")
    key = max((row["period_end_ms"], row["available_ms"]) for row in eligible)
    selected = [row for row in eligible if (row["period_end_ms"], row["available_ms"]) == key]
    if len(selected) != 1:
        raise ValueError(
            "[WARN-006] multiple disclosures share the selected period and availability"
        )
    return selected[0]


def decode_warning_row(document: JSONObject) -> WarningRow:
    """Validate an at-risk prediction opportunity and its separate outcome window.

    Args:
        document: JSON row with nested disclosure and follow-up evidence.

    Returns:
        WarningRow: Immutable row; incomplete follow-up remains explicitly censored.

    Raises:
        TypeError: Fields are missing or incorrectly typed.
        ValueError: WARN-007 for leakage, mismatched identity, missing at-risk coverage
            or an already breached borrower; evidence codecs preserve their own codes.
    """
    cutoff = require_warning_time(document, "cutoff_ms")
    end = require_warning_time(document, "horizon_end_ms")
    disclosure = decode_warning_disclosure(require_dict(document, "disclosure"))
    followup = decode_warning_followup(require_dict(document, "followup"))
    if not disclosure["available_ms"] <= cutoff < end:
        raise ValueError("[WARN-007] require available disclosure <= cutoff < horizon end")
    if disclosure["borrower"] != followup["borrower"]:
        raise ValueError("[WARN-007] disclosure and follow-up must identify the same borrower")
    if not followup["coverage_start_ms"] <= cutoff <= followup["observed_through_ms"]:
        raise ValueError("[WARN-007] follow-up must establish at-risk coverage at the cutoff")
    breach = followup["first_breach_ms"]
    if breach is not None and breach <= cutoff:
        raise ValueError("[WARN-007] incident warning excludes breach at or before cutoff")
    return WarningRow(
        cutoff_ms=cutoff, horizon_end_ms=end, disclosure=disclosure, followup=followup
    )


def encode_warning_row(row: WarningRow) -> JSONObject:
    """Encode predictors and outcomes in distinct nested records.

    Args:
        row: Validated prediction opportunity.

    Returns:
        JSONObject: Complete row for offline replay.
    """
    return {
        "cutoff_ms": row["cutoff_ms"],
        "horizon_end_ms": row["horizon_end_ms"],
        "disclosure": encode_warning_disclosure(row["disclosure"]),
        "followup": encode_warning_followup(row["followup"]),
    }


def warning_label(row: WarningRow) -> int | None:
    """Derive a future incident label without turning censoring into negatives.

    Args:
        row: Validated at-risk row.

    Returns:
        int | None: One for breach in (cutoff, horizon], zero for a fully observed
        event-free horizon, or None when follow-up ends before classification.
    """
    breach = row["followup"]["first_breach_ms"]
    if breach is not None and breach <= row["horizon_end_ms"]:
        return 1
    if row["followup"]["observed_through_ms"] >= row["horizon_end_ms"]:
        return 0
    return None


def require_warning_split(
    training: Sequence[WarningRow],
    evaluation: Sequence[WarningRow],
    as_of_ms: int,
) -> None:
    """Require later, borrower-disjoint evaluation with matured training labels.

    Args:
        training: Validated labeled rows proposed for fitting.
        evaluation: Validated labeled rows proposed for scoring.
        as_of_ms: Timestamp by which all evaluation outcome evidence must be known.

    Returns:
        None: Both partitions satisfy the evidence contract.

    Raises:
        ValueError: WARN-008 for empty sets, duplicate opportunities or borrower overlap;
            WARN-009 for censoring, lookahead or outcomes unavailable at analysis time.
    """
    if not training or not evaluation:
        raise ValueError("[WARN-008] training and evaluation must both be nonempty")
    train_borrowers = {row["disclosure"]["borrower"] for row in training}
    test_borrowers = {row["disclosure"]["borrower"] for row in evaluation}
    if train_borrowers & test_borrowers:
        raise ValueError("[WARN-008] training and evaluation borrowers must be disjoint")
    first_test = min(row["cutoff_ms"] for row in evaluation)
    horizons = {row["horizon_end_ms"] - row["cutoff_ms"] for row in (*training, *evaluation)}
    if len(horizons) != 1:
        raise ValueError("[WARN-008] all opportunities must share the frozen forecast horizon")
    for rows, limit in ((training, first_test - 1), (evaluation, as_of_ms)):
        keys = [(row["disclosure"]["borrower"], row["cutoff_ms"]) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("[WARN-008] duplicate borrower prediction opportunity")
        for row in rows:
            if warning_label(row) is None:
                raise ValueError("[WARN-009] censored rows cannot enter a binary labeled split")
            if row["horizon_end_ms"] > limit or row["followup"]["available_ms"] > limit:
                raise ValueError("[WARN-009] prediction or label evidence exceeds partition cutoff")


__all__ = [
    "WarningRow",
    "decode_warning_row",
    "encode_warning_row",
    "require_warning_split",
    "select_warning_disclosure",
    "warning_label",
]
