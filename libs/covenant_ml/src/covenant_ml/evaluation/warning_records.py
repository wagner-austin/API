"""Dated evidence for incident covenant-warning studies, never inferred vintages."""

from __future__ import annotations

import math

from platform_core.json_utils import JSONObject, require_float, require_int, require_str
from typing_extensions import ReadOnly, TypedDict


class WarningDisclosure(TypedDict):
    """Immutable financial ratios from one identified disclosure vintage."""

    borrower: ReadOnly[str]
    period_end_ms: ReadOnly[int]
    available_ms: ReadOnly[int]
    source_sha256: ReadOnly[str]
    debt_to_ebitda: ReadOnly[float]
    interest_cover: ReadOnly[float]
    current_ratio: ReadOnly[float]


class WarningFollowup(TypedDict):
    """Complete incident-event surveillance over an explicitly bounded interval.

    first_breach_ms is the first contractual breach within this coverage interval,
    not its disclosure date. available_ms is when this outcome evidence was known.
    Absence of a breach is asserted only through observed_through_ms.
    """

    borrower: ReadOnly[str]
    coverage_start_ms: ReadOnly[int]
    observed_through_ms: ReadOnly[int]
    available_ms: ReadOnly[int]
    first_breach_ms: ReadOnly[int | None]
    source_sha256: ReadOnly[str]


def require_warning_time(document: JSONObject, key: str) -> int:
    """Require a nonnegative, exactly representable epoch millisecond.

    Args:
        document: JSON evidence object.
        key: Required timestamp field.

    Returns:
        int: Validated millisecond timestamp.

    Raises:
        TypeError: Field is missing or not an integer, including booleans.
        ValueError: WARN-001 when outside the supported timestamp range.
    """
    value = require_int(document, key)
    if not 0 <= value <= 2**53 - 1:
        raise ValueError(f"[WARN-001] {key} must be a nonnegative exact millisecond")
    return value


def _require_identity(document: JSONObject) -> tuple[str, str]:
    """Require a borrower identity and lowercase source SHA256.

    Args:
        document: JSON evidence object.

    Returns:
        tuple[str, str]: Borrower identifier and source digest.

    Raises:
        TypeError: Required strings are missing or invalid.
        ValueError: WARN-002 for invalid identity or source digest.
    """
    borrower = require_str(document, "borrower")
    digest = require_str(document, "source_sha256")
    if not borrower.strip() or borrower != borrower.strip():
        raise ValueError("[WARN-002] borrower must be nonblank without surrounding whitespace")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("[WARN-002] source_sha256 must identify archived evidence")
    return borrower, digest


def _require_ratio(document: JSONObject, key: str) -> float:
    """Require a finite financial ratio without imputing missing values.

    Args:
        document: JSON disclosure.
        key: Required ratio field.

    Returns:
        float: Finite ratio; negative earnings ratios remain valid.

    Raises:
        TypeError: Ratio is missing or not numeric.
        ValueError: WARN-003 for a nonfinite value.
    """
    value = require_float(document, key)
    if not math.isfinite(value):
        raise ValueError(f"[WARN-003] {key} must be finite")
    return value


def decode_warning_disclosure(document: JSONObject) -> WarningDisclosure:
    """Validate disclosure identity, chronology and finite ratios.

    Args:
        document: JSON disclosure with an explicit availability timestamp.

    Returns:
        WarningDisclosure: Immutable dated inputs.

    Raises:
        TypeError: A required field is missing or has the wrong type.
        ValueError: WARN-001/002/003 for invalid fields; WARN-004 for chronology.
    """
    borrower, digest = _require_identity(document)
    period = require_warning_time(document, "period_end_ms")
    available = require_warning_time(document, "available_ms")
    if available < period:
        raise ValueError("[WARN-004] disclosure availability precedes its period end")
    return WarningDisclosure(
        borrower=borrower,
        period_end_ms=period,
        available_ms=available,
        source_sha256=digest,
        debt_to_ebitda=_require_ratio(document, "debt_to_ebitda"),
        interest_cover=_require_ratio(document, "interest_cover"),
        current_ratio=_require_ratio(document, "current_ratio"),
    )


def encode_warning_disclosure(row: WarningDisclosure) -> JSONObject:
    """Encode a disclosure without mutating its evidence.

    Args:
        row: Validated disclosure.

    Returns:
        JSONObject: Independent JSON representation.
    """
    return {
        "borrower": row["borrower"],
        "period_end_ms": row["period_end_ms"],
        "available_ms": row["available_ms"],
        "source_sha256": row["source_sha256"],
        "debt_to_ebitda": row["debt_to_ebitda"],
        "interest_cover": row["interest_cover"],
        "current_ratio": row["current_ratio"],
    }


def decode_warning_followup(document: JSONObject) -> WarningFollowup:
    """Validate bounded follow-up, distinguishing null from absent breach evidence.

    Args:
        document: JSON incident-event surveillance record.

    Returns:
        WarningFollowup: Immutable outcome evidence.

    Raises:
        TypeError: Required field is absent or incorrectly typed.
        ValueError: WARN-005 for inconsistent coverage or breach chronology.
    """
    borrower, digest = _require_identity(document)
    start = require_warning_time(document, "coverage_start_ms")
    through = require_warning_time(document, "observed_through_ms")
    available = require_warning_time(document, "available_ms")
    if "first_breach_ms" not in document:
        raise ValueError("[WARN-005] explicit first_breach_ms or null is required")
    breach = (
        None
        if document["first_breach_ms"] is None
        else require_warning_time(document, "first_breach_ms")
    )
    if not start <= through <= available:
        raise ValueError("[WARN-005] require coverage start <= observed through <= availability")
    if breach is not None and not start <= breach <= through:
        raise ValueError("[WARN-005] breach must fall inside the asserted coverage interval")
    return WarningFollowup(
        borrower=borrower,
        coverage_start_ms=start,
        observed_through_ms=through,
        available_ms=available,
        first_breach_ms=breach,
        source_sha256=digest,
    )


def encode_warning_followup(row: WarningFollowup) -> JSONObject:
    """Encode the full follow-up interval and outcome provenance.

    Args:
        row: Validated follow-up record.

    Returns:
        JSONObject: Independent JSON representation, including explicit null.
    """
    return {
        "borrower": row["borrower"],
        "coverage_start_ms": row["coverage_start_ms"],
        "observed_through_ms": row["observed_through_ms"],
        "available_ms": row["available_ms"],
        "first_breach_ms": row["first_breach_ms"],
        "source_sha256": row["source_sha256"],
    }


__all__ = [
    "WarningDisclosure",
    "WarningFollowup",
    "decode_warning_disclosure",
    "decode_warning_followup",
    "encode_warning_disclosure",
    "encode_warning_followup",
    "require_warning_time",
]
