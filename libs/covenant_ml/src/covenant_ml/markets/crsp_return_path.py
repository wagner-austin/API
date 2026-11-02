"""Compound evidenced held-return intervals without valuing missing closes."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

from platform_core.json_utils import JSONObject, require_float

from .crsp_calendar import crsp_return_span


def _missing_price(row: JSONObject) -> bool:
    """Recognize only the explicit unpriced, nonterminal MP/MR observation."""
    return (
        row["dlyret"] is None
        and row["dlyprevdt"] is None
        and row["dlyretmissflg"] == "MP"
        and row["dlyretdurflg"] == "MR"
        and row["dlydelflg"] == "N"
        and row["dlyprc"] is None
        and row["dlyprcflg"] == "MP"
    )


def crsp_compound_held(
    returns: Mapping[tuple[int, str], JSONObject],
    permno: int,
    entry: str,
    days: tuple[str, ...],
) -> float:
    """Consume a complete sequence of one- or multi-session held returns once.

    Args:
        returns: Decoded daily rows indexed by security and date.
        permno: Owned security, never filtered using its later observations.
        entry: Validated entry close, the first return's required predecessor.
        days: Every independently declared session after entry, ending at the
            exit or in-window terminal event. Later rows cannot close a gap.

    Returns:
        Compounded value at the end of the complete interval. Explicit MP/MR
        rows defer measurement until an NA/P1-P9 return closes exactly that
        span. No value or zero return is assigned to an intermediate close.

    Raises:
        ValueError: CRSP-009 for missing rows, inconsistent spans or nonfinite
            growth; CRSP-011 for an unresolved missing-price tail.
    """
    growth = 1.0
    previous = entry
    missing = 0
    for day in days:
        row = returns.get((permno, day))
        if row is None:
            raise ValueError(f"[CRSP-009] held return for {permno} on {day} is absent")
        if _missing_price(row):
            missing += 1
            continue
        if not crsp_return_span(row, previous, missing + 1):
            raise ValueError(f"[CRSP-009] held return for {permno} on {day} has an invalid span")
        growth *= 1 + require_float(row, "dlyret")
        previous, missing = day, 0
    if missing:
        raise ValueError(f"[CRSP-011] held return for {permno} ends with an unresolved price gap")
    if not isfinite(growth):
        raise ValueError("[CRSP-009] compounded held return is not finite")
    return growth
