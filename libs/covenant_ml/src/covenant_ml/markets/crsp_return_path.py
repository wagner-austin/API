"""Compound evidenced held-return intervals without valuing missing closes."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite, prod
from typing import NamedTuple

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


class HeldRefusal(NamedTuple):
    """Why a held interval cannot be valued from the vendor's rows.

    Attributes:
        reason: A short key for counting: ``absent row``, ``span <missing
            flag>/<duration flag>`` (with `` after a price gap`` when MP rows
            preceded it) or ``price gap tail``.
        message: The full error, naming the code, security and session.
    """

    reason: str
    message: str


class HeldResolution(NamedTuple):
    """A held interval's spans, or the first reason it cannot be valued.

    Attributes:
        spans: One ``(index, growth)`` per resolved span; empty when refused.
        refusal: None when every session resolved.
    """

    spans: tuple[tuple[int, float], ...]
    refusal: HeldRefusal | None


def _refused_span(permno: int, day: str, row: JSONObject, missing: int) -> HeldRefusal:
    """Describe a present row whose return does not close the span it must.

    Args:
        permno: Owned security.
        day: The row's session.
        row: The refused row.
        missing: Explicit MP/MR rows immediately before it.

    Returns:
        The refusal, keyed by the row's own missing and duration flags.
    """
    after = " after a price gap" if missing else ""
    return HeldRefusal(
        f"span {row['dlyretmissflg']}/{row['dlyretdurflg']}{after}",
        f"[CRSP-009] held return for {permno} on {day} has an invalid span",
    )


def crsp_resolve_held(
    returns: Mapping[tuple[int, str], JSONObject],
    permno: int,
    entry: str,
    days: tuple[str, ...],
) -> HeldResolution:
    """Resolve held returns span by span, reporting rather than raising the first refusal.

    Args:
        returns: Decoded daily rows indexed by security and date.
        permno: Owned security, never filtered using its later observations.
        entry: Validated entry close, the first return's required predecessor.
        days: Every independently declared session after entry, ending at the
            exit or in-window terminal event. Later rows cannot close a gap.

    Returns:
        One ``(index, growth)`` per resolved span, in order: the position in
        ``days`` of the session whose NA/P1-P9 return closes the span, and one
        plus that return. Explicit MP/MR rows inside a span get no entry; no
        value is assigned to an intermediate close. A missing row, a row whose
        flags do not close its span (including a return the vendor flags
        missing, such as MV, missing corporate action value) or an unclosed
        price-gap tail is the refusal instead.
    """
    spans: list[tuple[int, float]] = []
    previous = entry
    missing = 0
    for index, day in enumerate(days):
        row = returns.get((permno, day))
        if row is None:
            message = f"[CRSP-009] held return for {permno} on {day} is absent"
            return HeldResolution((), HeldRefusal("absent row", message))
        if _missing_price(row):
            missing += 1
            continue
        if not crsp_return_span(row, previous, missing + 1):
            return HeldResolution((), _refused_span(permno, day, row, missing))
        spans.append((index, 1 + require_float(row, "dlyret")))
        previous, missing = day, 0
    if missing:
        message = f"[CRSP-011] held return for {permno} ends with an unresolved price gap"
        return HeldResolution((), HeldRefusal("price gap tail", message))
    return HeldResolution(tuple(spans), None)


def crsp_held_spans(
    returns: Mapping[tuple[int, str], JSONObject],
    permno: int,
    entry: str,
    days: tuple[str, ...],
) -> tuple[tuple[int, float], ...]:
    """Resolve a complete sequence of one- or multi-session held returns, span by span.

    Args:
        returns: Decoded daily rows indexed by security and date.
        permno: Owned security, never filtered using its later observations.
        entry: Validated entry close, the first return's required predecessor.
        days: Every independently declared session after entry, ending at the
            exit or in-window terminal event. Later rows cannot close a gap.

    Returns:
        :func:`crsp_resolve_held`'s spans.

    Raises:
        ValueError: CRSP-009 for missing rows or inconsistent spans; CRSP-011
            for an unresolved missing-price tail.
    """
    resolution = crsp_resolve_held(returns, permno, entry, days)
    if resolution.refusal is not None:
        raise ValueError(resolution.refusal.message)
    return resolution.spans


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
        Compounded value at the end of the complete interval: the product of
        :func:`crsp_held_spans`' growths, in order.

    Raises:
        ValueError: CRSP-009 for missing rows, inconsistent spans or nonfinite
            growth; CRSP-011 for an unresolved missing-price tail.
    """
    growth = prod(span for _, span in crsp_held_spans(returns, permno, entry, days))
    if not isfinite(growth):
        raise ValueError("[CRSP-009] compounded held return is not finite")
    return growth
