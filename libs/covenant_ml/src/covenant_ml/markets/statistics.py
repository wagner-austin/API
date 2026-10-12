"""Inference over event dates, reusing the library's paired estimators."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray
from platform_core.json_utils import JSONObject, require_dict, require_int
from typing_extensions import ReadOnly, TypedDict

from ..evaluation.codec_records import decode_interval, encode_interval
from ..evaluation.paired import paired_mean_difference
from ..evaluation.resampling import build_resampling
from ..evaluation.types import IntervalEstimate, ResamplingUnit
from .types import DOLLAR, ReplayResult, ReplayStatus


class ReplaySummary(TypedDict):
    """Economic totals and uncertainty; insufficient samples have explicit nulls."""

    markets: ReadOnly[int]
    fills: ReadOnly[int]
    contracts: ReadOnly[int]
    event_days: ReadOnly[int]
    pnl: ReadOnly[int]
    max_drawdown: ReadOnly[int]
    per_order: ReadOnly[IntervalEstimate | None]
    per_calendar_day: ReadOnly[IntervalEstimate | None]
    friction_adjusted_pnl: ReadOnly[int]
    friction_per_contract: ReadOnly[int]


def summarize_replay(
    results: Sequence[ReplayResult],
    start: date,
    end: date,
    friction: int,
    confidence: float = 0.95,
    serial_lags: int = 7,
) -> ReplaySummary:
    """Report cluster-by-day order P&L and serially robust daily P&L.

    Args:
        results: Unique-contract replay results.
        start: First calendar day, including days without fills.
        end: Last calendar day, inclusive.
        friction: Extra execution cost per filled contract in currency units.
        confidence: Interval coverage.
        serial_lags: Newey-West lag count for calendar-day P&L.

    Returns:
        Totals, event-date aggregated drawdown, and explicit uncertainty estimates.

    Raises:
        ValueError: MARKET-013 for invalid dates, duplicates or cost assumptions.
    """
    if end < start or friction < 0 or serial_lags < 1 or not 0 < confidence < 1:
        raise ValueError("[MARKET-013] invalid dates, friction, lags or confidence")
    if len({r["ticker"] for r in results}) != len(results):
        raise ValueError("[MARKET-013] duplicated contract in summary")
    n_days = (end - start).days + 1
    daily = {(start + timedelta(days=i)).isoformat(): 0 for i in range(n_days)}
    filled = [r for r in results if r["status"] is ReplayStatus.FILLED]
    for result in results:
        if result["event_day"] not in daily:
            raise ValueError("[MARKET-013] result outside the declared date window")
        daily[result["event_day"]] += result["pnl"]
    order_interval: IntervalEstimate | None = None
    daily_interval: IntervalEstimate | None = None
    days = {r["event_day"] for r in filled}
    if len(days) >= 2:
        pnl_values: list[float] = [r["pnl"] / DOLLAR for r in filled]
        day_values: list[int] = [int(r["event_day"].replace("-", "")) for r in filled]
        pnl: NDArray[np.float64] = np.asarray(pnl_values, dtype=np.float64)
        clusters: NDArray[np.int64] = np.asarray(day_values, dtype=np.int64)
        units = build_resampling(ResamplingUnit.CLUSTER, len(filled), clusters, 1)
        order_interval = paired_mean_difference(pnl, units, confidence)
    if n_days > serial_lags + 1:
        daily_values: list[float] = [value / DOLLAR for value in daily.values()]
        daily_pnl: NDArray[np.float64] = np.asarray(daily_values, dtype=np.float64)
        indices: NDArray[np.int64] = np.arange(n_days, dtype=np.int64)
        units = build_resampling(ResamplingUnit.BLOCK, n_days, indices, serial_lags + 1)
        daily_interval = paired_mean_difference(daily_pnl, units, confidence)
    running = peak = drawdown = 0
    for amount in daily.values():
        running += amount
        peak = max(peak, running)
        drawdown = max(drawdown, peak - running)
    contracts = sum(r["contracts"] for r in filled)
    return ReplaySummary(
        markets=len(results),
        fills=len(filled),
        contracts=contracts,
        event_days=len(days),
        pnl=running,
        max_drawdown=drawdown,
        per_order=order_interval,
        per_calendar_day=daily_interval,
        friction_adjusted_pnl=running - friction * contracts,
        friction_per_contract=friction,
    )


def encode_summary(record: ReplaySummary) -> JSONObject:
    """Encode a replay summary and the existing interval record types.

    Args:
        record: Summary produced by the evaluator.

    Returns:
        JSON object with explicitly null insufficient-sample intervals.
    """
    per_order, per_day = record["per_order"], record["per_calendar_day"]
    return {
        "markets": record["markets"],
        "fills": record["fills"],
        "contracts": record["contracts"],
        "event_days": record["event_days"],
        "pnl": record["pnl"],
        "max_drawdown": record["max_drawdown"],
        "per_order": None if per_order is None else encode_interval(per_order),
        "per_calendar_day": None if per_day is None else encode_interval(per_day),
        "friction_adjusted_pnl": record["friction_adjusted_pnl"],
        "friction_per_contract": record["friction_per_contract"],
    }


def decode_summary(document: JSONObject) -> ReplaySummary:
    """Decode a summary and enforce its accounting identities.

    Args:
        document: JSON object.

    Returns:
        Immutable typed summary.

    Raises:
        ValueError: MARKET-014 for inconsistent counts or accounting.
        TypeError: Missing or incorrectly typed fields.
        KeyError: Missing required nullable interval fields.
    """
    markets, fills = require_int(document, "markets"), require_int(document, "fills")
    contracts, days = require_int(document, "contracts"), require_int(document, "event_days")
    pnl, adjusted = require_int(document, "pnl"), require_int(document, "friction_adjusted_pnl")
    friction = require_int(document, "friction_per_contract")
    drawdown = require_int(document, "max_drawdown")
    if not 0 <= days <= fills <= markets or contracts < fills or min(friction, drawdown) < 0:
        raise ValueError("[MARKET-014] invalid summary counts")
    if (fills == 0) != (days == 0) or (fills == 0) != (contracts == 0):
        raise ValueError("[MARKET-014] empty fills, event days and positions must agree")
    if adjusted != pnl - friction * contracts:
        raise ValueError("[MARKET-014] friction-adjusted P&L does not reconcile")
    per_order = (
        None
        if document["per_order"] is None
        else decode_interval(require_dict(document, "per_order"))
    )
    per_day = (
        None
        if document["per_calendar_day"] is None
        else decode_interval(require_dict(document, "per_calendar_day"))
    )
    if (per_order is None) != (days < 2):
        raise ValueError("[MARKET-014] order interval requires at least two filled event dates")
    return ReplaySummary(
        markets=markets,
        fills=fills,
        contracts=contracts,
        event_days=days,
        pnl=pnl,
        max_drawdown=drawdown,
        per_order=per_order,
        per_calendar_day=per_day,
        friction_adjusted_pnl=adjusted,
        friction_per_contract=friction,
    )


__all__ = ["ReplaySummary", "decode_summary", "encode_summary", "summarize_replay"]
