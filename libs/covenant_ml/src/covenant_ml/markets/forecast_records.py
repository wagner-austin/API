"""Immutable contracts for the chronological incremental-information experiment."""

from __future__ import annotations

import math

from platform_core.json_utils import JSONObject, require_float, require_list
from typing_extensions import ReadOnly, TypedDict

from .validation import require_bounded_int, require_day, require_nonempty

CITY_COORDINATES: dict[str, tuple[str, str]] = {
    "KXHIGHTBOS": ("42.36", "-71.01"),
    "KXHIGHPHIL": ("39.87", "-75.24"),
    "KXHIGHTDC": ("38.85", "-77.04"),
}


class ComparisonPlan(TypedDict):
    """Fixed chronological partitions bound to a complete research protocol."""

    name: ReadOnly[str]
    start: ReadOnly[str]
    train_end: ReadOnly[str]
    end: ReadOnly[str]
    protocol_digest: ReadOnly[str]
    excluded_days: ReadOnly[tuple[str, ...]]
    forecast_hour: ReadOnly[int]


class ForecastRow(TypedDict):
    """An auditable binary contract observation, prior quote and forecast feature."""

    ticker: ReadOnly[str]
    series: ReadOnly[str]
    event_day: ReadOnly[str]
    decision_ms: ReadOnly[int]
    settled_ms: ReadOnly[int]
    quote_ms: ReadOnly[int]
    forecast_init_ms: ReadOnly[int]
    yes: ReadOnly[int]
    bid: ReadOnly[int]
    ask: ReadOnly[int]
    forecast_f: ReadOnly[float]
    weather_p: ReadOnly[float]
    settlement_rule: ReadOnly[str]


def encode_comparison_plan(plan: ComparisonPlan) -> JSONObject:
    """Encode every partition and its protocol identity.

    Args:
        plan: Validated immutable plan.

    Returns:
        JSON object.
    """
    return {
        "name": plan["name"],
        "start": plan["start"],
        "train_end": plan["train_end"],
        "end": plan["end"],
        "protocol_digest": plan["protocol_digest"],
        "excluded_days": list(plan["excluded_days"]),
        "forecast_hour": plan["forecast_hour"],
    }


def decode_comparison_plan(document: JSONObject) -> ComparisonPlan:
    """Validate ordered training/evaluation dates and protocol SHA-256.

    Args:
        document: Decoded plan document.

    Returns:
        Immutable experiment plan.

    Raises:
        ValueError: FORECAST-001 for invalid chronology or digest.
        TypeError: Required fields have incorrect types.
    """
    start = require_day(document, "start")
    train_end = require_day(document, "train_end")
    end = require_day(document, "end")
    digest = require_nonempty(document, "protocol_digest")
    if not start <= train_end < end or len(digest) != 64:
        raise ValueError("[FORECAST-001] require ordered dates and a SHA-256 digest")
    if any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("[FORECAST-001] protocol digest must be lowercase hexadecimal")
    excluded = tuple(
        require_day({"day": value}, "day") for value in require_list(document, "excluded_days")
    )
    if len(excluded) != len(set(excluded)) or any(not start <= day <= end for day in excluded):
        raise ValueError("[FORECAST-001] excluded dates must be unique and within the experiment")
    hour = require_bounded_int(document, "forecast_hour", 0, 12)
    if hour not in (0, 6, 12):
        raise ValueError("[FORECAST-001] forecast hour must identify a prior GFS cycle")
    return ComparisonPlan(
        name=require_nonempty(document, "name"),
        start=start,
        train_end=train_end,
        end=end,
        protocol_digest=digest,
        excluded_days=excluded,
        forecast_hour=hour,
    )


def encode_forecast_row(row: ForecastRow) -> JSONObject:
    """Encode one complete comparison row.

    Args:
        row: Validated immutable observation.

    Returns:
        JSON object retaining timing and settlement terms.
    """
    return {
        "ticker": row["ticker"],
        "series": row["series"],
        "event_day": row["event_day"],
        "decision_ms": row["decision_ms"],
        "settled_ms": row["settled_ms"],
        "quote_ms": row["quote_ms"],
        "forecast_init_ms": row["forecast_init_ms"],
        "yes": row["yes"],
        "bid": row["bid"],
        "ask": row["ask"],
        "forecast_f": row["forecast_f"],
        "weather_p": row["weather_p"],
        "settlement_rule": row["settlement_rule"],
    }


def decode_forecast_row(document: JSONObject) -> ForecastRow:
    """Require valid prices, binary labels and strictly prior forecast information.

    Args:
        document: Decoded comparison observation.

    Returns:
        Immutable validated row.

    Raises:
        ValueError: FORECAST-002 for inconsistent identity, chronology or features.
        TypeError: A required field is malformed.
    """
    series = require_nonempty(document, "series")
    ticker = require_nonempty(document, "ticker")
    decision = require_bounded_int(document, "decision_ms", 0, 2**63 - 1)
    settled = require_bounded_int(document, "settled_ms", 0, 2**63 - 1)
    quote = require_bounded_int(document, "quote_ms", 0, 2**63 - 1)
    initialized = require_bounded_int(document, "forecast_init_ms", 0, 2**63 - 1)
    bid = require_bounded_int(document, "bid", 0, 10000)
    ask = require_bounded_int(document, "ask", 0, 10000)
    probability = require_float(document, "weather_p")
    forecast = require_float(document, "forecast_f")
    if series not in CITY_COORDINATES or not ticker.startswith(series + "-"):
        raise ValueError("[FORECAST-002] contract does not match a declared city")
    if not 0 <= decision - quote <= 3600000 or initialized + 21600000 > decision:
        raise ValueError("[FORECAST-002] stale quote or forecast unavailable at decision")
    if settled <= decision or bid > ask:
        raise ValueError("[FORECAST-002] crossed quote or invalid settlement chronology")
    if not math.isfinite(forecast) or not 0 <= probability <= 1:
        raise ValueError("[FORECAST-002] forecast and probability must be finite and valid")
    return ForecastRow(
        ticker=ticker,
        series=series,
        event_day=require_day(document, "event_day"),
        decision_ms=decision,
        settled_ms=settled,
        quote_ms=quote,
        forecast_init_ms=initialized,
        yes=require_bounded_int(document, "yes", 0, 1),
        bid=bid,
        ask=ask,
        forecast_f=forecast,
        weather_p=probability,
        settlement_rule=require_nonempty(document, "settlement_rule"),
    )


__all__ = [
    "CITY_COORDINATES",
    "ComparisonPlan",
    "ForecastRow",
    "decode_comparison_plan",
    "decode_forecast_row",
    "encode_comparison_plan",
    "encode_forecast_row",
]
