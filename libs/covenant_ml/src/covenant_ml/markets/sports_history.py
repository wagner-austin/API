"""Historical NHL selection and exact-cutoff evidence for the frozen experiment."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date

from platform_core.json_utils import (
    JSONObject,
    narrow_json_to_dict,
    require_list,
    require_str,
)

from .kalshi import MarketSpec, decode_candle, decode_spec
from .types import DOLLAR, Quote
from .validation import parse_fixed, parse_timestamp

SERIES = "KXNHLGAME"
TRAIN_START = "2026-01-01"
TEST_START = "2026-03-01"
TEST_END = "2026-04-15"


def event_day(event: str) -> str:
    """Validate an NHL game identity and derive its declared calendar date.

    Args:
        event: Native event ticker string.

    Returns:
        Canonical ISO day, independent of eventual game end.

    Raises:
        ValueError: STRATEGY-001 for an invalid identity; invalid calendar dates.
    """
    match = re.fullmatch(r"KXNHLGAME-(\d{2})([A-Z]{3})(\d{2})[A-Z]{4,6}", event)
    months = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
    if match is None or match[2] not in months:
        raise ValueError("[STRATEGY-001] invalid NHL game event ticker")
    return date(2000 + int(match[1]), months.index(match[2]) + 1, int(match[3])).isoformat()


def historical_inventory(
    read: Callable[[str, dict[str, str | int]], JSONObject],
) -> tuple[JSONObject, ...]:
    """Page the complete archive with unique identities and validated event dates.

    Args:
        read: Explicit capture or offline replay method from ResponseArchive.

    Returns:
        Complete native records, without price or outcome selection.

    Raises:
        ValueError: STRATEGY-002 for repeated cursors or tickers; invalid metadata.
        HTTPStatusError: Acquisition failed.
    """
    cursor = ""
    cursors: set[str] = set()
    tickers: set[str] = set()
    records: list[JSONObject] = []
    while True:
        if cursor in cursors:
            raise ValueError("[STRATEGY-002] historical pagination cursor repeated")
        cursors.add(cursor)
        document = read(
            "/historical/markets", {"series_ticker": SERIES, "limit": 1000, "cursor": cursor}
        )
        for value in require_list(document, "markets"):
            native = narrow_json_to_dict(value)
            ticker, event = require_str(native, "ticker"), require_str(native, "event_ticker")
            if ticker in tickers or not ticker.startswith(event + "-"):
                raise ValueError("[STRATEGY-002] duplicate or mismatched market ticker")
            tickers.add(ticker)
            event_day(event)
            records.append(native)
        cursor = require_str(document, "cursor")
        if not cursor:
            break
    return tuple(records)


def historical_selection(
    read: Callable[[str, dict[str, str | int]], JSONObject],
) -> tuple[tuple[MarketSpec, ...], dict[str, int]]:
    """Choose one contract per game before inspecting outcomes and quote eligibility.

    Args:
        read: Explicit archive capture or replay method.

    Returns:
        Ordered specifications and exclusions for the frozen experiment dates.

    Raises:
        ValueError: Invalid native identities, pagination or chronology.
        HTTPStatusError: Acquisition failed.
    """
    selected: dict[str, JSONObject] = {}
    for native in historical_inventory(read):
        ticker, event = require_str(native, "ticker"), require_str(native, "event_ticker")
        day = event_day(event)
        if TRAIN_START <= day <= TEST_END and (
            event not in selected or ticker < require_str(selected[event], "ticker")
        ):
            selected[event] = native
    return _eligible_specs(selected)


def _spec_key(spec: MarketSpec) -> tuple[str, str]:
    """Order records by date and native ticker.

    Args:
        spec: Validated market record.

    Returns:
        Stable chronological ordering key.
    """
    return spec["day"], spec["ticker"]


def _eligible_specs(
    selected: dict[str, JSONObject],
) -> tuple[tuple[MarketSpec, ...], dict[str, int]]:
    """Validate settlement eligibility after fixing contract selection.

    Args:
        selected: One lexicographically selected native record per event.

    Returns:
        Sorted specifications and explicit exclusions.

    Raises:
        ValueError: Malformed prices, dates or chronology.
    """
    counts = {
        "selected_games": len(selected),
        "nonbinary": 0,
        "not_open": 0,
        "late_training_settlement": 0,
    }
    specs: list[MarketSpec] = []
    for event, native in sorted(selected.items()):
        day = event_day(event)
        cutoff = parse_timestamp(day + "T12:00:00Z")
        close = parse_timestamp(require_str(native, "close_time"))
        settled = parse_timestamp(require_str(native, "settlement_ts"))
        payout = parse_fixed(require_str(native, "settlement_value_dollars"), 4)
        if require_str(native, "status") != "finalized" or payout not in (0, DOLLAR):
            counts["nonbinary"] += 1
        elif not parse_timestamp(require_str(native, "open_time")) <= cutoff < close:
            counts["not_open"] += 1
        elif day < TEST_START and settled >= parse_timestamp(TEST_START + "T00:00:00Z"):
            counts["late_training_settlement"] += 1
        else:
            specs.append(
                decode_spec(
                    {
                        "ticker": require_str(native, "ticker"),
                        "series": SERIES,
                        "day": day,
                        "decision_ms": cutoff,
                        "close_ms": close,
                        "settled_ms": settled,
                        "settlement": payout,
                    }
                )
            )
    return tuple(sorted(specs, key=_spec_key)), counts


def historical_quote(
    spec: MarketSpec,
    read: Callable[[str, dict[str, str | int]], JSONObject],
) -> tuple[Quote | None, str]:
    """Request one exact decision candle without manufacturing an executable fill.

    Args:
        spec: Validated selected game contract.
        read: Explicit archive capture or replay method.

    Returns:
        Eligible quote and status, or an explicit exclusion reason.

    Raises:
        ValueError: STRATEGY-003 for wrong ticker, duplicate or off-cutoff candles.
        HTTPStatusError: Acquisition failed.
    """
    cutoff = spec["decision_ms"] // 1000
    document = read(
        "/historical/markets/" + spec["ticker"] + "/candlesticks",
        {"start_ts": cutoff, "end_ts": cutoff, "period_interval": 1},
    )
    candles = require_list(document, "candlesticks")
    if require_str(document, "ticker") != spec["ticker"] or len(candles) > 1:
        raise ValueError("[STRATEGY-003] candle identity or count mismatch")
    if not candles:
        return None, "missing_candle"
    quote = decode_candle(narrow_json_to_dict(candles[0]), historical=True)
    if quote["timestamp_ms"] != spec["decision_ms"]:
        raise ValueError("[STRATEGY-003] candle does not end at the declared cutoff")
    bid, ask = quote["bid"], quote["ask"]
    if bid is not None and ask is not None and bid > ask:
        raise ValueError("[STRATEGY-003] crossed historical quote")
    if bid is None or ask is None or bid == 0 or ask == DOLLAR:
        return None, "unavailable_quote"
    if ask - bid > 500:
        return None, "wide_spread"
    return quote, "eligible"
