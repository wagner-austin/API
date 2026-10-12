"""Immutable acquisition windows and visible binary-book liquidity for paper research."""

from __future__ import annotations

from datetime import date, timedelta

from platform_core.json_utils import (
    JSONObject,
    require_dict,
    require_list,
    require_str,
)
from typing_extensions import ReadOnly, TypedDict

from .forecast_records import CITY_COORDINATES
from .validation import (
    parse_fixed,
    parse_timestamp,
    require_bounded_int,
    require_day,
    require_nonempty,
)


class PaperPlan(TypedDict):
    """Frozen single-city, single-day prospective acquisition specification."""

    name: ReadOnly[str]
    series: ReadOnly[str]
    event_day: ReadOnly[str]
    forecast_hour: ReadOnly[int]
    window_start: ReadOnly[str]
    window_end: ReadOnly[str]
    model_digest: ReadOnly[str]
    protocol_digest: ReadOnly[str]


class BookTop(TypedDict):
    """YES prices in 1/10000 dollars and available size in 1/100 contracts."""

    bid: ReadOnly[int]
    ask: ReadOnly[int]
    bid_size: ReadOnly[int]
    ask_size: ReadOnly[int]


def encode_paper_plan(plan: PaperPlan) -> JSONObject:
    """Encode all frozen acquisition constraints.

    Args:
        plan: Validated immutable plan.

    Returns:
        JSON plan object.
    """
    return {
        "name": plan["name"],
        "series": plan["series"],
        "event_day": plan["event_day"],
        "forecast_hour": plan["forecast_hour"],
        "window_start": plan["window_start"],
        "window_end": plan["window_end"],
        "model_digest": plan["model_digest"],
        "protocol_digest": plan["protocol_digest"],
    }


def decode_paper_plan(document: JSONObject) -> PaperPlan:
    """Require a bounded window strictly before the target standard-time day.

    Args:
        document: Decoded plan object.

    Returns:
        Immutable plan with validated content identities and timing.

    Raises:
        ValueError: PAPER-001 for invalid dates, cycle, city, window or digests.
        TypeError: Required fields are missing or malformed.
    """
    day = require_day(document, "event_day")
    series = require_nonempty(document, "series")
    hour = require_bounded_int(document, "forecast_hour", 0, 12)
    start, end = require_str(document, "window_start"), require_str(document, "window_end")
    start_ms, end_ms = parse_timestamp(start), parse_timestamp(end)
    prior = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    initialized = parse_timestamp(prior + f"T{hour:02d}:00:00Z")
    if (
        series not in CITY_COORDINATES
        or hour not in (0, 6, 12)
        or not initialized < start_ms < end_ms < parse_timestamp(day + "T05:00:00Z")
        or end_ms - start_ms > 14400000
    ):
        raise ValueError("[PAPER-001] require a declared city/cycle and pre-event window <=4 hours")
    model, protocol = (
        require_nonempty(document, "model_digest"),
        require_nonempty(document, "protocol_digest"),
    )
    for digest in (model, protocol):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("[PAPER-001] content identities must be lowercase SHA-256")
    return PaperPlan(
        name=require_nonempty(document, "name"),
        series=series,
        event_day=day,
        forecast_hour=hour,
        window_start=start,
        window_end=end,
        model_digest=model,
        protocol_digest=protocol,
    )


def encode_book_top(book: BookTop) -> JSONObject:
    """Encode visible prices and sizes without floating-point conversions.

    Args:
        book: Validated two-sided book.

    Returns:
        JSON liquidity object.
    """
    return {
        "bid": book["bid"],
        "ask": book["ask"],
        "bid_size": book["bid_size"],
        "ask_size": book["ask_size"],
    }


def decode_book_top(document: JSONObject) -> BookTop:
    """Validate a noncrossed two-sided book with strictly positive visible size.

    Args:
        document: Decoded normalized book object.

    Returns:
        Immutable book top.

    Raises:
        ValueError: PAPER-002 for crossed prices; MARKET-001 for invalid bounds.
        TypeError: Missing or malformed integers.
    """
    bid, ask = (
        require_bounded_int(document, "bid", 0, 10000),
        require_bounded_int(document, "ask", 0, 10000),
    )
    if bid > ask:
        raise ValueError("[PAPER-002] crossed binary order book")
    return BookTop(
        bid=bid,
        ask=ask,
        bid_size=require_bounded_int(document, "bid_size", 1, 2**63 - 1),
        ask_size=require_bounded_int(document, "ask_size", 1, 2**63 - 1),
    )


def _best_bid(document: JSONObject, key: str) -> tuple[int, int] | None:
    """Validate every native price level and select the highest bid deterministically.

    Args:
        document: Native fixed-point order book.
        key: Required side array.

    Returns:
        Best price and size, or explicit absent-side state.

    Raises:
        ValueError: PAPER-003 for malformed, duplicate or out-of-range levels.
        TypeError: Native side, price or quantity has an incorrect type.
    """
    levels: dict[int, int] = {}
    for value in require_list(document, key):
        pair = require_list({"level": value}, "level")
        if len(pair) != 2:
            raise ValueError("[PAPER-003] order book level needs price and size")
        price = parse_fixed(require_str({"price": pair[0]}, "price"), 4)
        size = parse_fixed(require_str({"size": pair[1]}, "size"), 2)
        if price > 10000 or not 0 < size < 2**63 or price in levels:
            raise ValueError("[PAPER-003] invalid or duplicate order book level")
        levels[price] = size
    if not levels:
        return None
    price = max(levels)
    return price, levels[price]


def decode_native_book(document: JSONObject) -> BookTop | None:
    """Derive YES asks from opposing NO bids, retaining matching available size.

    Args:
        document: Native public orderbook response.

    Returns:
        Validated two-sided top, or explicit exclusion for a missing side.

    Raises:
        ValueError: Invalid, duplicate or crossed price levels.
        TypeError: Malformed native fields; absence never becomes empty liquidity.
    """
    native = require_dict(document, "orderbook_fp")
    yes, no = _best_bid(native, "yes_dollars"), _best_bid(native, "no_dollars")
    if yes is None or no is None:
        return None
    return decode_book_top(
        {"bid": yes[0], "ask": 10000 - no[0], "bid_size": yes[1], "ask_size": no[1]}
    )


__all__ = [
    "BookTop",
    "PaperPlan",
    "decode_book_top",
    "decode_native_book",
    "decode_paper_plan",
    "encode_book_top",
    "encode_paper_plan",
]
