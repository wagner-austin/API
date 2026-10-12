"""Strict encode/decode boundaries for immutable market evidence and rules."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_bool,
    require_list,
)

from .types import DOLLAR, MakerRule, MarketHistory, Quote, Trade
from .validation import (
    require_bounded_int,
    require_day,
    require_nonempty,
    require_optional_price,
)

MAX_INTEGER = 2**63 - 1


def encode_quote(record: Quote) -> JSONObject:
    """Encode a quote, preserving explicitly missing sides.

    Args:
        record: Validated quote.

    Returns:
        JSON object.
    """
    return {"timestamp_ms": record["timestamp_ms"], "bid": record["bid"], "ask": record["ask"]}


def decode_quote(document: JSONObject) -> Quote:
    """Validate a quote's time and nullable price sides.

    Args:
        document: JSON object.

    Returns:
        Immutable typed quote.

    Raises:
        ValueError: Invalid timestamp or price range.
        TypeError: Missing or incorrectly typed fields.
    """
    return Quote(
        timestamp_ms=require_bounded_int(document, "timestamp_ms", 0, MAX_INTEGER),
        bid=require_optional_price(document, "bid"),
        ask=require_optional_price(document, "ask"),
    )


def encode_trade(record: Trade) -> JSONObject:
    """Encode a normalized public trade.

    Args:
        record: Validated trade.

    Returns:
        JSON object.
    """
    return {
        "trade_id": record["trade_id"],
        "timestamp_ms": record["timestamp_ms"],
        "price": record["price"],
        "quantity": record["quantity"],
        "taker_yes": record["taker_yes"],
        "block": record["block"],
    }


def decode_trade(document: JSONObject) -> Trade:
    """Validate public trade identity, precision, side and block status.

    Args:
        document: JSON object.

    Returns:
        Immutable typed trade.

    Raises:
        ValueError: Invalid identity or numeric range.
        TypeError: Missing or incorrectly typed fields.
    """
    return Trade(
        trade_id=require_nonempty(document, "trade_id"),
        timestamp_ms=require_bounded_int(document, "timestamp_ms", 0, MAX_INTEGER),
        price=require_bounded_int(document, "price", 0, DOLLAR),
        quantity=require_bounded_int(document, "quantity", 1, MAX_INTEGER),
        taker_yes=require_bool(document, "taker_yes"),
        block=require_bool(document, "block"),
    )


def encode_history(record: MarketHistory) -> JSONObject:
    """Encode the complete evidence used for one order.

    Args:
        record: Validated history.

    Returns:
        JSON object containing every quote and trade.
    """
    quotes: list[JSONValue] = [encode_quote(q) for q in record["quotes"]]
    trades: list[JSONValue] = [encode_trade(t) for t in record["trades"]]
    return {
        "ticker": record["ticker"],
        "series": record["series"],
        "event_day": record["event_day"],
        "decision_ms": record["decision_ms"],
        "close_ms": record["close_ms"],
        "settlement": record["settlement"],
        "quotes": quotes,
        "trades": trades,
    }


def decode_history(document: JSONObject) -> MarketHistory:
    """Validate complete history, rejecting duplicate evidence and invalid chronology.

    Args:
        document: JSON object.

    Returns:
        Immutable typed history.

    Raises:
        ValueError: MARKET-007 for chronology, MARKET-008 for duplicate records.
        TypeError: Missing or incorrectly typed fields.
    """
    decision = require_bounded_int(document, "decision_ms", 0, MAX_INTEGER)
    close = require_bounded_int(document, "close_ms", 0, MAX_INTEGER)
    if close <= decision:
        raise ValueError("[MARKET-007] close must follow the decision")
    quotes = tuple(decode_quote(narrow_json_to_dict(q)) for q in require_list(document, "quotes"))
    trades = tuple(decode_trade(narrow_json_to_dict(t)) for t in require_list(document, "trades"))
    if len({q["timestamp_ms"] for q in quotes}) != len(quotes):
        raise ValueError("[MARKET-008] duplicate quote timestamps")
    if len({t["trade_id"] for t in trades}) != len(trades):
        raise ValueError("[MARKET-008] duplicate trade identifiers")
    return MarketHistory(
        ticker=require_nonempty(document, "ticker"),
        series=require_nonempty(document, "series"),
        event_day=require_day(document, "event_day"),
        decision_ms=decision,
        close_ms=close,
        settlement=require_bounded_int(document, "settlement", 0, DOLLAR),
        quotes=quotes,
        trades=trades,
    )


def encode_rule(record: MakerRule) -> JSONObject:
    """Encode every choice a maker experiment fixes before looking at returns.

    Args:
        record: Validated rule.

    Returns:
        JSON object with no implicit defaults.
    """
    return {
        "min_ask": record["min_ask"],
        "max_ask": record["max_ask"],
        "improve": record["improve"],
        "through": record["through"],
        "contracts": record["contracts"],
        "max_quote_age_ms": record["max_quote_age_ms"],
        "latency_ms": record["latency_ms"],
        "maker_rate_bps": record["maker_rate_bps"],
    }


def decode_rule(document: JSONObject) -> MakerRule:
    """Validate price bands, quote age, latency, size and fees together.

    Args:
        document: JSON object.

    Returns:
        Immutable typed rule.

    Raises:
        ValueError: MARKET-009 for an inconsistent band or nonpositive offer.
        TypeError: Missing or incorrectly typed fields.
    """
    low = require_bounded_int(document, "min_ask", 1, DOLLAR - 1)
    high = require_bounded_int(document, "max_ask", 1, DOLLAR - 1)
    improve = require_bounded_int(document, "improve", 1, DOLLAR - 1)
    if low > high or improve >= low:
        raise ValueError("[MARKET-009] band must be ordered and improvement below minimum ask")
    return MakerRule(
        min_ask=low,
        max_ask=high,
        improve=improve,
        through=require_bounded_int(document, "through", 0, DOLLAR),
        contracts=require_bounded_int(document, "contracts", 1, MAX_INTEGER),
        max_quote_age_ms=require_bounded_int(document, "max_quote_age_ms", 0, MAX_INTEGER),
        latency_ms=require_bounded_int(document, "latency_ms", 0, MAX_INTEGER),
        maker_rate_bps=require_bounded_int(document, "maker_rate_bps", 0, DOLLAR),
    )


__all__ = [
    "decode_history",
    "decode_quote",
    "decode_rule",
    "decode_trade",
    "encode_history",
    "encode_quote",
    "encode_rule",
    "encode_trade",
]
