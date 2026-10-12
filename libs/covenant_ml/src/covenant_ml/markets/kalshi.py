"""Current Kalshi public schema decoded into the shared replay records."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from platform_core.json_utils import (
    JSONObject,
    require_bool,
    require_dict,
    require_int,
    require_str,
)
from typing_extensions import ReadOnly, TypedDict

from .codecs import decode_quote, decode_trade
from .types import DOLLAR, Quote, Trade
from .validation import (
    parse_fixed,
    parse_timestamp,
    require_bounded_int,
    require_day,
    require_nonempty,
)


class MarketSpec(TypedDict):
    """Settled contract identity and dates required to request its evidence."""

    ticker: ReadOnly[str]
    series: ReadOnly[str]
    day: ReadOnly[str]
    decision_ms: ReadOnly[int]
    close_ms: ReadOnly[int]
    settled_ms: ReadOnly[int]
    settlement: ReadOnly[int]


def encode_spec(record: MarketSpec) -> JSONObject:
    """Encode one normalized settled contract.

    Args:
        record: Validated market specification.

    Returns:
        JSON object.
    """
    return {
        "ticker": record["ticker"],
        "series": record["series"],
        "day": record["day"],
        "decision_ms": record["decision_ms"],
        "close_ms": record["close_ms"],
        "settled_ms": record["settled_ms"],
        "settlement": record["settlement"],
    }


def decode_spec(document: JSONObject) -> MarketSpec:
    """Validate normalized market identity and chronology.

    Args:
        document: JSON object.

    Returns:
        Immutable contract specification.

    Raises:
        ValueError: MARKET-018 for invalid chronology or identity.
        TypeError: Missing or incorrectly typed fields.
    """
    ticker, series = require_nonempty(document, "ticker"), require_nonempty(document, "series")
    decision = require_int(document, "decision_ms")
    close, settled = require_int(document, "close_ms"), require_int(document, "settled_ms")
    if not ticker.startswith(series + "-") or not 0 <= decision < close <= settled:
        raise ValueError("[MARKET-018] invalid contract identity or chronology")
    return MarketSpec(
        ticker=ticker,
        series=series,
        day=require_day(document, "day"),
        decision_ms=decision,
        close_ms=close,
        settled_ms=settled,
        settlement=require_bounded_int(document, "settlement", 0, DOLLAR),
    )


def decode_market(document: JSONObject, series: str, timezone: str) -> MarketSpec:
    """Decode a settled daily weather contract using its event date.

    Args:
        document: Native market JSON with settlement timestamp.
        series: Expected series identity.
        timezone: IANA timezone for D-1 16:00 placement.

    Returns:
        Validated normalized contract.

    Raises:
        ValueError: MARKET-019 for unsupported result or identity;
            MARKET-032 for payout conflict.
        TypeError: Missing or incorrectly typed required fields.
        ZoneInfoNotFoundError: Unknown timezone.
    """
    event = require_nonempty(document, "event_ticker")
    ticker = require_nonempty(document, "ticker")
    result = require_str(document, "result")
    if (
        not event.startswith(series + "-")
        or not ticker.startswith(event + "-")
        or result not in ("yes", "no", "scalar")
    ):
        raise ValueError(
            "[MARKET-019] expected a settled contract in the requested event and series"
        )
    settlement = parse_fixed(require_str(document, "settlement_value_dollars"), 4)
    if (result == "yes" and settlement != DOLLAR) or (result == "no" and settlement != 0):
        raise ValueError("[MARKET-032] binary result conflicts with its settlement payout")
    code = event[len(series) + 1 :]
    months = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
    if len(code) != 7 or code[2:5] not in months:
        raise ValueError("[MARKET-019] expected YYMMMDD daily event date")
    day = datetime(2000 + int(code[:2]), months.index(code[2:5]) + 1, int(code[5:])).date()
    before = day - timedelta(days=1)
    placed = datetime(before.year, before.month, before.day, 16, tzinfo=ZoneInfo(timezone))
    return decode_spec(
        {
            "ticker": ticker,
            "series": series,
            "day": day.isoformat(),
            "decision_ms": int(placed.timestamp() * 1000),
            "close_ms": parse_timestamp(require_str(document, "close_time")),
            "settled_ms": parse_timestamp(require_str(document, "settlement_ts")),
            "settlement": settlement,
        }
    )


def _price(side: JSONObject, field: str) -> int | None:
    """Read the required nullable dollar string for the selected endpoint.

    Args:
        side: Native candle side object.
        field: Field required by that endpoint's schema.

    Returns:
        Integer currency units or explicit null.

    Raises:
        KeyError: Required field is absent.
        ValueError: Dollar precision or spelling is invalid.
        TypeError: Price is not a string or null.
    """
    value = side[field]
    if value is None:
        return None
    return parse_fixed(require_str(side, field), 4)


def decode_candle(document: JSONObject, historical: bool) -> Quote:
    """Decode the endpoint's current dollar schema without guessing field names.

    Args:
        document: Native candle JSON.
        historical: Historical endpoint uses close; live uses close_dollars.

    Returns:
        Validated timestamped bid and ask.

    Raises:
        ValueError: Invalid price or timestamp.
        TypeError: Missing or incorrectly typed fields.
        KeyError: Missing required close price.
    """
    field = "close" if historical else "close_dollars"
    return decode_quote(
        {
            "timestamp_ms": require_int(document, "end_period_ts") * 1000,
            "bid": _price(require_dict(document, "yes_bid"), field),
            "ask": _price(require_dict(document, "yes_ask"), field),
        }
    )


def decode_public_trade(document: JSONObject, ticker: str) -> Trade:
    """Decode exact fractional volume, aggressor outcome and block-trade status.

    Args:
        document: Native trade JSON.
        ticker: Expected contract identity.

    Returns:
        Validated normalized trade.

    Raises:
        ValueError: MARKET-020 for an unexpected contract or outcome side.
        TypeError: Missing or incorrectly typed fields.
    """
    side = require_str(document, "taker_outcome_side")
    if require_str(document, "ticker") != ticker or side not in ("yes", "no"):
        raise ValueError("[MARKET-020] trade contract or aggressor outcome mismatch")
    return decode_trade(
        {
            "trade_id": require_str(document, "trade_id"),
            "timestamp_ms": parse_timestamp(require_str(document, "created_time")),
            "price": parse_fixed(require_str(document, "yes_price_dollars"), 4),
            "quantity": parse_fixed(require_str(document, "count_fp"), 2),
            "taker_yes": side == "yes",
            "block": require_bool(document, "is_block_trade"),
        }
    )


__all__ = [
    "MarketSpec",
    "decode_candle",
    "decode_market",
    "decode_public_trade",
    "decode_spec",
    "encode_spec",
]
