"""Native historical payloads and validated observations for real strategy tests."""

from __future__ import annotations

from platform_core.json_utils import JSONObject

from covenant_ml.markets.kalshi import MarketSpec, decode_spec
from covenant_ml.markets.sports_history import event_day
from covenant_ml.markets.types import Quote
from covenant_ml.markets.validation import parse_timestamp


def native_market(stamp: str = "26JAN01", team: str = "CHI") -> JSONObject:
    event = "KXNHLGAME-" + stamp + "DALCHI"
    day = event_day(event)
    return {
        "ticker": event + "-" + team,
        "event_ticker": event,
        "status": "finalized",
        "open_time": day + "T00:00:00Z",
        "close_time": day + "T22:00:00Z",
        "settlement_ts": day + "T23:00:00Z",
        "settlement_value_dollars": "1.0000",
    }


def observation(
    stamp: str = "26JAN01", payout: int = 10000, bid: int = 6900, ask: int = 7100
) -> tuple[MarketSpec, Quote]:
    event = "KXNHLGAME-" + stamp + "DALCHI"
    day = event_day(event)
    cutoff = parse_timestamp(day + "T12:00:00Z")
    spec = decode_spec(
        {
            "ticker": event + "-CHI",
            "series": "KXNHLGAME",
            "day": day,
            "decision_ms": cutoff,
            "close_ms": cutoff + 3600000,
            "settled_ms": cutoff + 7200000,
            "settlement": payout,
        }
    )
    return spec, Quote(timestamp_ms=cutoff, bid=bid, ask=ask)


def candle_payload(stamp: str = "26JAN01") -> JSONObject:
    spec, _ = observation(stamp)
    return {
        "ticker": spec["ticker"],
        "candlesticks": [
            {
                "end_period_ts": spec["decision_ms"] // 1000,
                "yes_bid": {"close": "0.6900"},
                "yes_ask": {"close": "0.7100"},
            }
        ],
    }
