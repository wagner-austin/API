"""Small explicit market histories; no model or execution behavior is replaced."""

from __future__ import annotations

from platform_core.json_utils import JSONObject, JSONValue

from covenant_ml.markets.codecs import decode_history, decode_rule
from covenant_ml.markets.types import MakerRule, MarketHistory


def rule(**changes: int) -> MakerRule:
    document: JSONObject = {
        "min_ask": 200,
        "max_ask": 1600,
        "improve": 100,
        "through": 200,
        "contracts": 1,
        "max_quote_age_ms": 3_600_000,
        "latency_ms": 1000,
        "maker_rate_bps": 0,
    }
    document.update(changes)
    return decode_rule(document)


def history_document() -> JSONObject:
    return {
        "ticker": "EXAMPLE-26JAN01-B1",
        "series": "EXAMPLE",
        "event_day": "2026-01-01",
        "decision_ms": 10_000_000,
        "close_ms": 20_000_000,
        "settlement": 0,
        "quotes": [{"timestamp_ms": 10_000_000, "bid": 400, "ask": 1000}],
        "trades": [
            {
                "trade_id": "one",
                "timestamp_ms": 10_002_000,
                "price": 1100,
                "quantity": 100,
                "taker_yes": True,
                "block": False,
            }
        ],
    }


def history(**changes: JSONValue) -> MarketHistory:
    document = history_document()
    document.update(changes)
    return decode_history(document)


__all__ = ["history", "history_document", "rule"]
