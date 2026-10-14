"""Native historical selection and candle validation through the actual HTTP stack."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest
from httpx import HTTPStatusError
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_dict,
    require_list,
)

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.archive import ResponseArchive
from covenant_ml.markets.decision_quotes import decision_quote
from covenant_ml.markets.sports_history import event_day, historical_selection

from ._server import ApiServer, api_server
from ._strategy_fixtures import candle_payload, native_market, observation

__all__ = ["api_server"]


@pytest.mark.parametrize("event", ["bad", "KXNHLGAME-26XYZ01DALCHI", "KXNHLGAME-26FEB30DALCHI"])
def test_invalid_event_date(event: str) -> None:
    with pytest.raises(ValueError):
        event_day(event)


def test_paginated_selection_exclusions_and_replay(api_server: ApiServer, tmp_path: Path) -> None:
    late = native_market("26FEB28")
    late["settlement_ts"] = "2026-03-01T01:00:00Z"
    pending = native_market("26JAN03")
    pending["status"] = "closed"
    partial = native_market("26JAN04")
    partial["settlement_value_dollars"] = "0.5000"
    unopened = native_market("26JAN05")
    unopened["open_time"] = "2026-01-05T13:00:00Z"
    api_server.documents["/historical/markets"] = [
        {
            "markets": [native_market(team="DAL"), native_market("25DEC31"), late, pending],
            "cursor": "next",
        },
        {
            "markets": [
                native_market(),
                native_market("26JAN01", team="ZZZ"),
                partial,
                unopened,
                native_market("26MAR01"),
            ],
            "cursor": "",
        },
    ]
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        store = ResponseArchive(
            tmp_path, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        specs, counts = historical_selection(store.capture)
        assert tuple(row["ticker"] for row in specs) == (
            "KXNHLGAME-26JAN01DALCHI-CHI",
            "KXNHLGAME-26MAR01DALCHI-CHI",
        )
        assert counts == {
            "selected_games": 6,
            "nonbinary": 2,
            "not_open": 1,
            "late_training_settlement": 1,
        }
        assert historical_selection(store.replay) == (specs, counts)
    assert len(api_server.requests) == 2


@pytest.mark.parametrize("kind", ["cursor", "duplicate", "identity"])
def test_invalid_inventory(api_server: ApiServer, tmp_path: Path, kind: str) -> None:
    row = native_market()
    if kind == "identity":
        row["ticker"] = "wrong"
    documents: list[JSONObject] = [
        {
            "markets": [row, row] if kind == "duplicate" else [row],
            "cursor": "next" if kind == "cursor" else "",
        }
    ]
    if kind == "cursor":
        documents.append({"markets": [], "cursor": "next"})
    api_server.documents["/historical/markets"] = documents
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        store = ResponseArchive(
            tmp_path, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        with pytest.raises(ValueError, match="STRATEGY-002"):
            historical_selection(store.capture)


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("valid", "eligible"),
        ("empty", "missing_candle"),
        ("null_bid", "unavailable_quote"),
        ("null_ask", "unavailable_quote"),
        ("zero_bid", "unavailable_quote"),
        ("one_ask", "unavailable_quote"),
        ("wide", "wide_spread"),
        ("crossed", "error"),
        ("identity", "error"),
        ("duplicate", "error"),
        ("time", "error"),
    ],
)
def test_candle_validation(api_server: ApiServer, tmp_path: Path, kind: str, expected: str) -> None:
    payload = candle_payload()
    candles = require_list(payload, "candlesticks")
    candle = narrow_json_to_dict(candles[0])
    if kind == "empty":
        candles.clear()
    elif kind == "identity":
        payload["ticker"] = "wrong"
    elif kind == "duplicate":
        candles.append(candle)
    elif kind == "time":
        candle["end_period_ts"] = 0
    else:
        edits: dict[str, tuple[str, JSONValue]] = {
            "null_bid": ("yes_bid", None),
            "null_ask": ("yes_ask", None),
            "zero_bid": ("yes_bid", "0.0000"),
            "one_ask": ("yes_ask", "1.0000"),
            "wide": ("yes_bid", "0.1000"),
            "crossed": ("yes_bid", "0.9000"),
        }
        if kind in edits:
            side, price = edits[kind]
            require_dict(candle, side)["close"] = price
    spec, quote = observation()
    api_server.documents["/historical/markets/" + spec["ticker"] + "/candlesticks"] = [payload]
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        store = ResponseArchive(
            tmp_path, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        if expected == "error":
            with pytest.raises(ValueError, match="STRATEGY-003"):
                decision_quote(spec, store.capture, historical=True)
        else:
            actual = decision_quote(spec, store.capture, historical=True)
            assert actual == (quote if expected == "eligible" else None, expected)
            assert decision_quote(spec, store.replay, historical=True) == actual
    assert len(api_server.requests) == 1


def test_http_failure_propagates(api_server: ApiServer, tmp_path: Path) -> None:
    api_server.documents["/historical/markets"] = [{"error": "unavailable"}]
    api_server.statuses["/historical/markets"] = 503
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        store = ResponseArchive(
            tmp_path, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        with pytest.raises(HTTPStatusError) as failure:
            historical_selection(store.capture)
    assert failure.value.response.status_code == 503
    assert list(tmp_path.glob("*.json")) == []
