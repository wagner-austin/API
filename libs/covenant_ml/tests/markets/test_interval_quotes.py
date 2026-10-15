"""Real transport and temporal falsifiers for complete crypto quote attachments."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest
from httpx import HTTPStatusError
from platform_core.json_utils import JSONObject, JSONValue, narrow_json_to_dict, require_list

from covenant_ml.markets.collection import MarketCollector
from covenant_ml.markets.interval_quotes import (
    _current_batch,
    collect_interval_quotes,
    require_interval_plan,
)
from covenant_ml.markets.validation import parse_timestamp

from ._server import ApiServer, api_server
from .test_acquisition import archive

__all__ = ["api_server"]

START = parse_timestamp("2026-01-01T00:00:00Z") // 1000
BTC = "KXBTC15M-EVENT-A"
ETH = "KXETH15M-EVENT-A"


def plan_document() -> JSONObject:
    return {
        "start_ts": START,
        "end_ts": START + 86400,
        "decision_offset_seconds": 300,
        "min_quote_age_seconds": 1,
        "max_quote_age_seconds": 120,
    }


def market_document(series: str) -> JSONObject:
    return {
        "ticker": series + "-EVENT-A",
        "event_ticker": series + "-EVENT",
        "open_time": "2026-01-01T00:00:00Z",
        "close_time": "2026-01-01T00:15:00Z",
        "settlement_ts": "2026-01-01T00:16:00Z",
        "result": "yes",
        "volume_fp": "0.00",
    }


def cohort_document() -> JSONObject:
    return {
        "schema_version": 1,
        "selection_clock": "capture_time_close_time",
        "start_ts_inclusive": START,
        "end_ts_exclusive": START + 86400,
        "families": [
            {"series": "KXBTC15M", "markets": [market_document("KXBTC15M")]},
            {"series": "KXETH15M", "markets": [market_document("KXETH15M")]},
            {"series": "KXNFLGAME", "markets": []},
        ],
    }


def candle(
    seconds: int = 240,
    bid: str | None = "0.4000",
    ask: str | None = "0.6000",
    historical: bool = False,
) -> JSONObject:
    field = "close" if historical else "close_dollars"
    return {"end_period_ts": START + seconds, "yes_bid": {field: bid}, "yes_ask": {field: ask}}


def batch_document(candles: list[JSONValue]) -> JSONObject:
    return {
        "markets": [
            {"market_ticker": BTC, "candlesticks": candles},
            {"market_ticker": ETH, "candlesticks": []},
        ]
    }


def serve_quotes(server: ApiServer, response: JSONObject) -> None:
    server.documents["/historical/cutoff"] = [{"market_settled_ts": "2025-12-01T00:00:00Z"}]
    server.documents["/markets/candlesticks"] = [response]


def test_real_capture_replay_and_full_membership(tmp_path: Path, api_server: ApiServer) -> None:
    serve_quotes(api_server, batch_document([candle(0), candle(300), candle()]))
    store = archive(tmp_path, api_server)
    with closing(store.client):
        result = collect_interval_quotes(
            MarketCollector(store.capture), cohort_document(), plan_document()
        )
        assert (
            collect_interval_quotes(
                MarketCollector(store.replay), cohort_document(), plan_document()
            )
            == result
        )
    assert result["contract_count"] == 2
    assert result["status_counts"] == {"no_prior_candle": 1, "two_sided": 1}
    assert result["contracts"] == [
        {
            "ticker": BTC,
            "event_ticker": "KXBTC15M-EVENT",
            "series": "KXBTC15M",
            "open_ms": START * 1000,
            "decision_ms": (START + 300) * 1000,
            "tier": "current",
            "eligible_candle_count": 1,
            "quote": {"timestamp_ms": (START + 240) * 1000, "bid": 4000, "ask": 6000},
            "quote_age_ms": 60000,
            "status": "two_sided",
        },
        {
            "ticker": ETH,
            "event_ticker": "KXETH15M-EVENT",
            "series": "KXETH15M",
            "open_ms": START * 1000,
            "decision_ms": (START + 300) * 1000,
            "tier": "current",
            "eligible_candle_count": 0,
            "quote": None,
            "quote_age_ms": None,
            "status": "no_prior_candle",
        },
    ]
    assert api_server.requests == [
        "/historical/cutoff?",
        f"/markets/candlesticks?end_ts={START + 300}&market_tickers={BTC}%2C{ETH}"
        f"&period_interval=1&start_ts={START}",
    ]


def test_future_outcomes_volume_and_prices_cannot_change_features(
    tmp_path: Path, api_server: ApiServer
) -> None:
    cohort = cohort_document()
    serve_quotes(api_server, batch_document([candle(), candle(300), candle(600)]))
    first = archive(tmp_path / "first", api_server)
    with closing(first.client):
        original = collect_interval_quotes(MarketCollector(first.capture), cohort, plan_document())
    for value in require_list(cohort, "families"):
        for raw in require_list(narrow_json_to_dict(value), "markets"):
            market = narrow_json_to_dict(raw)
            market.update(result="no", volume_fp="999999.00")
    serve_quotes(
        api_server,
        batch_document([candle(600, "0.0100", "0.9900"), candle(300, None, None), candle()]),
    )
    second = archive(tmp_path / "second", api_server)
    with closing(second.client):
        changed = collect_interval_quotes(MarketCollector(second.capture), cohort, plan_document())
    assert changed == original


@pytest.mark.parametrize(
    "latest,status",
    [
        (candle(120), "stale"),
        (candle(180), "two_sided"),
        (candle(240, None, "0.6000"), "missing_side"),
        (candle(240, "0.4000", None), "missing_side"),
        (candle(240, "0.7000", "0.6000"), "crossed"),
        (candle(240, "0.5000", "0.5000"), "locked"),
    ],
)
def test_latest_invalid_quote_is_not_replaced_by_an_older_one(
    tmp_path: Path, api_server: ApiServer, latest: JSONObject, status: str
) -> None:
    serve_quotes(api_server, batch_document([candle(60), latest]))
    store = archive(tmp_path, api_server)
    with closing(store.client):
        report = collect_interval_quotes(
            MarketCollector(store.capture), cohort_document(), plan_document()
        )
    assert report["status_counts"] == {status: 1, "no_prior_candle": 1}
    first = narrow_json_to_dict(require_list(report, "contracts")[0])
    assert first["eligible_candle_count"] == 2
    assert first["status"] == status


def test_historical_boundary_and_omitted_current_contract(
    tmp_path: Path, api_server: ApiServer
) -> None:
    cohort = cohort_document()
    family = narrow_json_to_dict(require_list(cohort, "families")[0])
    market = narrow_json_to_dict(require_list(family, "markets")[0])
    market["settlement_ts"] = "2026-01-01T00:15:59.999Z"
    api_server.documents["/historical/cutoff"] = [{"market_settled_ts": "2026-01-01T00:16:00Z"}]
    path = "/historical/markets/" + BTC + "/candlesticks"
    api_server.documents[path] = [{"ticker": BTC, "candlesticks": [candle(historical=True)]}]
    api_server.documents["/markets/candlesticks"] = [{"markets": []}]
    store = archive(tmp_path, api_server)
    with closing(store.client):
        report = collect_interval_quotes(MarketCollector(store.capture), cohort, plan_document())
        assert (
            collect_interval_quotes(MarketCollector(store.replay), cohort, plan_document())
            == report
        )
    assert report["status_counts"] == {"two_sided": 1, "omitted_by_endpoint": 1}
    assert [narrow_json_to_dict(row)["tier"] for row in require_list(report, "contracts")] == [
        "historical",
        "current",
    ]
    assert (
        api_server.requests[1] == f"{path}?end_ts={START + 300}&period_interval=1&start_ts={START}"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"start_ts": START - 1},
        {"end_ts": START + 86401},
        {"min_quote_age_seconds": 0},
        {"max_quote_age_seconds": 301},
        {"decision_offset_seconds": 900},
        {"max_quote_age_seconds": 0},
    ],
)
def test_invalid_plan(change: JSONObject) -> None:
    with pytest.raises(ValueError, match="IQUOTE-001"):
        require_interval_plan(dict(plan_document(), **change), cohort_document())


@pytest.mark.parametrize("change", [{"schema_version": 2}, {"selection_clock": "future"}])
def test_invalid_cohort_schema(change: JSONObject) -> None:
    with pytest.raises(ValueError, match="IQUOTE-001"):
        require_interval_plan(plan_document(), dict(cohort_document(), **change))


@pytest.mark.parametrize(
    "field,value",
    [
        ("open_time", "2026-01-01T00:00:01Z"),
        ("close_time", "2026-01-01T00:14:00Z"),
        ("event_ticker", "OTHER"),
        ("ticker", "OTHER"),
    ],
)
def test_invalid_contract(tmp_path: Path, api_server: ApiServer, field: str, value: str) -> None:
    cohort = cohort_document()
    family = narrow_json_to_dict(require_list(cohort, "families")[0])
    narrow_json_to_dict(require_list(family, "markets")[0])[field] = value
    store = archive(tmp_path, api_server)
    with closing(store.client), pytest.raises(ValueError, match="IQUOTE-002"):
        collect_interval_quotes(MarketCollector(store.capture), cohort, plan_document())
    assert api_server.requests == []


@pytest.mark.parametrize("kind", ["family", "missing", "ticker", "slot"])
def test_duplicate_or_missing_identities(tmp_path: Path, api_server: ApiServer, kind: str) -> None:
    cohort = cohort_document()
    families = require_list(cohort, "families")
    family = narrow_json_to_dict(families[0])
    if kind == "family":
        families.append(family)
    elif kind == "missing":
        families.pop(1)
    else:
        duplicate = market_document("KXBTC15M")
        if kind == "slot":
            duplicate["ticker"] = "KXBTC15M-EVENT-B"
        require_list(family, "markets").append(duplicate)
    store = archive(tmp_path, api_server)
    with closing(store.client), pytest.raises(ValueError, match="IQUOTE-002"):
        collect_interval_quotes(MarketCollector(store.capture), cohort, plan_document())
    assert api_server.requests == []


def test_empty_families_and_window_exclusions(tmp_path: Path, api_server: ApiServer) -> None:
    cohort = cohort_document()
    for value in require_list(cohort, "families"):
        family = narrow_json_to_dict(value)
        for raw in require_list(family, "markets"):
            narrow_json_to_dict(raw)["close_time"] = "2026-01-02T00:00:00Z"
    serve_quotes(api_server, {"markets": []})
    store = archive(tmp_path, api_server)
    with closing(store.client):
        report = collect_interval_quotes(MarketCollector(store.capture), cohort, plan_document())
    assert (report["contract_count"], report["contracts"], report["status_counts"]) == (0, [], {})
    assert api_server.requests == ["/historical/cutoff?"]


@pytest.mark.parametrize(
    "response,code",
    [
        (batch_document([candle(), candle()]), "IQUOTE-003"),
        (batch_document([candle(241)]), "IQUOTE-003"),
        ({"markets": [{"market_ticker": "OTHER", "candlesticks": []}]}, "IQUOTE-004"),
        (
            {
                "markets": [
                    {"market_ticker": BTC, "candlesticks": []},
                    {"market_ticker": BTC, "candlesticks": []},
                ]
            },
            "IQUOTE-004",
        ),
    ],
)
def test_native_response_integrity(
    tmp_path: Path, api_server: ApiServer, response: JSONObject, code: str
) -> None:
    serve_quotes(api_server, response)
    store = archive(tmp_path, api_server)
    with closing(store.client), pytest.raises(ValueError, match=code):
        collect_interval_quotes(MarketCollector(store.capture), cohort_document(), plan_document())


@pytest.mark.parametrize("mismatch", [False, True])
def test_historical_http_failure_and_identity(
    tmp_path: Path, api_server: ApiServer, mismatch: bool
) -> None:
    api_server.documents["/historical/cutoff"] = [{"market_settled_ts": "2026-02-01T00:00:00Z"}]
    if mismatch:
        api_server.documents["/historical/markets/" + BTC + "/candlesticks"] = [
            {"ticker": "OTHER", "candlesticks": []}
        ]
    store = archive(tmp_path, api_server)
    with closing(store.client):
        if mismatch:
            with pytest.raises(ValueError, match="IQUOTE-005"):
                collect_interval_quotes(
                    MarketCollector(store.capture), cohort_document(), plan_document()
                )
        else:
            with pytest.raises(HTTPStatusError, match="404"):
                collect_interval_quotes(
                    MarketCollector(store.capture), cohort_document(), plan_document()
                )


@pytest.mark.parametrize("count,decision", [(101, 300), (2, 300000)])
def test_batch_capacity_refuses_before_transport(
    tmp_path: Path, api_server: ApiServer, count: int, decision: int
) -> None:
    markets: list[JSONObject] = [
        {"open_ms": START * 1000, "decision_ms": (START + decision) * 1000}
    ] * count
    store = archive(tmp_path, api_server)
    with closing(store.client), pytest.raises(ValueError, match="IQUOTE-004"):
        _current_batch(MarketCollector(store.capture), markets, plan_document())
    assert api_server.requests == []
