"""Real HTTP, disk integrity, native schema and pagination integration tests."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from time import perf_counter

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_str,
)

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.archive import ResponseArchive, decode_capture, encode_capture
from covenant_ml.markets.collection import MarketCollector
from covenant_ml.markets.kalshi import (
    decode_candle,
    decode_market,
    decode_public_trade,
    decode_spec,
    encode_spec,
)
from covenant_ml.markets.replay import replay_maker
from covenant_ml.markets.replay_codecs import decode_result, encode_result
from covenant_ml.markets.types import ReplayStatus
from covenant_ml.markets.validation import parse_timestamp

from ._fixtures import rule
from ._server import ApiServer, api_server

__all__ = ["api_server"]

SERIES = "KXHIGHTBOS"
TICKER = SERIES + "-26JAN02-T40"


def market_document() -> JSONObject:
    return {
        "ticker": TICKER,
        "event_ticker": SERIES + "-26JAN02",
        "result": "no",
        "settlement_value_dollars": "0.0000",
        "close_time": "2026-01-03T05:00:00Z",
        "settlement_ts": "2026-01-03T11:00:00Z",
    }


def trade_document() -> JSONObject:
    return {
        "ticker": TICKER,
        "trade_id": "trade-1",
        "created_time": "2026-01-01T21:00:02Z",
        "yes_price_dollars": "0.1100",
        "count_fp": "1.00",
        "taker_outcome_side": "yes",
        "is_block_trade": False,
    }


def candle_document() -> JSONObject:
    return {
        "end_period_ts": 1767301200,
        "yes_bid": {"close_dollars": "0.0400"},
        "yes_ask": {"close_dollars": "0.1000"},
    }


def require_test_pause(seconds: float) -> None:
    """Validate the transport's pacing request without delaying data assertions.

    Args:
        seconds: Delay requested by the actual archive implementation.

    Raises:
        AssertionError: The archive changed its required request pacing.
    """
    assert seconds == 0.3, "archive request pacing changed"


def archive(
    root: Path, server: ApiServer, pause: Callable[[float], None] = require_test_pause
) -> ResponseArchive:
    """Bind real HTTP and disk with explicit pacing for the tested concern.

    The elapsed-time test passes the production sleep implementation. Other
    callers validate the same requested delay without repeating wall-clock
    waits while testing native data, selection, replay or transport failures.

    Args:
        root: Immutable response archive directory.
        server: Actual local HTTP origin.
        pause: Delay contract validator or the production sleeper.

    Returns:
        Production archive with its real client and timestamp reader.
    """
    return ResponseArchive(
        root,
        server.url,
        _test_hooks.client_factory(5.0, None),
        _test_hooks.clock,
        pause,
    )


def test_transport_fixture_refuses_changed_request_pacing() -> None:
    with pytest.raises(AssertionError, match="archive request pacing changed"):
        require_test_pause(0.0)


def test_real_http_archive_replays_after_server_payload_is_consumed(
    tmp_path: Path,
    api_server: ApiServer,
) -> None:
    api_server.documents["/evidence"] = [{"answer": 42}]
    store = archive(tmp_path, api_server, _test_hooks.pause)
    started = perf_counter()
    assert store.capture("/evidence", {"z": 2, "a": "yes"}) == {"answer": 42}
    assert perf_counter() - started >= 0.3
    assert store.replay("/evidence", {"a": "yes", "z": 2}) == {"answer": 42}
    assert api_server.requests == ["/evidence?a=yes&z=2"]
    with pytest.raises(FileExistsError, match="MARKET-016"):
        store.capture("/evidence", {"z": 2, "a": "yes"})
    target = store.file_for(store.request_url("/evidence", {"z": 2, "a": "yes"}))
    document = narrow_json_to_dict(load_json_str(target.read_text("utf-8")))
    capture = decode_capture(document)
    assert decode_capture(encode_capture(capture)) == capture
    assert parse_timestamp(capture["captured_at"]) > 0
    document["body"] = "{}"
    with pytest.raises(ValueError, match="MARKET-015"):
        decode_capture(document)
    document = encode_capture(capture)
    document["url"] = "https://different.test"
    target.write_text(dump_json_str(document), "utf-8")
    with pytest.raises(ValueError, match="MARKET-017"):
        store.replay("/evidence", {"z": 2, "a": "yes"})
    store.client.close()


@pytest.mark.parametrize("conflict", [False, True])
def test_trade_identity_conflicts_never_increase_fill_volume(
    tmp_path: Path,
    api_server: ApiServer,
    conflict: bool,
) -> None:
    path = "/series/" + SERIES + "/markets/" + TICKER + "/candlesticks"
    api_server.documents[path] = [{"ticker": TICKER, "candlesticks": [candle_document()]}]
    repeated = trade_document()
    if conflict:
        repeated["yes_price_dollars"] = "0.1200"
        api_server.documents["/historical/trades"] = [{"trades": [trade_document()], "cursor": ""}]
        api_server.documents["/markets/trades"] = [{"trades": [repeated], "cursor": ""}]
    else:
        api_server.documents["/historical/trades"] = [
            {"trades": [trade_document(), repeated], "cursor": ""}
        ]
    store = archive(tmp_path, api_server)
    collector = MarketCollector(store.capture)
    with pytest.raises(ValueError, match="MARKET-030" if conflict else "MARKET-029"):
        collector.history(
            decode_market(market_document(), SERIES, "America/New_York"),
            {
                "market_settled_ts": "2025-01-01T00:00:00Z",
                "trades_created_ts": "2026-01-02T00:00:00Z",
            },
            rule(),
        )
    assert len(api_server.requests) == (3 if conflict else 2)
    store.client.close()


def test_http_failure_and_missing_capture_propagate(tmp_path: Path, api_server: ApiServer) -> None:
    store = archive(tmp_path, api_server)
    # The dynamic library boundary supplies a typed exception class, not a mocked response.
    module = __import__("httpx")
    error: type[Exception] = module.HTTPStatusError
    with pytest.raises(error, match="404"):
        store.capture("/absent", {})
    with pytest.raises(FileNotFoundError):
        store.replay("/absent", {})
    assert list(tmp_path.iterdir()) == []
    store.client.close()


@pytest.mark.parametrize(
    "cutoff", ["2025-12-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-02-01T00:00:00Z"]
)
def test_http_collection_to_pnl_routes_both_cutoffs_and_all_trade_pages(
    tmp_path: Path,
    api_server: ApiServer,
    cutoff: str,
) -> None:
    candle_path = "/historical" if cutoff.startswith("2026-02") else "/series/" + SERIES
    candle_path += "/markets/" + TICKER + "/candlesticks"
    candle = candle_document()
    if cutoff.startswith("2026-02"):
        candle["yes_bid"] = {"close": "0.0400"}
        candle["yes_ask"] = {"close": "0.1000"}
    api_server.documents[candle_path] = [{"ticker": TICKER, "candlesticks": [candle]}]
    api_server.documents["/historical/trades"] = [{"trades": [trade_document()], "cursor": ""}]
    api_server.documents["/markets/trades"] = [
        {"trades": [trade_document()], "cursor": "next"},
        {"trades": [], "cursor": ""},
    ]
    if cutoff.startswith("2026-02"):
        api_server.documents["/historical/trades"] = api_server.documents["/markets/trades"]
    store = archive(tmp_path, api_server)
    collector = MarketCollector(store.capture)
    spec = decode_market(market_document(), SERIES, "America/New_York")
    assert decode_spec(encode_spec(spec)) == spec
    cutoffs: JSONObject = {"market_settled_ts": cutoff, "trades_created_ts": cutoff}
    evidence = collector.history(spec, cutoffs, rule())
    assert evidence["decision_ms"] == 1767301200000
    result = replay_maker(evidence, rule())
    assert (result["status"], result["pnl"], result["matching_quantity"]) == (
        ReplayStatus.FILLED,
        900,
        100,
    )
    assert MarketCollector(store.replay).history(spec, cutoffs, rule()) == evidence
    assert len(api_server.requests) == (4 if cutoff.startswith("2026-01") else 3)
    store.client.close()


def test_inventory_pagination_fee_validation_and_duplicate_refusal(
    tmp_path: Path,
    api_server: ApiServer,
) -> None:
    api_server.documents["/markets"] = [
        {"markets": [market_document()], "cursor": "n"},
        {"markets": [], "cursor": ""},
    ]
    api_server.documents["/historical/markets"] = [{"markets": [market_document()], "cursor": ""}]
    api_server.documents["/series/" + SERIES] = [
        {"series": {"ticker": SERIES, "fee_type": "quadratic", "fee_multiplier": 1}}
    ]
    store = archive(tmp_path, api_server)
    collector = MarketCollector(store.capture)
    collector.verify_fee(SERIES)
    records = collector.inventory(SERIES, "America/New_York", date(2026, 1, 1), date(2026, 1, 3))
    assert [r["ticker"] for r in records] == [TICKER]
    offline = MarketCollector(store.replay)
    with pytest.raises(ValueError, match=r"MARKET-022.*no contracts"):
        offline.inventory(SERIES, "America/New_York", date(2025, 1, 1), date(2025, 2, 1))
    store.client.close()


@pytest.mark.parametrize(
    "key,value",
    [("result", ""), ("event_ticker", "WRONG-26JAN02"), ("event_ticker", SERIES + "-26BAD02")],
)
def test_native_market_schema_refuses_unsupported_identity(key: str, value: JSONValue) -> None:
    document = market_document()
    document[key] = value
    document["ticker"] = require_str(document, "event_ticker") + "-T40"
    with pytest.raises(ValueError, match="MARKET-019"):
        decode_market(document, SERIES, "America/New_York")


def test_market_chronology_and_native_side_validation() -> None:
    spec = encode_spec(decode_market(market_document(), SERIES, "America/New_York"))
    spec["close_ms"] = 0
    with pytest.raises(ValueError, match="MARKET-018"):
        decode_spec(spec)
    trade = trade_document()
    trade["taker_outcome_side"] = "unknown"
    with pytest.raises(ValueError, match="MARKET-020"):
        decode_public_trade(trade, TICKER)
    candle = candle_document()
    candle["yes_bid"] = {"close_dollars": None}
    assert decode_candle(candle, False)["bid"] is None


@pytest.mark.parametrize(
    "code,utc",
    [
        ("26MAR08", "2026-03-07T21:00:00+00:00"),
        ("26MAR09", "2026-03-08T20:00:00+00:00"),
        ("26NOV01", "2026-10-31T20:00:00+00:00"),
        ("26NOV02", "2026-11-01T21:00:00+00:00"),
    ],
)
def test_d_minus_one_placement_respects_both_dst_transitions(code: str, utc: str) -> None:
    document = market_document()
    event = SERIES + "-" + code
    document["event_ticker"] = event
    document["ticker"] = event + "-T40"
    document["close_time"] = (datetime.fromisoformat(utc) + timedelta(days=2)).isoformat()
    document["settlement_ts"] = document["close_time"]
    assert decode_market(document, SERIES, "America/New_York")["decision_ms"] == parse_timestamp(
        utc
    )


def test_market_cannot_borrow_another_events_placement_date() -> None:
    document = market_document()
    document["event_ticker"] = SERIES + "-26JAN03"
    with pytest.raises(ValueError, match="MARKET-019"):
        decode_market(document, SERIES, "America/New_York")


@pytest.mark.parametrize(
    "payout,expected", [("0.0000", 900), ("0.0100", 800), ("0.9900", -9000), ("1.0000", -9100)]
)
def test_scalar_settlement_preserves_the_actual_exchange_cash_payout(
    payout: str, expected: int
) -> None:
    document = market_document()
    document["result"] = "scalar"
    document["settlement_value_dollars"] = payout
    spec = decode_market(document, SERIES, "America/New_York")
    from ._fixtures import history

    result = replay_maker(history(settlement=spec["settlement"]), rule())
    assert result["pnl"] == expected
    assert decode_result(encode_result(result)) == result
    assert decode_spec(encode_spec(spec))["settlement"] == result["settlement"]


@pytest.mark.parametrize("result,payout", [("yes", "0.9900"), ("no", "0.0100")])
def test_binary_result_cannot_disagree_with_declared_cash_payout(result: str, payout: str) -> None:
    document = market_document()
    document["result"] = result
    document["settlement_value_dollars"] = payout
    with pytest.raises(ValueError, match="MARKET-032"):
        decode_market(document, SERIES, "America/New_York")


@pytest.mark.parametrize("failure", ["cursor", "duplicate", "fee", "candle"])
def test_collector_refusals_use_real_http(
    tmp_path: Path, api_server: ApiServer, failure: str
) -> None:
    api_server.documents["/markets"] = [{"markets": [market_document()], "cursor": ""}]
    conflicting = market_document()
    conflicting["result"] = "yes"
    conflicting["settlement_value_dollars"] = "1.0000"
    api_server.documents["/historical/markets"] = [{"markets": [conflicting], "cursor": ""}]
    api_server.documents["/series/" + SERIES] = [
        {"series": {"ticker": SERIES, "fee_type": "linear"}}
    ]
    path = "/series/" + SERIES + "/markets/" + TICKER + "/candlesticks"
    api_server.documents[path] = [{"ticker": "wrong"}]
    api_server.documents["/cycle"] = [
        {"rows": [], "cursor": "same"},
        {"rows": [], "cursor": "same"},
    ]
    store = archive(tmp_path, api_server)
    collector = MarketCollector(store.capture)
    with pytest.raises(ValueError, match="MARKET-02"):
        if failure == "cursor":
            list(collector.pages("/cycle", {}, "rows"))
        elif failure == "duplicate":
            collector.inventory(SERIES, "America/New_York", date(2026, 1, 1), date(2026, 1, 3))
        elif failure == "fee":
            collector.verify_fee(SERIES)
        else:
            collector.history(
                decode_market(market_document(), SERIES, "America/New_York"),
                {
                    "market_settled_ts": "2025-01-01T00:00:00Z",
                    "trades_created_ts": "2025-01-01T00:00:00Z",
                },
                rule(),
            )
    assert len(api_server.requests) == (2 if failure in ("cursor", "duplicate") else 1)
    store.client.close()
