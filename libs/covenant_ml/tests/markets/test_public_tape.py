"""Exercise public tape capture through real HTTP, native codecs and archive replay."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest
from httpx import HTTPStatusError
from platform_core.json_utils import JSONObject, JSONValue

from covenant_ml.markets.collection import MarketCollector
from covenant_ml.markets.public_tape import collect_public_tape
from covenant_ml.markets.public_tape_records import decode_tape_trade, require_market_snapshot
from covenant_ml.markets.validation import parse_timestamp

from ._server import ApiServer, api_server
from .test_acquisition import TICKER, archive, market_document, trade_document

__all__ = ["api_server"]

START = parse_timestamp("2026-01-01T21:00:00Z") // 1000
END = START + 60


def tape_trade() -> JSONObject:
    return dict(trade_document(), taker_book_side="ask", no_price_dollars="0.8900")


def tape_market() -> JSONObject:
    return dict(market_document(), status="settled")


def serve_tape(server: ApiServer) -> None:
    server.documents["/historical/cutoff"] = [{"trades_created_ts": "2025-01-01T00:00:00Z"}]
    server.documents["/markets/trades"] = [{"trades": [tape_trade()], "cursor": ""}]
    server.documents["/markets"] = [{"markets": [tape_market()], "cursor": ""}]
    server.documents["/historical/markets"] = [{"markets": [], "cursor": ""}]


def test_capture_and_offline_replay_exact_counts(tmp_path: Path, api_server: ApiServer) -> None:
    serve_tape(api_server)
    second = dict(tape_trade(), trade_id="trade-2", is_block_trade=True, count_fp="0.50")
    api_server.documents["/markets/trades"] = [
        {"trades": [tape_trade()], "cursor": "page2"},
        {"trades": [second], "cursor": ""},
    ]
    store = archive(tmp_path, api_server)
    with closing(store.client):
        report = collect_public_tape(MarketCollector(store.capture), START, END)
        assert collect_public_tape(MarketCollector(store.replay), START, END) == report
    assert report == {
        "schema_version": 1,
        "start_ts_inclusive": START,
        "end_ts_exclusive": END,
        "trade_tiers": ["/markets/trades"],
        "trade_count": 2,
        "contract_count": 1,
        "event_count": 1,
        "contracts_with_declared_result_at_capture": 1,
        "contracts_without_declared_result_at_capture": 0,
        "quantity_centi_contracts": 150,
        "block_trade_count": 1,
        "boundary_rows_excluded": 0,
        "trades": [tape_trade(), second],
        "markets": [tape_market()],
        "orders_submitted": 0,
        "interpretation": (
            "Public executions only; event IDs are not independent observations. "
            "Metadata, results and quotes are capture-time snapshots, not past features. "
            "No order-book history, fees, trader identities, queue position or profitability "
            "is established. Untraded contracts are outside this cohort."
        ),
    }
    assert api_server.requests == [
        "/historical/cutoff?",
        f"/markets/trades?limit=1000&max_ts={END}&min_ts={START - 1}",
        f"/markets/trades?cursor=page2&limit=1000&max_ts={END}&min_ts={START - 1}",
        f"/markets?limit=1000&tickers={TICKER}",
        f"/historical/markets?limit=1000&tickers={TICKER}",
    ]


@pytest.mark.parametrize("cutoff", ["2026-01-02T00:00:00Z", "2026-01-01T21:00:30Z"])
def test_historical_routing_reconciles_overlap(
    tmp_path: Path, api_server: ApiServer, cutoff: str
) -> None:
    serve_tape(api_server)
    api_server.documents["/historical/cutoff"] = [{"trades_created_ts": cutoff}]
    historical_only = dict(tape_trade(), trade_id="historical-only", count_fp="2.00")
    live_only = dict(
        tape_trade(), trade_id="live-only", count_fp="3.00", created_time="2026-01-01T21:00:45Z"
    )
    api_server.documents["/historical/trades"] = [
        {"trades": [historical_only, tape_trade()], "cursor": ""}
    ]
    api_server.documents["/markets/trades"] = [{"trades": [tape_trade(), live_only], "cursor": ""}]
    api_server.documents["/historical/markets"] = [{"markets": [tape_market()], "cursor": ""}]
    store = archive(tmp_path, api_server)
    with closing(store.client):
        report = collect_public_tape(MarketCollector(store.capture), START, END)
    assert report["markets"] == [tape_market()]
    expected = ["/historical/trades"]
    expected_rows = [historical_only, tape_trade()]
    expected_quantity = 300
    if cutoff.endswith("21:00:30Z"):
        expected.append("/markets/trades")
        expected_rows.append(live_only)
        expected_quantity = 600
    assert report["trade_tiers"] == expected
    assert report["trades"] == expected_rows
    assert report["trade_count"] == len(expected_rows)
    assert report["quantity_centi_contracts"] == expected_quantity
    assert api_server.requests == [
        "/historical/cutoff?",
        *(f"{path}?limit=1000&max_ts={END}&min_ts={START - 1}" for path in expected),
        f"/markets?limit=1000&tickers={TICKER}",
        f"/historical/markets?limit=1000&tickers={TICKER}",
    ]


@pytest.mark.parametrize("start,end", [(-1, 1), (2, 2), (3, 2), (0, 86401)])
def test_invalid_windows_make_no_requests(
    tmp_path: Path, api_server: ApiServer, start: int, end: int
) -> None:
    store = archive(tmp_path, api_server)
    with closing(store.client), pytest.raises(ValueError, match="TAPE-001"):
        collect_public_tape(MarketCollector(store.capture), start, end)
    assert api_server.requests == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("created_time", "2026-01-01T20:59:58.999999Z"),
        ("created_time", "2026-01-01T21:01:00.001Z"),
        ("taker_book_side", "unknown"),
        ("no_price_dollars", "0.8800"),
    ],
)
def test_bad_trade_fields_fail_closed(key: str, value: JSONValue) -> None:
    row = tape_trade()
    row[key] = value
    with pytest.raises(ValueError, match="TAPE-002"):
        decode_tape_trade(row, START - 1, END)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "unexpected", "conflict"])
def test_metadata_coverage_is_required(tmp_path: Path, api_server: ApiServer, failure: str) -> None:
    serve_tape(api_server)
    rows: list[JSONValue] = [tape_market()]
    if failure == "missing":
        rows = []
    elif failure == "duplicate":
        rows.append(tape_market())
    elif failure == "unexpected":
        rows = [dict(tape_market(), ticker="WRONG")]
    else:
        api_server.documents["/historical/markets"] = [
            {"markets": [dict(tape_market(), status="closed")], "cursor": ""}
        ]
    api_server.documents["/markets"] = [{"markets": rows, "cursor": ""}]
    store = archive(tmp_path, api_server)
    with (
        closing(store.client),
        pytest.raises(ValueError, match="TAPE-004" if failure == "missing" else "TAPE-003"),
    ):
        collect_public_tape(MarketCollector(store.capture), START, END)
    assert len(api_server.requests) == (3 if failure in ("duplicate", "unexpected") else 4)


@pytest.mark.parametrize(
    "result,payout",
    [("other", "0.0000"), ("yes", "0.5000"), ("no", "0.5000"), ("scalar", "1.0100")],
)
def test_inconsistent_labels_are_rejected(result: str, payout: str) -> None:
    with pytest.raises(ValueError, match="TAPE-005"):
        require_market_snapshot(dict(tape_market(), result=result, settlement_value_dollars=payout))


@pytest.mark.parametrize("result,payout", [("", ""), ("yes", "1.0000"), ("scalar", "0.5000")])
def test_unresolved_and_scalar_contracts_are_retained(
    tmp_path: Path, api_server: ApiServer, result: str, payout: str
) -> None:
    serve_tape(api_server)
    row = dict(tape_market(), result=result, settlement_value_dollars=payout)
    api_server.documents["/markets"] = [{"markets": [row], "cursor": ""}]
    store = archive(tmp_path, api_server)
    with closing(store.client):
        report = collect_public_tape(MarketCollector(store.capture), START, END)
    assert report["markets"] == [row]
    assert report["contracts_with_declared_result_at_capture"] == int(result != "")
    assert report["contracts_without_declared_result_at_capture"] == int(result == "")


def test_empty_window_and_failed_http(tmp_path: Path, api_server: ApiServer) -> None:
    serve_tape(api_server)
    api_server.documents["/markets/trades"] = [{"trades": [], "cursor": ""}]
    store = archive(tmp_path / "empty", api_server)
    with closing(store.client):
        report = collect_public_tape(MarketCollector(store.capture), START, END)
    assert (report["trades"], report["markets"], report["event_count"]) == ([], [], 0)
    serve_tape(api_server)
    api_server.statuses["/markets/trades"] = 503
    failed = archive(tmp_path / "failed", api_server)
    with closing(failed.client), pytest.raises(HTTPStatusError, match="503"):
        collect_public_tape(MarketCollector(failed.capture), START, END)
    assert len(tuple((tmp_path / "failed").glob("*.json"))) == 1


def test_half_open_window_preserves_last_fractional_second(
    tmp_path: Path, api_server: ApiServer
) -> None:
    serve_tape(api_server)
    rows: list[JSONValue] = [
        dict(tape_trade(), trade_id=str(index), created_time=timestamp)
        for index, timestamp in enumerate(
            (
                "2026-01-01T20:59:59.999999Z",
                "2026-01-01T21:00:00Z",
                "2026-01-01T21:00:59.999999Z",
                "2026-01-01T21:01:00Z",
            )
        )
    ]
    api_server.documents["/markets/trades"] = [{"trades": rows, "cursor": ""}]
    store = archive(tmp_path, api_server)
    with closing(store.client):
        report = collect_public_tape(MarketCollector(store.capture), START, END)
    assert report["trades"] == rows[1:3]
    assert report["boundary_rows_excluded"] == 2
    assert report["quantity_centi_contracts"] == 200


def test_metadata_batches_count_contracts_separately_from_events(
    tmp_path: Path, api_server: ApiServer
) -> None:
    serve_tape(api_server)
    rows: list[JSONValue] = []
    markets: list[JSONValue] = []
    for index in range(21):
        ticker = f"KXSERIES-{'E' * 280}-C{index:02d}"
        rows.append(dict(tape_trade(), trade_id=str(index), ticker=ticker))
        markets.append(dict(tape_market(), ticker=ticker, event_ticker="KXSERIES-EVENT"))
    api_server.documents["/markets/trades"] = [{"trades": rows, "cursor": ""}]
    api_server.documents["/markets"] = [
        {"markets": markets[:20], "cursor": ""},
        {"markets": markets[20:], "cursor": ""},
    ]
    api_server.documents["/historical/markets"] = [
        {"markets": [], "cursor": ""},
        {"markets": [], "cursor": ""},
    ]
    store = archive(tmp_path, api_server)
    with closing(store.client):
        report = collect_public_tape(MarketCollector(store.capture), START, END)
    assert report["markets"] == markets
    assert (report["trade_count"], report["contract_count"], report["event_count"]) == (21, 21, 1)
    assert len(api_server.requests) == 6


def test_oversized_identifier_is_not_sent(tmp_path: Path, api_server: ApiServer) -> None:
    serve_tape(api_server)
    api_server.documents["/markets/trades"] = [
        {"trades": [dict(tape_trade(), ticker="X" * 6000)], "cursor": ""}
    ]
    store = archive(tmp_path, api_server)
    with closing(store.client), pytest.raises(ValueError, match="TAPE-007"):
        collect_public_tape(MarketCollector(store.capture), START, END)
    assert len(api_server.requests) == 2
