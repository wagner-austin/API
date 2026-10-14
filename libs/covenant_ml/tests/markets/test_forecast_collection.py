"""Native market/forecast acquisition, exclusions and complete offline CLI reproduction."""

from __future__ import annotations

import runpy
from collections.abc import Callable
from contextlib import closing
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_list,
    require_str,
)
from platform_core.run_record import decode_run_record
from scripts.forecast_research import main

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.archive import ResponseArchive
from covenant_ml.markets.collection import MarketCollector
from covenant_ml.markets.forecast_collection import collect_forecast_rows
from covenant_ml.markets.forecast_records import (
    ComparisonPlan,
    decode_comparison_plan,
    encode_comparison_plan,
)
from covenant_ml.markets.provenance import implementation_digest
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._forecast_fixtures import comparison_plan, serve_comparison
from ._server import ApiServer, api_server

__all__ = ["api_server"]


def _capture(
    server: ApiServer, root: Path, plan: ComparisonPlan, pause: Callable[[float], None]
) -> tuple[tuple[str, ...], dict[str, int]]:
    with closing(_test_hooks.client_factory(10.0, None)) as client:
        market = ResponseArchive(root / "markets", server.url, client, _test_hooks.clock, pause)
        forecast = ResponseArchive(root / "forecasts", server.url, client, _test_hooks.clock, pause)
        rows, counts = collect_forecast_rows(
            plan,
            MarketCollector(market.resume_capture),
            forecast.resume_capture,
            print,
        )
    return tuple(row["ticker"] for row in rows), counts


def test_real_cli_capture_replay_and_hashes(api_server: ApiServer, tmp_path: Path) -> None:
    serve_comparison(api_server)
    tickers, counts = _capture(api_server, tmp_path, comparison_plan(), _test_hooks.pause)
    assert len(tickers) == len(set(tickers)) == 96
    assert counts == {"selected_contracts": 96, "included": 96}
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"frozen":true}', encoding="utf-8")
    plan = encode_comparison_plan(comparison_plan())
    plan["protocol_digest"] = sha256(protocol.read_bytes()).hexdigest()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(dump_json_str(plan), encoding="utf-8")
    argv = [
        "--plan",
        str(plan_path),
        "--protocol",
        str(protocol),
        "--markets-archive",
        str(tmp_path / "markets"),
        "--forecast-archive",
        str(tmp_path / "forecasts"),
        "--forecast-mode",
        "replay",
        "--market-url",
        api_server.url,
        "--forecast-url",
        api_server.url,
    ]
    request_count = len(api_server.requests)
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    assert main([*argv, "--out", str(first)], stand_in_pin) == 0
    assert main([*argv, "--out", str(second)], stand_in_pin) == 0
    assert len(api_server.requests) == request_count
    assert first.read_bytes() == second.read_bytes()
    report = narrow_json_to_dict(load_json_str(first.read_text("utf-8")))
    assert report["implementation_sha256"] == implementation_digest()
    assert len(require_list(report, "training_rows")) == 30
    assert len(require_list(report, "evaluation_rows")) == 60
    record = decode_run_record(
        narrow_json_to_dict(load_json_str(Path(str(first) + ".runrecord.json").read_text("utf-8")))
    )
    assert record["payload_digest"] == sha256(first.read_bytes()).hexdigest()
    assert len(record["observations"]) == 4
    protocol.write_text('{"frozen":false}', encoding="utf-8")
    with pytest.raises(ValueError, match="FORECAST-010"):
        main([*argv, "--out", str(tmp_path / "bad.json")], stand_in_pin)
    assert len(api_server.requests) == request_count


def test_cli_entry_refuses_late_determinism_pin() -> None:
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.forecast_research", run_name="__main__")


def test_required_city_cannot_silently_disappear(api_server: ApiServer, tmp_path: Path) -> None:
    serve_comparison(api_server)
    api_server.documents["/markets"][0]["markets"] = []
    api_server.documents["/historical/markets"][0]["markets"] = []
    with pytest.raises(ValueError, match=r"FORECAST-011.*KXHIGHTBOS"):
        _capture(api_server, tmp_path, comparison_plan(), _test_hooks.pause)
    assert not any("/v1/forecast" in request for request in api_server.requests)


@pytest.mark.parametrize("conflict", ["strike", "settlement", "rules"])
def test_native_cross_tier_conflicts_fail(
    api_server: ApiServer, tmp_path: Path, conflict: str
) -> None:
    serve_comparison(api_server)
    historical = api_server.documents["/historical/markets"][0]
    documents = require_list(historical, "markets")
    original = narrow_json_to_dict(documents[0])
    changed: JSONObject = dict(original)
    if conflict == "strike":
        changed["cap_strike"] = 99
    elif conflict == "settlement":
        changed["settlement_ts"] = "2026-04-04T13:00:00Z"
    else:
        changed["rules_primary"] = "Different settlement source"
    replacement = list(documents)
    replacement[0] = changed
    historical["markets"] = replacement
    with pytest.raises(ValueError, match="FORECAST-005"):
        _capture(api_server, tmp_path, comparison_plan(), _test_hooks.pause)


def test_collection_accounts_for_all_exclusions(api_server: ApiServer, tmp_path: Path) -> None:
    serve_comparison(api_server)
    documents = require_list(api_server.documents["/markets"][0], "markets")
    scalar = narrow_json_to_dict(documents[0])
    scalar["result"], scalar["settlement_value_dollars"] = "scalar", "0.0100"
    narrow_json_to_dict(documents[2])["strike_type"] = None
    stale = narrow_json_to_dict(documents[4])
    ticker = require_str(stale, "ticker")
    candle = api_server.documents["/historical/markets/" + ticker + "/candlesticks"][0]
    candle["candlesticks"] = []
    # An out-of-window contract must not consume a quote or forecast request.
    extra: JSONObject = dict(narrow_json_to_dict(documents[-1]))
    extra["ticker"] = "KXHIGHTBOS-26APR19-B60.5"
    extra["event_ticker"] = "KXHIGHTBOS-26APR19"
    extra["close_time"] = "2026-04-20T04:00:00Z"
    extra["settlement_ts"] = "2026-04-20T12:00:00Z"
    documents.append(extra)
    # One-sided native contracts omit their irrelevant strike bound.
    greater = narrow_json_to_dict(documents[6])
    greater["strike_type"] = "greater"
    del greater["cap_strike"]
    less = narrow_json_to_dict(documents[7])
    less["strike_type"] = "less"
    del less["floor_strike"]
    plan_document = encode_comparison_plan(comparison_plan())
    plan_document["excluded_days"] = ["2026-04-18"]
    plan_document["forecast_hour"] = 12
    # The last forecast for each city is never requested under this protocol.
    api_server.documents["/v1/forecast"] = [
        document
        for index, document in enumerate(api_server.documents["/v1/forecast"])
        if index % 16 != 15
    ]
    pauses: list[float] = []
    tickers, counts = _capture(
        api_server, tmp_path, decode_comparison_plan(plan_document), pauses.append
    )
    assert pauses == [0.3] * len(api_server.requests)
    assert len(tickers) == 87
    assert all("26APR18" not in ticker for ticker in tickers)
    forecast_requests = [url for url in api_server.requests if "/v1/forecast?" in url]
    assert len(forecast_requests) == 45
    assert all("T12%3A00" in url for url in forecast_requests)
    assert counts == {
        "selected_contracts": 96,
        "nonbinary_settlement": 1,
        "unsupported_strike": 1,
        "no_quote": 1,
        "included": 87,
        "protocol_excluded_day": 6,
    }
