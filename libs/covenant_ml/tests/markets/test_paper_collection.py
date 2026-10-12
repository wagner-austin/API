"""Real public-client transport, native contract validation and unavailable-model gates."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import httpx
import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_dict,
    require_list,
)

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.archive import ResponseArchive
from covenant_ml.markets.paper_collection import PaperReader, collect_paper_predictions
from covenant_ml.markets.paper_records import PaperPlan

from ._paper_fixtures import fitted_report, paper_plan, serve_paper
from ._server import ApiServer, api_server

__all__ = ["api_server"]


def collect(
    server: ApiServer,
    root: Path,
    plan: PaperPlan,
    report: JSONObject,
    market_clock: Callable[[], str] = _test_hooks.clock,
    forecast_clock: Callable[[], str] = _test_hooks.clock,
) -> JSONObject:
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        market = ResponseArchive(
            root / "markets", server.url, client, market_clock, _test_hooks.pause
        )
        forecast = ResponseArchive(
            root / "forecasts", server.url, client, forecast_clock, _test_hooks.pause
        )
        return collect_paper_predictions(
            plan, report, PaperReader(market, plan, True), PaperReader(forecast, plan, True)
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("event_ticker", "OTHER"),
        ("ticker", "OTHER-STRIKE"),
        ("status", "settled"),
        ("result", "yes"),
        ("open_time", "2100-01-01T00:00:00Z"),
        ("close_time", "2020-01-01T00:00:00Z"),
    ],
)
def test_invalid_live_contract(
    api_server: ApiServer, tmp_path: Path, key: str, value: JSONValue
) -> None:
    plan = paper_plan()
    serve_paper(api_server, plan)
    native = narrow_json_to_dict(require_list(api_server.documents["/markets"][0], "markets")[0])
    native[key] = value
    with pytest.raises(ValueError, match="PAPER-006"):
        collect(api_server, tmp_path, plan, fitted_report())
    assert not any("orderbook" in path for path in api_server.requests)


@pytest.mark.parametrize("duplicate", [False, True])
def test_invalid_inventory(api_server: ApiServer, tmp_path: Path, duplicate: bool) -> None:
    plan = paper_plan()
    serve_paper(api_server, plan)
    documents = require_list(api_server.documents["/markets"][0], "markets")
    if duplicate:
        documents.append(documents[0])
    else:
        documents.clear()
    with pytest.raises(ValueError, match="PAPER-006"):
        collect(api_server, tmp_path, plan, fitted_report())
    assert len(api_server.requests) == 2


@pytest.mark.parametrize(
    "mutation", ["cycle", "future", "empty", "unsettled", "baseline", "augmented"]
)
def test_models_require_available_training(
    api_server: ApiServer, tmp_path: Path, mutation: str
) -> None:
    plan, report = paper_plan(), fitted_report()
    if mutation == "cycle":
        require_dict(report, "plan")["forecast_hour"] = 12
    elif mutation == "future":
        require_dict(report, "plan")["end"] = plan["event_day"]
    elif mutation == "empty":
        report["training_rows"] = []
    elif mutation == "unsettled":
        row = narrow_json_to_dict(require_list(report, "training_rows")[0])
        row["settled_ms"] = 2**60
    else:
        models = require_dict(report, "models")
        destination, source = (
            ("market_calibrated", "market_plus_forecast")
            if mutation == "baseline"
            else ("market_plus_forecast", "market_calibrated")
        )
        models[destination] = models[source]
    with pytest.raises(ValueError, match="PAPER-005"):
        collect(api_server, tmp_path, plan, report)
    assert api_server.requests == []


@pytest.mark.parametrize("unsupported", [False, True])
def test_exclusions_retain_ticker(api_server: ApiServer, tmp_path: Path, unsupported: bool) -> None:
    plan = paper_plan()
    ticker = serve_paper(api_server, plan)
    if unsupported:
        native = narrow_json_to_dict(
            require_list(api_server.documents["/markets"][0], "markets")[0]
        )
        native["strike_type"] = "unknown"
    else:
        book = api_server.documents["/markets/" + ticker + "/orderbook"][0]
        require_dict(book, "orderbook_fp")["yes_dollars"] = []
    report = collect(api_server, tmp_path, plan, fitted_report())
    assert report["predictions"] == []
    assert report["exclusions"] == [
        {"ticker": ticker, "reason": "unsupported_strike" if unsupported else "missing_book_side"}
    ]
    assert report["selected_contracts"] == 1


def test_real_http_failure_cannot_become_predictions(api_server: ApiServer, tmp_path: Path) -> None:
    with pytest.raises(httpx.HTTPStatusError, match="404"):
        collect(api_server, tmp_path, paper_plan(), fitted_report())
    assert api_server.requests == ["/series/KXHIGHTBOS?"]
    assert list(tmp_path.rglob("*.json")) == []


def test_receipt_window_and_cross_source_freshness(api_server: ApiServer, tmp_path: Path) -> None:
    plan = paper_plan()
    serve_paper(api_server, plan)
    with pytest.raises(ValueError, match="PAPER-004"):
        collect(
            api_server, tmp_path / "early", plan, fitted_report(), lambda: "2000-01-01T00:00:00Z"
        )
    serve_paper(api_server, plan)
    with pytest.raises(ValueError, match="PAPER-007"):
        collect(
            api_server,
            tmp_path / "stale",
            plan,
            fitted_report(),
            lambda: plan["window_end"],
            lambda: plan["window_start"],
        )
    serve_paper(api_server, plan)
    with pytest.raises(ValueError, match="PAPER-007"):
        collect(
            api_server,
            tmp_path / "reversed",
            plan,
            fitted_report(),
            lambda: plan["window_start"],
            lambda: plan["window_end"],
        )
