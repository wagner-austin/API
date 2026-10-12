"""Actual fitted models and native socket-server documents for prospective tests."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from pathlib import Path

from platform_core.json_utils import JSONObject, dump_json_str

from covenant_ml.markets.forecast_model import encode_fitted_logit, fit_logit
from covenant_ml.markets.forecast_records import encode_comparison_plan, encode_forecast_row
from covenant_ml.markets.paper_records import PaperPlan, decode_paper_plan, encode_paper_plan

from ._forecast_fixtures import comparison_plan, comparison_rows, forecast_document, native_market
from ._server import ApiServer


def fitted_report() -> JSONObject:
    rows = tuple(row for row in comparison_rows() if row["event_day"] <= "2026-04-08")
    return {
        "plan": encode_comparison_plan(comparison_plan()),
        "training_rows": [encode_forecast_row(row) for row in rows],
        "models": {
            "market_calibrated": encode_fitted_logit(fit_logit(rows, False)),
            "market_plus_forecast": encode_fitted_logit(fit_logit(rows, True)),
        },
    }


def paper_plan() -> PaperPlan:
    now = datetime.now(UTC)
    return decode_paper_plan(
        {
            "name": "paper-test",
            "series": "KXHIGHTBOS",
            "event_day": (now.date() + timedelta(days=1)).isoformat(),
            "forecast_hour": 0,
            "window_start": now.isoformat(),
            "window_end": (now + timedelta(minutes=30)).isoformat(),
            "model_digest": "a" * 64,
            "protocol_digest": "b" * 64,
        }
    )


def native_book() -> JSONObject:
    return {
        "orderbook_fp": {
            "yes_dollars": [["0.1000", "1.25"], ["0.4000", "7.50"]],
            "no_dollars": [["0.4000", "9.00"], ["0.2000", "2.00"]],
        }
    }


def serve_paper(server: ApiServer, plan: PaperPlan) -> str:
    day = date.fromisoformat(plan["event_day"])
    market = native_market(day)
    market.update(status="active", result="", open_time=plan["window_start"])
    # Future records genuinely contain no payout or settlement timestamp.
    del market["settlement_value_dollars"]
    del market["settlement_ts"]
    ticker = str(market["ticker"])
    server.documents["/series/KXHIGHTBOS"] = [
        {"series": {"ticker": "KXHIGHTBOS", "fee_type": "quadratic", "fee_multiplier": 1}}
    ]
    server.documents["/markets"] = [{"markets": [market], "cursor": ""}]
    server.documents["/v1/forecast"] = [forecast_document(day)]
    server.documents["/markets/" + ticker + "/orderbook"] = [native_book()]
    return ticker


def cli_inputs(root: Path, plan: PaperPlan) -> list[str]:
    protocol, model, plan_path = root / "protocol.json", root / "model.json", root / "plan.json"
    protocol.write_text('{"purpose":"test pre-outcome capture"}', encoding="utf-8")
    model.write_text(dump_json_str(fitted_report()), encoding="utf-8")
    document = encode_paper_plan(plan)
    document["model_digest"] = sha256(model.read_bytes()).hexdigest()
    document["protocol_digest"] = sha256(protocol.read_bytes()).hexdigest()
    plan_path.write_text(dump_json_str(document), encoding="utf-8")
    return [
        "--plan",
        str(plan_path),
        "--models",
        str(model),
        "--protocol",
        str(protocol),
        "--archive",
        str(root / "capture"),
    ]


__all__ = ["cli_inputs", "fitted_report", "native_book", "paper_plan", "serve_paper"]
