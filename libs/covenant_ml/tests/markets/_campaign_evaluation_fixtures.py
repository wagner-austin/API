"""Real prediction capture and native settlement archives for campaign analysis tests."""

from __future__ import annotations

from contextlib import closing
from datetime import date, timedelta
from hashlib import sha256
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.json_utils import JSONObject, dump_json_str
from scripts.paper_research import main as capture_main

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.archive import ResponseArchive
from covenant_ml.markets.campaign_records import Attempt, AttemptState, Campaign, encode_attempt
from covenant_ml.markets.paper_records import PaperPlan
from covenant_ml.markets.paper_scoring import load_verified_predictions, reconcile_predictions
from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
from covenant_ml.markets.reporting import write_market_report
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._campaign_fixtures import single_campaign
from ._forecast_fixtures import native_market
from ._paper_fixtures import serve_paper
from ._server import ApiServer


def record_attempt(root: Path, name: str, operation: str, observed: str) -> None:
    (root / "attempts").mkdir(exist_ok=True)
    report = root / "reports" / f"{name}-{operation}.json"
    attempt = Attempt(
        name=name,
        operation=operation,
        state=AttemptState.COMPLETE,
        observed_at=observed,
        returncode=0,
        report_digest=sha256(report.read_bytes()).hexdigest(),
    )
    (root / "attempts" / f"{name}-{operation}.json").write_text(
        dump_json_str(encode_attempt(attempt)), encoding="utf-8"
    )


def capture_plan(server: ApiServer, root: Path, campaign: Campaign, plan: PaperPlan) -> str:
    """Capture one plan through the real collector CLI, record it, return its ticker."""
    ticker = serve_paper(server, plan)
    model = next(model for model in campaign["models"] if model["hour"] == plan["forecast_hour"])
    assert (
        capture_main(
            [
                "--plan",
                str(root / "plans" / (plan["name"] + ".json")),
                "--models",
                model["path"],
                "--protocol",
                campaign["protocol_path"],
                "--archive",
                str(root / "archives" / (plan["name"] + "-capture")),
                "--out",
                str(root / "reports" / (plan["name"] + "-capture.json")),
                "--mode",
                "capture",
                "--market-url",
                server.url,
                "--forecast-url",
                server.url,
            ],
            stand_in_pin,
        )
        == 0
    )
    record_attempt(root, plan["name"], "capture", _test_hooks.clock())
    return ticker


def settle_plan(
    server: ApiServer,
    root: Path,
    plan: PaperPlan,
    ticker: str,
    digest: str,
    kind: str = "binary",
    yes: int = 1,
) -> str:
    """Archive a native settlement two days after the event and return that instant."""
    day = date.fromisoformat(plan["event_day"])
    observed = (day + timedelta(days=2)).isoformat() + "T15:00:00Z"
    market = native_market(day)
    market.update(
        ticker=ticker,
        status="finalized",
        result="yes" if yes else "no",
        settlement_value_dollars="1.0000" if yes else "0.0000",
    )
    if kind == "pending":
        market["status"] = "active"
    elif kind == "scalar":
        market.update(result="scalar", settlement_value_dollars="0.2500")
    elif kind == "changed":
        market["rules_primary"] = "Revised rules"
    server.documents["/markets/" + ticker] = [{"market": market}]
    prediction_path = root / "reports" / (plan["name"] + "-capture.json")
    predictions, sources = load_verified_predictions(prediction_path)
    archive_path = root / "archives" / (plan["name"] + "-settlement")
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        archive = ResponseArchive(
            archive_path, server.url, client, lambda: observed, _test_hooks.pause
        )
        report = reconcile_predictions(predictions, archive, True)
    report["prediction_report_sha256"] = sha256(prediction_path.read_bytes()).hexdigest()
    report["implementation_sha256"] = implementation_digest()
    write_market_report(
        report,
        market_fingerprint(stand_in_pin(), config_test_hooks.get_env),
        root / "reports" / (plan["name"] + "-settlement.json"),
        (*sources, *archive_path.glob("*.json")),
        "prospective-market-settlements",
        digest,
        (),
    )
    record_attempt(root, plan["name"], "settlement", observed)
    return observed


def completed_campaign(
    server: ApiServer, destination: Path, kind: str = "binary"
) -> tuple[Campaign, Path, str, str]:
    campaign, root, digest = single_campaign(destination)
    plan = campaign["plans"][0]
    ticker = capture_plan(server, root, campaign, plan)
    observed = settle_plan(server, root, plan, ticker, digest, kind)
    return campaign, root, digest, observed


def evaluate(root: Path, digest: str, observed: str, url: str) -> JSONObject:
    from covenant_ml.markets.campaign_evaluation import evaluate_campaign

    with closing(_test_hooks.client_factory(5.0, None)) as client:

        def factory(path: Path) -> ResponseArchive:
            return ResponseArchive(path, url, client, _test_hooks.clock, _test_hooks.pause)

        report, sources = evaluate_campaign(root, digest, observed, factory)
    assert root / "campaign.json" in sources
    return report


__all__ = ["capture_plan", "completed_campaign", "evaluate", "record_attempt", "settle_plan"]
