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
from covenant_ml.markets.paper_scoring import load_verified_predictions, reconcile_predictions
from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
from covenant_ml.markets.reporting import write_market_report
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._campaign_fixtures import single_campaign
from ._forecast_fixtures import native_market
from ._paper_fixtures import serve_paper
from ._server import ApiServer


def record_attempt(root: Path, campaign: Campaign, operation: str, observed: str) -> None:
    (root / "attempts").mkdir(exist_ok=True)
    name = campaign["plans"][0]["name"]
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


def completed_campaign(
    server: ApiServer, destination: Path, kind: str = "binary"
) -> tuple[Campaign, Path, str, str]:
    campaign, root, digest = single_campaign(destination)
    plan = campaign["plans"][0]
    ticker = serve_paper(server, plan)
    prediction_path = root / "reports" / (plan["name"] + "-capture.json")
    assert (
        capture_main(
            [
                "--plan",
                str(root / "plans" / (plan["name"] + ".json")),
                "--models",
                campaign["models"][0]["path"],
                "--protocol",
                campaign["protocol_path"],
                "--archive",
                str(root / "archives" / (plan["name"] + "-capture")),
                "--out",
                str(prediction_path),
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
    record_attempt(root, campaign, "capture", _test_hooks.clock())
    day = date.fromisoformat(plan["event_day"])
    observed = (day + timedelta(days=2)).isoformat() + "T15:00:00Z"
    market = native_market(day)
    market["status"] = "finalized"
    if kind == "pending":
        market["status"] = "active"
    elif kind == "scalar":
        market.update(result="scalar", settlement_value_dollars="0.2500")
    elif kind == "changed":
        market["rules_primary"] = "Revised rules"
    server.documents["/markets/" + ticker] = [{"market": market}]
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
    record_attempt(root, campaign, "settlement", observed)
    return campaign, root, digest, observed


def evaluate(root: Path, digest: str, observed: str, url: str) -> JSONObject:
    from covenant_ml.markets.campaign_evaluation import evaluate_campaign

    with closing(_test_hooks.client_factory(5.0, None)) as client:

        def factory(path: Path) -> ResponseArchive:
            return ResponseArchive(path, url, client, _test_hooks.clock, _test_hooks.pause)

        report, sources = evaluate_campaign(root, digest, observed, factory)
    assert root / "campaign.json" in sources
    return report


__all__ = ["completed_campaign", "evaluate", "record_attempt"]
