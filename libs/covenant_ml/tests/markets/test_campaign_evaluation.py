"""Actual archive replay, date weighting and fail-closed campaign evidence tests."""

from __future__ import annotations

import runpy
import subprocess
import sys
from collections.abc import Generator
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    narrow_json_to_dict,
    require_dict,
    require_list,
)
from scripts.paper_campaign_report import main

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.campaign_evaluation import summarize_dates
from covenant_ml.markets.campaign_inputs import acquisition_window
from covenant_ml.markets.campaign_records import AttemptState, encode_campaign
from covenant_ml.markets.settlement_records import ARMS
from covenant_ml.markets.validation import read_document
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._campaign_evaluation_fixtures import completed_campaign, evaluate, record_attempt
from ._campaign_fixtures import single_campaign
from ._server import ApiServer, api_server
from .test_paper_settlements_cli import rewrite_report

__all__ = ["api_server"]


def test_date_weighting_is_not_contract_weighting() -> None:
    first: JSONObject = {
        arm: {"brier": score} for arm, score in zip(ARMS, (0.1, 0.2, 0.3, 0.4), strict=True)
    }
    second: JSONObject = {
        arm: {"brier": score} for arm, score in zip(ARMS, (0.5, 0.6, 0.7, 0.8), strict=True)
    }
    result = summarize_dates(
        (("2026-01-01", 1, first), ("2026-01-01", 3, second), ("2026-01-02", 1, first))
    )
    assert result["eligible_dates"] == 2
    assert result["eligible_contracts"] == 5
    expected: dict[str, float] = dict(zip(ARMS, (0.25, 0.35, 0.45, 0.55), strict=True))
    assert require_dict(result, "mean_brier") == pytest.approx(expected)
    assert result["paired_brier_difference"] == pytest.approx(0.1)
    dates = require_list(result, "dates")
    assert [narrow_json_to_dict(value)["contracts"] for value in dates] == [4, 1]
    assert require_dict(narrow_json_to_dict(dates[0]), "brier")["raw_market"] == pytest.approx(0.4)


@pytest.mark.parametrize("kind", ["binary", "pending", "scalar", "changed"])
def test_actual_completed_campaign_replays_without_requests(
    api_server: ApiServer, tmp_path: Path, kind: str
) -> None:
    campaign, root, digest, observed = completed_campaign(api_server, tmp_path, kind)
    result = evaluate(root, digest, observed, api_server.url)
    assert len(api_server.requests) == 5
    assert result["acquisitions"] == {"complete": 2}
    assert require_dict(result, "primary_12utc") == {
        "eligible_dates": 0,
        "eligible_contracts": 0,
        "dates": [],
        "mean_brier": None,
        "paired_brier_difference": None,
    }
    scored = require_dict(result, "secondary_00utc")
    assert scored["eligible_dates"] == int(kind == "binary")
    row = narrow_json_to_dict(require_list(result, "plans")[0])
    assert row["capture_exclusions"] == []
    outcomes = require_dict(row, "outcomes")
    assert outcomes["pending"] == int(kind == "pending")
    assert outcomes["scalar"] == int(kind == "scalar")
    assert outcomes["rules_changed"] == int(kind == "changed")
    if kind == "binary":
        saved = read_document(
            root / "reports" / (campaign["plans"][0]["name"] + "-settlement.json")
        )
        assert require_dict(scored, "mean_brier") == {
            arm: require_dict(require_dict(saved, "scores"), arm)["brier"] for arm in ARMS
        }
    else:
        assert scored["paired_brier_difference"] is None


@pytest.mark.parametrize(
    "timing,state", [("before", "future"), ("during", "due"), ("after", "unrecorded_overdue")]
)
def test_unrecorded_windows_are_explicit(tmp_path: Path, timing: str, state: str) -> None:
    campaign, root, digest = single_campaign(tmp_path)
    plan = campaign["plans"][0]
    observed = (
        campaign["created_at"]
        if timing == "before"
        else plan["window_start"]
        if timing == "during"
        else (datetime.fromisoformat(plan["window_end"]) + timedelta(seconds=1)).isoformat()
    )
    result = evaluate(root, digest, observed, "http://127.0.0.1:1")
    assert narrow_json_to_dict(require_list(result, "plans")[0])["capture"] == state
    assert not (root / "attempts").exists()


@pytest.mark.parametrize(
    "state", [state for state in AttemptState if state is not AttemptState.COMPLETE]
)
def test_all_terminal_omissions_are_retained(tmp_path: Path, state: AttemptState) -> None:
    campaign, root, digest = single_campaign(tmp_path)
    (root / "attempts").mkdir()
    name = campaign["plans"][0]["name"]
    document: JSONObject = {
        "name": name,
        "operation": "capture",
        "state": state.value,
        "observed_at": campaign["created_at"],
        "returncode": 1 if state is AttemptState.FAILED else None,
        "report_digest": None,
    }
    (root / "attempts" / (name + "-capture.json")).write_text(
        dump_json_str(document), encoding="utf-8"
    )
    result = evaluate(root, digest, _test_hooks.clock(), "http://127.0.0.1:1")
    assert result["acquisitions"] == {state.value: 1, "future": 1}


@pytest.mark.parametrize("change", ["none", "digest", "future"])
def test_unfinished_start_marker(tmp_path: Path, change: str) -> None:
    campaign, root, digest = single_campaign(tmp_path)
    (root / "attempts").mkdir()
    marker = root / "attempts" / (campaign["plans"][0]["name"] + "-capture.started.json")
    marker.write_text(
        dump_json_str(
            {
                "campaign_digest": "b" * 64 if change == "digest" else digest,
                "started_at": "2999-01-01T00:00:00Z"
                if change == "future"
                else campaign["created_at"],
            }
        ),
        encoding="utf-8",
    )
    if change == "none":
        assert evaluate(root, digest, _test_hooks.clock(), "http://127.0.0.1:1")[
            "acquisitions"
        ] == {"started_unfinished": 1, "future": 1}
    else:
        with pytest.raises(ValueError, match="CAMPAIGN-011"):
            evaluate(root, digest, _test_hooks.clock(), "http://127.0.0.1:1")
    assert not marker.with_name(marker.name.replace(".started", "")).exists()


@pytest.mark.parametrize(
    "change,code",
    [
        ("name", "011"),
        ("operation", "011"),
        ("future", "011"),
        ("report_digest", "012"),
        ("collector", "012"),
        ("plan", "013"),
        ("prediction_digest", "013"),
        ("replay", "014"),
        ("missing_capture", "013"),
    ],
)
def test_changed_evidence_refused(
    api_server: ApiServer, tmp_path: Path, change: str, code: str
) -> None:
    campaign, root, digest, observed = completed_campaign(api_server, tmp_path)
    name = campaign["plans"][0]["name"]
    attempt_path = root / "attempts" / (name + "-capture.json")
    attempt = read_document(attempt_path)
    if change in ("name", "operation", "future", "report_digest", "missing_capture"):
        if change == "name":
            attempt["name"] = "other"
        elif change == "operation":
            attempt["operation"] = "settlement"
        elif change == "future":
            attempt["observed_at"] = "2999-01-01T00:00:00Z"
        elif change == "report_digest":
            attempt["report_digest"] = "b" * 64
        else:
            attempt.update(state="failed", returncode=1, report_digest=None)
        attempt_path.write_text(dump_json_str(attempt), encoding="utf-8")
    else:
        operation = "capture" if change in ("collector", "plan") else "settlement"
        path = root / "reports" / f"{name}-{operation}.json"
        report = read_document(path)
        if change == "collector":
            report["implementation_sha256"] = "b" * 64
        elif change == "plan":
            require_dict(report, "plan")["name"] = "other"
        elif change == "prediction_digest":
            report["prediction_report_sha256"] = "b" * 64
        else:
            report["pending"] = 8
        rewrite_report(path, report)
        record_attempt(root, campaign, operation, observed)
        if change == "plan":
            (root / "attempts" / (name + "-settlement.json")).unlink()
    with pytest.raises(ValueError, match="CAMPAIGN-" + code):
        evaluate(root, digest, observed, api_server.url)
    assert len(api_server.requests) == 5


def test_analyzer_does_not_require_collectors_implementation(tmp_path: Path) -> None:
    campaign, root, _ = single_campaign(tmp_path)
    document = encode_campaign(campaign)
    document["implementation_digest"] = "b" * 64
    path = root / "campaign.json"
    path.write_text(dump_json_str(document), encoding="utf-8")
    report = evaluate(
        root, sha256(path.read_bytes()).hexdigest(), campaign["created_at"], "http://127.0.0.1:1"
    )
    assert report["collector_implementation_sha256"] == "b" * 64


def test_cli_persists_provenance_and_real_subprocess(tmp_path: Path) -> None:
    _, root, digest = single_campaign(tmp_path)
    output = tmp_path / "evaluation.json"
    args = ["--root", str(root), "--digest", digest, "--out", str(output)]
    assert main(args, stand_in_pin) == 0
    assert read_document(output)["acquisitions"] == {"due": 1, "future": 1}
    assert len(require_list(read_document(output), "sources")) == 4
    with pytest.raises(FileExistsError, match="MARKET-027"):
        main(args, stand_in_pin)
    second = tmp_path / "subprocess.json"
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.paper_campaign_report", *args[:-1], str(second)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (
        completed.stdout == f"Campaign evaluation: {second}\nRun record: {second}.runrecord.json\n"
    )
    assert read_document(second)["acquisitions"] == read_document(output)["acquisitions"]
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.paper_campaign_report", run_name="__main__")


@pytest.mark.parametrize("operation", ["capture", "settlement"])
def test_receipts_after_observation_are_refused(
    api_server: ApiServer, tmp_path: Path, operation: str
) -> None:
    campaign, root, digest, _ = completed_campaign(api_server, tmp_path)
    plan = campaign["plans"][0]
    observed = plan["window_start"] if operation == "capture" else _test_hooks.clock()
    record_attempt(root, campaign, operation, observed)
    if operation == "capture":
        (root / "attempts" / (plan["name"] + "-settlement.json")).unlink()
    with pytest.raises(ValueError, match=r"CAMPAIGN-011.*later than analysis"):
        evaluate(root, digest, observed, api_server.url)
    assert len(api_server.requests) == 5


@pytest.fixture()
def restore_clock() -> Generator[None, None, None]:
    original = _test_hooks.clock
    yield
    _test_hooks.clock = original


def test_cli_replays_actual_saved_settlement(
    api_server: ApiServer, tmp_path: Path, restore_clock: None
) -> None:
    _, root, digest, observed = completed_campaign(api_server, tmp_path)
    _test_hooks.clock = lambda: observed
    output = tmp_path / "scored.json"
    assert (
        main(
            [
                "--root",
                str(root),
                "--digest",
                digest,
                "--market-url",
                api_server.url,
                "--out",
                str(output),
            ],
            stand_in_pin,
        )
        == 0
    )
    assert require_dict(read_document(output), "secondary_00utc")["eligible_contracts"] == 1
    assert len(api_server.requests) == 5


def test_missing_saved_source_is_not_refetched(api_server: ApiServer, tmp_path: Path) -> None:
    campaign, root, digest, observed = completed_campaign(api_server, tmp_path)
    archive = root / "archives" / (campaign["plans"][0]["name"] + "-settlement")
    paths = list(archive.glob("*.json"))
    assert len(paths) == 1
    paths[0].unlink()
    with pytest.raises(FileNotFoundError):
        evaluate(root, digest, observed, api_server.url)
    assert len(api_server.requests) == 5


def test_unknown_acquisition_operation_refused(tmp_path: Path) -> None:
    campaign, _, _ = single_campaign(tmp_path)
    with pytest.raises(ValueError, match="CAMPAIGN-015"):
        acquisition_window(campaign["plans"][0], "other")
