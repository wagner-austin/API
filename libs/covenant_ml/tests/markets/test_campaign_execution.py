"""Real process, socket, filesystem and lock twins for frozen acquisition execution."""

from __future__ import annotations

import sqlite3
import sys
from contextlib import closing
from datetime import UTC, date, datetime, time, timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str, narrow_json_to_dict, require_list

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.campaign_execution import load_campaign, tick_campaign
from covenant_ml.markets.campaign_process import run_command

from ._campaign_fixtures import single_campaign
from ._paper_fixtures import serve_paper
from ._server import ApiServer, api_server
from .test_paper_settlements_cli import read_document

__all__ = ["api_server"]


def test_real_capture_settlement_and_no_repeat(api_server: ApiServer, tmp_path: Path) -> None:
    campaign, root, digest = single_campaign(tmp_path)
    plan = campaign["plans"][0]
    ticker = serve_paper(api_server, plan)
    market = narrow_json_to_dict(require_list(api_server.documents["/markets"][0], "markets")[0])
    api_server.documents["/markets/" + ticker] = [{"market": market}]
    result = tick_campaign(
        root, digest, _test_hooks.clock, run_command, api_server.url, api_server.url
    )
    assert result == {
        "complete": 1,
        "missed": 0,
        "failed": 0,
        "interrupted": 0,
        "dependency_missing": 0,
        "future": 1,
    }
    assert (
        tick_campaign(root, digest, _test_hooks.clock, run_command, api_server.url, api_server.url)
        == result
    )
    assert len(api_server.requests) == 4
    settled_at = datetime.combine(
        date.fromisoformat(plan["event_day"]) + timedelta(days=2), time(15), UTC
    ).isoformat()
    completed = tick_campaign(
        root, digest, lambda: settled_at, run_command, api_server.url, api_server.url
    )
    assert completed == {
        "complete": 2,
        "missed": 0,
        "failed": 0,
        "interrupted": 0,
        "dependency_missing": 0,
        "future": 0,
    }
    assert (
        tick_campaign(root, digest, lambda: settled_at, run_command, api_server.url, api_server.url)
        == completed
    )
    assert len(api_server.requests) == 5
    settlement = read_document(root / "reports" / (plan["name"] + "-settlement.json"))
    assert settlement["pending"] == 1
    assert settlement["scores"] is None
    assert (
        settlement["prediction_report_sha256"]
        == sha256((root / "reports" / (plan["name"] + "-capture.json")).read_bytes()).hexdigest()
    )
    attempt = read_document(root / "attempts" / (plan["name"] + "-capture.json"))
    assert attempt["returncode"] == 0
    assert attempt["state"] == "complete"
    assert "Predictions:" in (root / "attempts" / (plan["name"] + "-capture.log")).read_text(
        "utf-8"
    )


def test_real_http_failure_recorded_and_not_repeated(api_server: ApiServer, tmp_path: Path) -> None:
    campaign, root, digest = single_campaign(tmp_path)
    name = campaign["plans"][0]["name"]
    with pytest.raises(RuntimeError, match="CAMPAIGN-010"):
        tick_campaign(root, digest, _test_hooks.clock, run_command, api_server.url, api_server.url)
    result = read_document(root / "attempts" / (name + "-capture.json"))
    assert result["state"] == "failed"
    assert result["returncode"] == 1
    assert result["report_digest"] is None
    assert "404" in (root / "attempts" / (name + "-capture.log")).read_text("utf-8")
    assert tick_campaign(
        root, digest, _test_hooks.clock, run_command, api_server.url, api_server.url
    ) == {
        "complete": 0,
        "missed": 0,
        "failed": 1,
        "interrupted": 0,
        "dependency_missing": 0,
        "future": 1,
    }
    assert len(api_server.requests) == 1


def test_missed_windows_and_missing_prediction(api_server: ApiServer, tmp_path: Path) -> None:
    campaign, root, digest = single_campaign(tmp_path)
    day = date.fromisoformat(campaign["plans"][0]["event_day"])
    before = campaign["created_at"]
    assert (
        tick_campaign(root, digest, lambda: before, run_command, api_server.url, api_server.url)[
            "future"
        ]
        == 2
    )
    due = datetime.combine(day + timedelta(days=2), time(15), UTC).isoformat()
    result = tick_campaign(root, digest, lambda: due, run_command, api_server.url, api_server.url)
    assert result == {
        "complete": 0,
        "missed": 1,
        "failed": 0,
        "interrupted": 0,
        "dependency_missing": 1,
        "future": 0,
    }
    assert api_server.requests == []


def test_all_windows_missed(api_server: ApiServer, tmp_path: Path) -> None:
    campaign, root, digest = single_campaign(tmp_path)
    day = date.fromisoformat(campaign["plans"][0]["event_day"]) + timedelta(days=3)
    observed = day.isoformat() + "T00:00:00Z"
    assert tick_campaign(
        root, digest, lambda: observed, run_command, api_server.url, api_server.url
    ) == {
        "complete": 0,
        "missed": 2,
        "failed": 0,
        "interrupted": 0,
        "dependency_missing": 0,
        "future": 0,
    }
    assert api_server.requests == []


def test_actual_launch_failure_leaves_interrupted_evidence(
    api_server: ApiServer, tmp_path: Path
) -> None:
    campaign, root, digest = single_campaign(tmp_path)
    name = campaign["plans"][0]["name"]

    def missing_executable(
        command: tuple[str, ...],
        directory: Path,
        log: Path,
        *,
        stdin_bytes: bytes | None = None,
    ) -> int:
        return run_command(
            (str(tmp_path / "absent.exe"), *command[1:]),
            directory,
            log,
            stdin_bytes=stdin_bytes,
        )

    with pytest.raises(FileNotFoundError):
        tick_campaign(
            root, digest, _test_hooks.clock, missing_executable, api_server.url, api_server.url
        )
    assert (root / "attempts" / (name + "-capture.started.json")).is_file()
    assert not (root / "attempts" / (name + "-capture.json")).exists()
    with pytest.raises(RuntimeError, match="CAMPAIGN-009"):
        tick_campaign(root, digest, _test_hooks.clock, run_command, api_server.url, api_server.url)
    assert read_document(root / "attempts" / (name + "-capture.json"))["state"] == "interrupted"
    assert (
        tick_campaign(root, digest, _test_hooks.clock, run_command, api_server.url, api_server.url)[
            "interrupted"
        ]
        == 1
    )
    assert api_server.requests == []


def test_real_lock_refuses_overlapping_tick(api_server: ApiServer, tmp_path: Path) -> None:
    _, root, digest = single_campaign(tmp_path)
    with closing(sqlite3.connect(root / "runner.sqlite")) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            tick_campaign(
                root, digest, _test_hooks.clock, run_command, api_server.url, api_server.url
            )
    assert list((root / "attempts").iterdir()) == []
    assert api_server.requests == []


@pytest.mark.parametrize("change", ["manifest", "implementation", "protocol", "model", "plan"])
def test_frozen_input_drift_refused(tmp_path: Path, change: str) -> None:
    campaign, root, digest = single_campaign(tmp_path)
    if change == "manifest":
        digest = "b" * 64
    elif change == "implementation":
        path = root / "campaign.json"
        document = read_document(path)
        document["implementation_digest"] = "b" * 64
        path.write_text(dump_json_str(document), encoding="utf-8")
        digest = sha256(path.read_bytes()).hexdigest()
    else:
        changed = tmp_path / ("protocol.json" if change == "protocol" else "model.json")
        if change == "plan":
            changed = root / "plans" / (campaign["plans"][0]["name"] + ".json")
        changed.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="CAMPAIGN-007"):
        load_campaign(root, digest)


@pytest.mark.parametrize("change", ["name", "operation", "report"])
def test_completed_attempt_identity_and_digest_refused(
    api_server: ApiServer, tmp_path: Path, change: str
) -> None:
    campaign, root, digest = single_campaign(tmp_path)
    plan = campaign["plans"][0]
    serve_paper(api_server, plan)
    tick_campaign(root, digest, _test_hooks.clock, run_command, api_server.url, api_server.url)
    path = root / "attempts" / (plan["name"] + "-capture.json")
    result = read_document(path)
    if change == "name":
        result["name"] = "different"
    elif change == "operation":
        result["operation"] = "settlement"
    else:
        result["report_digest"] = "b" * 64
    path.write_text(dump_json_str(result), encoding="utf-8")
    with pytest.raises(ValueError, match="CAMPAIGN-008"):
        tick_campaign(root, digest, _test_hooks.clock, run_command, api_server.url, api_server.url)
    assert len(api_server.requests) == 4


def test_real_process_exact_output_and_exit_code(tmp_path: Path) -> None:
    log = tmp_path / "process.log"
    assert (
        run_command((sys.executable, "-c", "print('durable'); raise SystemExit(7)"), tmp_path, log)
        == 7
    )
    assert log.read_text("utf-8") == "durable\n"
    with pytest.raises(FileExistsError):
        run_command((sys.executable, "-c", "raise SystemExit(0)"), tmp_path, log)
