"""Sealed prediction verification and actual settlement CLI capture/replay twins."""

from __future__ import annotations

import runpy
import subprocess
import sys
from datetime import datetime, timedelta
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
from platform_core.run_record import run_record_sidecar
from scripts.paper_research import main as capture_main
from scripts.paper_settlements import main

from covenant_ml.markets.paper_scoring import load_verified_predictions
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._paper_fixtures import cli_inputs, paper_plan, serve_paper
from ._server import ApiServer, api_server

__all__ = ["api_server"]


def captured_report(server: ApiServer, root: Path) -> Path:
    plan = paper_plan()
    ticker = serve_paper(server, plan)
    market = narrow_json_to_dict(require_list(server.documents["/markets"][0], "markets")[0])
    server.documents["/markets/" + ticker] = [{"market": market}]
    path = root / "predictions.json"
    arguments = [
        *cli_inputs(root, plan),
        "--market-url",
        server.url,
        "--forecast-url",
        server.url,
        "--mode",
        "capture",
        "--out",
        str(path),
    ]
    assert capture_main(arguments, stand_in_pin) == 0
    return path


def read_document(path: Path) -> JSONObject:
    return narrow_json_to_dict(load_json_str(path.read_text("utf-8")))


def rewrite_report(path: Path, document: JSONObject) -> None:
    path.write_text(dump_json_str(document), encoding="utf-8")
    record_path = run_record_sidecar(path)
    record = read_document(record_path)
    record["payload_digest"] = sha256(path.read_bytes()).hexdigest()
    record_path.write_text(dump_json_str(record), encoding="utf-8")


def test_actual_pending_capture_replay_and_entry(api_server: ApiServer, tmp_path: Path) -> None:
    prediction_path = captured_report(api_server, tmp_path)
    predictions, sources = load_verified_predictions(prediction_path)
    assert len(predictions) == 1
    assert len(sources) == 10
    base = [
        "--predictions",
        str(prediction_path),
        "--archive",
        str(tmp_path / "settlements"),
        "--market-url",
        api_server.url,
    ]
    first, second, third = (
        tmp_path / "first.json",
        tmp_path / "second.json",
        tmp_path / "third.json",
    )
    assert main([*base, "--mode", "capture", "--out", str(first)], stand_in_pin) == 0
    assert main([*base, "--mode", "replay", "--out", str(second)], stand_in_pin) == 0
    assert first.read_bytes() == second.read_bytes()
    assert run_record_sidecar(first).read_bytes() == run_record_sidecar(second).read_bytes()
    report = read_document(first)
    assert report["pending"] == report["selected"] == 1
    assert report["scores"] is None
    assert report["eligible_binary"] == 0
    assert report["complete"] is False
    assert report["prediction_report_sha256"] == sha256(prediction_path.read_bytes()).hexdigest()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.paper_settlements",
            *base,
            "--mode",
            "replay",
            "--out",
            str(third),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.stdout == f"Settlements: {third}\nRun record: {third}.runrecord.json\n"
    assert read_document(third)["settlements"] == report["settlements"]
    assert len(api_server.requests) == 5
    with pytest.raises(FileExistsError, match="SETTLEMENT-007"):
        main([*base, "--mode", "capture", "--out", str(first)], stand_in_pin)


@pytest.mark.parametrize(
    "change", ["report_hash", "source_hash", "missing_seal", "body", "timestamp", "identity"]
)
def test_changed_prediction_evidence_rejected(
    api_server: ApiServer, tmp_path: Path, change: str
) -> None:
    path = captured_report(api_server, tmp_path)
    report = read_document(path)
    if change == "report_hash":
        path.write_text(path.read_text("utf-8") + " ", encoding="utf-8")
    elif change == "source_hash":
        (tmp_path / "protocol.json").write_text("{}", encoding="utf-8")
    else:
        if change == "missing_seal":
            report["sources"] = []
        elif change == "body":
            report["forecast_f"] = 99.0
        elif change == "timestamp":
            report["recorded_at"] = "2000-01-01T00:00:00Z"
        else:
            seal_path = tmp_path / "capture" / "prediction.json"
            seal = read_document(seal_path)
            seal["url"] = "paper://different"
            seal_path.write_text(dump_json_str(seal), encoding="utf-8")
            for value in require_list(report, "sources"):
                source = narrow_json_to_dict(value)
                source_path = Path(require_str(source, "path"))
                source["sha256"] = sha256(source_path.read_bytes()).hexdigest()
        rewrite_report(path, report)
    with pytest.raises(ValueError, match="SETTLEMENT-004"):
        load_verified_predictions(path)
    assert len(api_server.requests) == 4


@pytest.mark.parametrize("change", ["schema", "orders", "early", "late", "duplicate"])
def test_invalid_original_semantics_rejected(
    api_server: ApiServer, tmp_path: Path, change: str
) -> None:
    path = captured_report(api_server, tmp_path)
    report = read_document(path)
    seal_path = tmp_path / "capture" / "prediction.json"
    seal = read_document(seal_path)
    if change == "schema":
        report["schema_version"] = 2
    elif change == "orders":
        report["orders_submitted"] = 1
    elif change == "early":
        report["last_received_ms"] = 2**60
    elif change == "late":
        timestamp = (
            datetime.fromisoformat(require_str(report, "recorded_at")) + timedelta(seconds=61)
        ).isoformat()
        report["recorded_at"] = seal["captured_at"] = timestamp
    else:
        rows = require_list(report, "predictions")
        rows.append(rows[0])
    core: JSONObject = {
        key: value
        for key, value in report.items()
        if key not in ("sources", "fingerprint", "recorded_at")
    }
    body = dump_json_str(core)
    seal["body"], seal["digest"] = body, sha256(body.encode()).hexdigest()
    seal_path.write_text(dump_json_str(seal), encoding="utf-8")
    for value in require_list(report, "sources"):
        source = narrow_json_to_dict(value)
        source["sha256"] = sha256(Path(require_str(source, "path")).read_bytes()).hexdigest()
    rewrite_report(path, report)
    with pytest.raises(ValueError, match="SETTLEMENT-005"):
        load_verified_predictions(path)


def test_entry_refuses_late_determinism_pin() -> None:
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.paper_settlements", run_name="__main__")
