"""Actual HTTP acquisition, persisted predictions, offline replay and entry-point twins."""

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
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_list,
)
from platform_core.run_record import decode_run_record
from scripts.paper_research import main

from covenant_ml.markets.archive import decode_capture, encode_capture
from covenant_ml.markets.forecast_model import decode_fitted_logit, predict_logit
from covenant_ml.markets.forecast_records import decode_forecast_features
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._paper_fixtures import cli_inputs, paper_plan, serve_paper
from ._server import ApiServer, api_server

__all__ = ["api_server"]


def test_complete_capture_replay_and_real_entry(api_server: ApiServer, tmp_path: Path) -> None:
    plan = paper_plan()
    ticker = serve_paper(api_server, plan)
    base = [
        *cli_inputs(tmp_path, plan),
        "--market-url",
        api_server.url,
        "--forecast-url",
        api_server.url,
    ]
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    assert main([*base, "--mode", "capture", "--out", str(first)], stand_in_pin) == 0
    assert len(api_server.requests) == 4
    assert main([*base, "--mode", "replay", "--out", str(second)], stand_in_pin) == 0
    assert first.read_bytes() == second.read_bytes()
    report = narrow_json_to_dict(load_json_str(first.read_text("utf-8")))
    predictions = require_list(report, "predictions")
    assert len(predictions) == 1
    row = narrow_json_to_dict(predictions[0])
    assert row["ticker"] == ticker
    assert row["book"] == {"bid": 4000, "ask": 6000, "bid_size": 750, "ask_size": 900}
    assert row["raw_market"] == 0.5
    assert row["raw_forecast"] == require_dict(row, "inputs")["weather_p"]
    inputs = decode_forecast_features(require_dict(row, "inputs"))
    models = require_dict(report, "models")
    for name in ("market_calibrated", "market_plus_forecast"):
        assert (
            row[name]
            == predict_logit(decode_fitted_logit(require_dict(models, name)), (inputs,))[0]
        )
    assert report["orders_submitted"] == 0
    assert report["exclusions"] == []
    record = decode_run_record(
        narrow_json_to_dict(load_json_str(Path(str(first) + ".runrecord.json").read_text("utf-8")))
    )
    assert record["payload_digest"] == sha256(first.read_bytes()).hexdigest()
    assert record["observations"] == ({"name": "pre_outcome_predictions", "value": 1.0},)
    third = tmp_path / "real-entry.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.paper_research",
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
    assert completed.stdout == f"Predictions: {third}\nRun record: {third}.runrecord.json\n"
    reproduced = narrow_json_to_dict(load_json_str(third.read_text("utf-8")))
    assert reproduced["predictions"] == predictions
    assert len(api_server.requests) == 4
    with pytest.raises(FileExistsError, match="PAPER-010"):
        main([*base, "--mode", "capture", "--out", str(first)], stand_in_pin)


def test_replay_refuses_changed_prediction_or_receipt(
    api_server: ApiServer, tmp_path: Path
) -> None:
    plan = paper_plan()
    serve_paper(api_server, plan)
    base = [
        *cli_inputs(tmp_path, plan),
        "--market-url",
        api_server.url,
        "--forecast-url",
        api_server.url,
    ]
    assert (
        main([*base, "--mode", "capture", "--out", str(tmp_path / "good.json")], stand_in_pin) == 0
    )
    path = tmp_path / "capture" / "prediction.json"
    original = decode_capture(narrow_json_to_dict(load_json_str(path.read_text("utf-8"))))
    for change in ("identity", "body", "early", "late", "outside"):
        changed = encode_capture(original)
        if change == "identity":
            changed["url"] = "paper://another-run"
        elif change == "body":
            changed["body"] = "{}"
            changed["digest"] = sha256(b"{}").hexdigest()
        elif change == "early":
            changed["captured_at"] = plan["window_start"]
        elif change == "late":
            changed["captured_at"] = (
                datetime.fromisoformat(original["captured_at"]) + timedelta(seconds=61)
            ).isoformat()
        else:
            changed["captured_at"] = (
                datetime.fromisoformat(plan["window_end"]) + timedelta(seconds=1)
            ).isoformat()
        path.write_text(dump_json_str(changed), encoding="utf-8")
        output = tmp_path / (change + ".json")
        with pytest.raises(ValueError, match="PAPER-009"):
            main([*base, "--mode", "replay", "--out", str(output)], stand_in_pin)
        assert not output.exists()
    assert len(api_server.requests) == 4


@pytest.mark.parametrize("name", ["model.json", "protocol.json"])
def test_frozen_file_identity_precedes_io(api_server: ApiServer, tmp_path: Path, name: str) -> None:
    base = cli_inputs(tmp_path, paper_plan())
    (tmp_path / name).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="PAPER-008"):
        main([*base, "--mode", "capture", "--out", str(tmp_path / "bad.json")], stand_in_pin)
    assert api_server.requests == []
    assert not (tmp_path / "capture").exists()


def test_entry_refuses_late_numeric_pin() -> None:
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.paper_research", run_name="__main__")
