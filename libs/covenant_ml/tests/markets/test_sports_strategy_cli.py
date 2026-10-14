"""Real capture/replay and process success/failure for the frozen strategy CLI."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import dump_json_str, require_dict
from platform_core.run_record import run_record_sidecar
from scripts.sports_strategy import main

from covenant_ml.markets.validation import read_document
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._server import ApiServer, api_server
from ._strategy_fixtures import candle_payload, native_market, observation

__all__ = ["api_server"]


def test_real_capture_replay_and_process_failure(api_server: ApiServer, tmp_path: Path) -> None:
    stamps = ("26JAN01", "26MAR01", "26MAR02", "26MAR03")
    api_server.documents["/historical/markets"] = [
        {"markets": [native_market(day) for day in stamps], "cursor": ""}
    ]
    for day in stamps:
        payload = candle_payload(day)
        if day == "26MAR03":
            payload["candlesticks"] = []
        api_server.documents[
            "/historical/markets/" + observation(day)[0]["ticker"] + "/candlesticks"
        ] = [payload]
    archive = tmp_path / "archive"
    args = ["--archive", str(archive), "--market-url", api_server.url]
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    assert main([*args, "--mode", "capture", "--out", str(first)], stand_in_pin) == 0
    assert main([*args, "--mode", "replay", "--out", str(second)], stand_in_pin) == 0
    assert first.read_bytes() == second.read_bytes()
    assert require_dict(read_document(first), "coverage") == {
        "selected_games": 4,
        "nonbinary": 0,
        "not_open": 0,
        "late_training_settlement": 0,
        "eligible": 3,
        "missing_candle": 1,
    }
    third = tmp_path / "third.json"
    command = [
        sys.executable,
        "-m",
        "scripts.sports_strategy",
        *args,
        "--mode",
        "replay",
        "--out",
        str(third),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    assert completed.stdout == f"Report: {third}\n"
    assert read_document(third)["decisions"] == read_document(first)["decisions"]
    for target in (first, tmp_path / "sidecar.json"):
        run_record_sidecar(target).write_text("{}", "utf-8")
        with pytest.raises(FileExistsError, match="STRATEGY-009"):
            main([*args, "--mode", "replay", "--out", str(target)], stand_in_pin)
    envelope = next((archive / "inventory").glob("*.json"))
    document = read_document(envelope)
    document["body"] = "{}"
    envelope.write_text(dump_json_str(document), "utf-8")
    bad = tmp_path / "bad.json"
    command[-1] = str(bad)
    failed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
    assert failed.returncode == 1
    assert (
        failed.stderr.splitlines()[-1] == "ValueError: [MARKET-015] archived response hash mismatch"
    )
    assert not bad.exists()
    assert len(api_server.requests) == 5


def test_entry_requires_early_pinning() -> None:
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.sports_strategy", run_name="__main__")
