"""Run public tape CLI capture, resume and offline replay using real boundaries."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import dump_json_str
from platform_core.run_record import run_record_sidecar
from scripts.public_tape import main

from covenant_ml.markets.reporting import read_verified_market_report
from covenant_ml.markets.validation import read_document
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._server import ApiServer, api_server
from .test_public_tape import END, START, serve_tape

__all__ = ["api_server"]


def test_real_cli_capture_resume_replay_and_integrity(
    tmp_path: Path, api_server: ApiServer
) -> None:
    serve_tape(api_server)
    plan, root, output = tmp_path / "plan.json", tmp_path / "raw", tmp_path / "report.json"
    plan.write_text(dump_json_str({"start_ts": START, "end_ts": END}), "utf-8")
    base = ["--plan", str(plan), "--archive", str(root), "--base-url", api_server.url]
    assert main([*base, "--mode", "capture", "--out", str(output)], stand_in_pin) == 0
    report, sources, digest = read_verified_market_report(output, "TAPE-TEST")
    assert (report["trade_count"], report["event_count"], len(sources), len(digest)) == (
        1,
        1,
        7,
        64,
    )
    resumed = tmp_path / "resumed.json"
    assert main([*base, "--mode", "resume", "--out", str(resumed)], stand_in_pin) == 0
    assert read_document(resumed) == report
    replay = tmp_path / "replay.json"
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.public_tape",
            *base,
            "--mode",
            "replay",
            "--out",
            str(replay),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert child.stdout == f"Report: {replay}\nRun record: {run_record_sidecar(replay)}\n"
    assert read_document(replay)["trades"] == report["trades"]
    assert len(api_server.requests) == 4
    with pytest.raises(FileExistsError, match="MARKET-027"):
        main([*base, "--mode", "capture", "--out", str(output)], stand_in_pin)
    replay.unlink()
    with pytest.raises(FileExistsError, match="MARKET-027"):
        main([*base, "--mode", "capture", "--out", str(replay)], stand_in_pin)
    plan.write_text(dump_json_str({"start_ts": START, "end_ts": 9999999999}), "utf-8")
    with pytest.raises(ValueError, match="TAPE-006"):
        main([*base, "--mode", "capture", "--out", str(tmp_path / "future.json")], stand_in_pin)
    with pytest.raises(ValueError, match="TAPE-TEST"):
        read_verified_market_report(output, "TAPE-TEST")
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.public_tape", run_name="__main__")
