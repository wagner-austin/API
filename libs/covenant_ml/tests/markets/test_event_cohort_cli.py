"""Exercise complete cohort CLI with real HTTP, child process and tampered evidence."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import dump_json_str
from platform_core.run_record import run_record_sidecar
from scripts.event_cohort import main

from covenant_ml.markets.reporting import read_verified_market_report
from covenant_ml.markets.validation import read_document
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._server import ApiServer, api_server
from .test_event_cohort import END, START, serve_cohort

__all__ = ["api_server"]


def test_real_cli_and_provenance(tmp_path: Path, api_server: ApiServer) -> None:
    serve_cohort(api_server)
    plan, root, output = tmp_path / "plan.json", tmp_path / "raw", tmp_path / "report.json"
    plan.write_text(
        dump_json_str({"series": ["KXTEST"], "start_ts": START, "end_ts": END}), "utf-8"
    )
    base = ["--plan", str(plan), "--archive", str(root), "--base-url", api_server.url]
    assert main([*base, "--mode", "capture", "--out", str(output)], stand_in_pin) == 0
    report, sources, digest = read_verified_market_report(output, "COHORT-TEST")
    assert (len(sources), len(digest)) == (5, 64)
    resumed = tmp_path / "resumed.json"
    assert main([*base, "--mode", "resume", "--out", str(resumed)], stand_in_pin) == 0
    assert read_document(resumed) == report
    replay = tmp_path / "replay.json"
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.event_cohort",
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
    assert read_document(replay) == report
    assert len(api_server.requests) == 2
    with pytest.raises(FileExistsError, match="MARKET-027"):
        main([*base, "--mode", "capture", "--out", str(output)], stand_in_pin)
    replay.unlink()
    with pytest.raises(FileExistsError, match="MARKET-027"):
        main([*base, "--mode", "capture", "--out", str(replay)], stand_in_pin)
    plan.write_text(
        dump_json_str({"series": ["KXTEST"], "start_ts": 9999999998, "end_ts": 9999999999}), "utf-8"
    )
    with pytest.raises(ValueError, match="COHORT-005"):
        main([*base, "--mode", "capture", "--out", str(tmp_path / "future.json")], stand_in_pin)
    with pytest.raises(ValueError, match="COHORT-TEST"):
        read_verified_market_report(output, "COHORT-TEST")
    with (
        pytest.warns(RuntimeWarning, match="found in sys.modules"),
        pytest.raises(NativeLibrariesAlreadyLoadedError),
    ):
        runpy.run_module("scripts.event_cohort", run_name="__main__")
