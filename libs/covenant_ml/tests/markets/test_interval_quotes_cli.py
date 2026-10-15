"""Actual CLI capture, child replay, provenance refusal and network failure."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from httpx import HTTPStatusError
from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import dump_json_str
from platform_core.run_record import run_record_sidecar
from scripts.interval_quotes import main

from covenant_ml.markets.provenance import market_fingerprint
from covenant_ml.markets.reporting import read_verified_market_report, write_market_report
from covenant_ml.markets.validation import read_document
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._server import ApiServer, api_server
from .test_interval_quotes import (
    batch_document,
    candle,
    cohort_document,
    plan_document,
    serve_quotes,
)

__all__ = ["api_server"]


def test_cli_capture_resume_child_replay_and_failures(
    tmp_path: Path, api_server: ApiServer
) -> None:
    cohort, plan, root = tmp_path / "cohort.json", tmp_path / "plan.json", tmp_path / "raw"
    write_market_report(
        cohort_document(),
        market_fingerprint(stand_in_pin(), config_test_hooks.get_env),
        cohort,
        (),
        "test-cohort",
        "fixture",
        (),
    )
    plan.write_text(dump_json_str(plan_document()), "utf-8")
    serve_quotes(api_server, batch_document([candle()]))
    args = [
        "--cohort",
        str(cohort),
        "--plan",
        str(plan),
        "--archive",
        str(root),
        "--base-url",
        api_server.url,
    ]
    output = tmp_path / "report.json"
    assert main([*args, "--mode", "capture", "--out", str(output)], stand_in_pin) == 0
    report, sources, digest = read_verified_market_report(output, "IQUOTE-TEST")
    assert (len(sources), len(digest)) == (7, 64)
    assert report["status_counts"] == {"two_sided": 1, "no_prior_candle": 1}
    resumed = tmp_path / "resumed.json"
    assert main([*args, "--mode", "resume", "--out", str(resumed)], stand_in_pin) == 0
    assert read_document(resumed) == report
    replay = tmp_path / "replay.json"
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.interval_quotes",
            *args,
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
        main([*args, "--mode", "capture", "--out", str(output)], stand_in_pin)
    replay.unlink()
    with pytest.raises(FileExistsError, match="MARKET-027"):
        main([*args, "--mode", "capture", "--out", str(replay)], stand_in_pin)
    failed = tmp_path / "failed.json"
    api_server.documents.pop("/historical/cutoff")
    with pytest.raises(HTTPStatusError, match="404"):
        main(
            [
                *args,
                "--archive",
                str(tmp_path / "failed-raw"),
                "--mode",
                "capture",
                "--out",
                str(failed),
            ],
            stand_in_pin,
        )
    assert (failed.exists(), run_record_sidecar(failed).exists()) == (False, False)
    cohort.write_text("{}", "utf-8")
    with pytest.raises(ValueError, match="IQUOTE-006"):
        main([*args, "--mode", "replay", "--out", str(tmp_path / "tampered.json")], stand_in_pin)
    with (
        pytest.warns(RuntimeWarning, match="found in sys.modules"),
        pytest.raises(NativeLibrariesAlreadyLoadedError),
    ):
        runpy.run_module("scripts.interval_quotes", run_name="__main__")
