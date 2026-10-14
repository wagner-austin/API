"""The real stock CLI captures, persists, replays and propagates HTTP failures."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import dump_json_str
from scripts.equity_outcomes import main

from covenant_ml.markets.reporting import read_verified_market_report
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._server import ApiServer, api_server
from .test_equity_outcomes import captured
from .test_equity_study import persist_snapshot, snapshot_document

__all__ = ["api_server"]


def arguments(root: Path, url: str) -> list[str]:
    """Persist a provenance-bound design for the actual CLI."""
    snapshot = root / "snapshot.json"
    calendar = root / "calendar.json"
    calendar.write_text(
        dump_json_str(persist_snapshot(snapshot, snapshot_document())), encoding="utf-8"
    )
    return [
        "--snapshot",
        str(snapshot),
        "--calendar",
        str(calendar),
        "--archive",
        str(root / "capture"),
        "--price-url",
        url,
    ]


def test_cli_capture_replay_real_process_and_output_collision(
    api_server: ApiServer, tmp_path: Path
) -> None:
    for symbol in ("AAPL", "MSFT"):
        api_server.documents["/v8/finance/chart/" + symbol] = [captured(symbol)]
    args = arguments(tmp_path, api_server.url)
    first, second, third = (tmp_path / name for name in ("first.json", "second.json", "third.json"))
    assert main([*args, "--mode", "capture", "--out", str(first)], stand_in_pin) == 0
    assert main([*args, "--mode", "replay", "--out", str(second)], stand_in_pin) == 0
    assert first.read_bytes() == second.read_bytes()
    report, evidence, digest = read_verified_market_report(first, "TEST")
    assert report["complete"] is True
    assert report["orders_submitted"] == 0
    assert len(evidence) == 8
    assert len(digest) == 64
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.equity_outcomes",
            *args,
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
    assert process.stdout == f"Equity outcomes: {third}\nRun record: {third}.runrecord.json\n"
    assert read_verified_market_report(third, "TEST")[0]["scores"] == report["scores"]
    with pytest.raises(FileExistsError, match="EQUITY-020"):
        main([*args, "--mode", "replay", "--out", str(first)], stand_in_pin)
    first.unlink()
    with pytest.raises(FileExistsError, match="EQUITY-020"):
        main([*args, "--mode", "replay", "--out", str(first)], stand_in_pin)
    assert len(api_server.requests) == 2


def test_real_cli_http_failure_exits_nonzero(api_server: ApiServer, tmp_path: Path) -> None:
    args = arguments(tmp_path, api_server.url)
    output = tmp_path / "failed.json"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.equity_outcomes",
            *args,
            "--mode",
            "capture",
            "--out",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 1
    failed_url = (
        api_server.url + "/v8/finance/chart/AAPL?events=div%2Csplits&interval=1d"
        "&period1=1598572800&period2=1598918400"
    )
    assert process.stderr.splitlines()[-2:] == [
        f"httpx.HTTPStatusError: Client error '404 Not Found' for url '{failed_url}'",
        "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404",
    ]
    assert output.exists() is False
    assert len(api_server.requests) == 1


def test_entry_rejects_late_arithmetic_pin() -> None:
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.equity_outcomes", run_name="__main__")
