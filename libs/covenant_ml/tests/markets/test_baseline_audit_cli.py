"""Real subprocess replay verifies report provenance and refuses changed sources."""

from __future__ import annotations

import runpy
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from scripts.forecast_baseline_audit import main

from covenant_ml.markets.provenance import market_fingerprint
from covenant_ml.markets.reporting import read_verified_market_report, write_market_report
from tests.benchmarking._cli_fixtures import stand_in_pin

from .test_baseline_audit import fitted_report


def test_real_cli_replay_and_failing_process(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    document = fitted_report(tmp_path)
    write_market_report(
        document,
        market_fingerprint(stand_in_pin(), config_test_hooks.get_env),
        source,
        tuple((tmp_path / "models").glob("*.txt")),
        "audit-test",
        "fixture",
        (),
    )
    args = ["--report", str(source), "--source-sha256", sha256(source.read_bytes()).hexdigest()]
    output = tmp_path / "first.json"
    assert main([*args, "--out", str(output)], stand_in_pin) == 0
    first, paths, _ = read_verified_market_report(output, "TEST")
    assert first["contracts"] == 40
    assert source in paths
    repeated = tmp_path / "second.json"
    process = subprocess.run(
        [sys.executable, "-m", "scripts.forecast_baseline_audit", *args, "--out", str(repeated)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert process.stdout == f"Report: {repeated}\nRun record: {repeated}.runrecord.json\n"
    replay, _, _ = read_verified_market_report(repeated, "TEST")
    for field in (
        "verified_scores",
        "policies",
        "logistic_minus_raw_market",
        "source_report_sha256",
    ):
        assert replay[field] == first[field]
    with pytest.raises(FileExistsError, match="MARKET-027"):
        main([*args, "--out", str(output)], stand_in_pin)
    wrong = ["--report", str(source), "--source-sha256", "0" * 64]
    with pytest.raises(ValueError, match="BASELINE-001"):
        main([*wrong, "--out", str(tmp_path / "wrong.json")], stand_in_pin)
    (tmp_path / "models/fold-1.txt").write_bytes(b"Changed model bytes.")
    failed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.forecast_baseline_audit",
            *args,
            "--out",
            str(tmp_path / "failed.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert failed.returncode == 1
    assert (
        failed.stderr.splitlines()[-1]
        == "ValueError: [BASELINE-001] original source evidence changed"
    )
    assert not (tmp_path / "failed.json").exists()
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.forecast_baseline_audit", run_name="__main__")
