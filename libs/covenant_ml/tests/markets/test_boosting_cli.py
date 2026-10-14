"""Real archived inputs, actual learner fits, subprocess execution and evidence failures."""

from __future__ import annotations

import runpy
import subprocess
import sys
from datetime import date
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import require_dict, require_list
from scripts.forecast_boosting import main

from covenant_ml.markets.reporting import read_verified_market_report
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._server import ApiServer, api_server
from .test_calibration_evidence import _capture

__all__ = ["api_server"]


@pytest.mark.parametrize("study", ["fixed", "selection"])
def test_actual_cli_and_replay_preserve_scores_and_models(
    api_server: ApiServer, tmp_path: Path, study: str
) -> None:
    source = _capture(api_server, tmp_path, date(2026, 5, 14), "2026-04-24")
    protocol = tmp_path / "protocol.md"
    protocol.write_bytes(b"Frozen real-model integration experiment.\n")
    args = [
        "--study",
        study,
        "--report",
        str(source),
        "--markets-archive",
        str(tmp_path / "markets"),
        "--protocol",
        str(protocol),
        "--market-url",
        api_server.url,
        "--source-sha256",
        sha256(source.read_bytes()).hexdigest(),
        "--protocol-sha256",
        sha256(protocol.read_bytes()).hexdigest(),
        "--first-fold-end",
        "2026-05-04",
    ]
    requests = len(api_server.requests)
    first, second, third = (tmp_path / name for name in ("first.json", "second.json", "third.json"))
    assert main([*args, "--out", str(first)], stand_in_pin) == 0
    assert main([*args, "--out", str(second)], stand_in_pin) == 0
    original, paths, _ = read_verified_market_report(first, "TEST")
    repeated, _, _ = read_verified_market_report(second, "TEST")
    assert original["folds"] == repeated["folds"]
    assert original["scores"] == repeated["scores"]
    assert len(require_list(require_dict(original, "scores"), "event_dates")) == 20
    for name in ("fold-1.txt", "fold-2.txt", "protocol.md"):
        assert (tmp_path / "first.models" / name).read_bytes() == (
            tmp_path / "second.models" / name
        ).read_bytes()
    assert all(path.is_absolute() for path in paths)
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.forecast_boosting", *args, "--out", str(third)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.stdout == f"Report: {third}\nRun record: {third}.runrecord.json\n"
    actual, _, _ = read_verified_market_report(third, "TEST")
    assert actual["scores"] == original["scores"]
    assert actual["folds"] == original["folds"]
    with pytest.raises(FileExistsError, match="BOOST-006"):
        main([*args, "--out", str(first)], stand_in_pin)
    sidecar_only = tmp_path / "orphan.json"
    Path(str(sidecar_only) + ".runrecord.json").write_bytes(b"existing evidence")
    with pytest.raises(FileExistsError, match="BOOST-006"):
        main([*args, "--out", str(sidecar_only)], stand_in_pin)
    wrong = list(args)
    wrong[wrong.index("--source-sha256") + 1] = "0" * 64
    with pytest.raises(ValueError, match=r"BOOST-005.*source report"):
        main([*wrong, "--out", str(tmp_path / "wrong.json")], stand_in_pin)
    protocol.write_bytes(b"changed protocol")
    with pytest.raises(ValueError, match=r"BOOST-005.*protocol"):
        main([*args, "--out", str(tmp_path / "bad.json")], stand_in_pin)
    failed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.forecast_boosting",
            *args,
            "--out",
            str(tmp_path / "failed.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert failed.returncode == 1
    assert failed.stderr.splitlines()[-1] == (
        "ValueError: [BOOST-005] protocol differs from frozen digest"
    )
    assert not (tmp_path / "failed.json").exists()
    # Changing the input protocol cannot change the archived copy already hashed.
    verified, _, _ = read_verified_market_report(first, "TEST")
    assert verified == original
    assert len(api_server.requests) == requests
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.forecast_boosting", run_name="__main__")
