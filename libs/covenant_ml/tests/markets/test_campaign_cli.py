"""Real campaign preparation, idle ticks and subprocess entry-point failure twins."""

from __future__ import annotations

import runpy
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import load_json_str, narrow_json_to_dict, require_dict
from scripts.paper_campaign import main

from tests.benchmarking._cli_fixtures import stand_in_pin


def test_prepare_tick_and_real_entry(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model, protocol = tmp_path / "model.json", tmp_path / "protocol.json"
    model.write_text("{}", encoding="utf-8")
    protocol.write_text('{"study":"future"}', encoding="utf-8")
    day = (datetime.now(UTC).date() + timedelta(days=2)).isoformat()
    root = tmp_path / "campaign"
    arguments = [
        "prepare",
        "--root",
        str(root),
        "--name",
        "future",
        "--first",
        day,
        "--last",
        day,
        "--protocol",
        str(protocol),
        "--models-00",
        str(model),
        "--models-12",
        str(model),
    ]
    assert main(arguments, stand_in_pin) == 0
    digest = sha256((root / "campaign.json").read_bytes()).hexdigest()
    assert capsys.readouterr().out == f"Campaign: {root / 'campaign.json'}\nSHA-256: {digest}\n"
    tick = ["tick", "--root", str(root), "--digest", digest]
    assert main(tick, stand_in_pin) == 0
    report = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    assert require_dict(report, "counts") == {
        "complete": 0,
        "missed": 0,
        "failed": 0,
        "interrupted": 0,
        "dependency_missing": 0,
        "future": 12,
    }
    results = sorted(
        path for path in (root / "ticks").glob("*.json") if ".started." not in path.name
    )
    assert len(results) == 1
    assert narrow_json_to_dict(load_json_str(results[0].read_text("utf-8"))) == report
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.paper_campaign", *tick],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (
        require_dict(narrow_json_to_dict(load_json_str(completed.stdout)), "counts")
        == report["counts"]
    )
    failed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.paper_campaign",
            "tick",
            "--root",
            str(root),
            "--digest",
            "b" * 64,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert failed.returncode == 1
    assert failed.stderr.splitlines()[-1] == (
        "ValueError: [CAMPAIGN-007] campaign manifest differs from scheduler identity"
    )
    assert len(list((root / "ticks").glob("*.started.json"))) == 3
    assert len(list((root / "ticks").glob("*.json"))) == 5


def test_entry_refuses_late_pin() -> None:
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.paper_campaign", run_name="__main__")
