"""Complete repeated acquisition, deterministic replay and actual process entry tests."""

from __future__ import annotations

import runpy
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import JSONValue, dump_json_str, require_list
from platform_core.run_record import decode_run_record, run_record_sidecar
from scripts.sports_capture import main

from covenant_ml.markets.sports_records import encode_sports_pair
from covenant_ml.markets.validation import read_document
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._server import ApiServer, api_server
from ._sports_fixtures import serve_sports, sports_pair

__all__ = ["api_server"]


def inputs(tmp_path: Path, origin: str) -> list[str]:
    selection = tmp_path / "selection.json"
    selection.write_text(dump_json_str({"pairs": [encode_sports_pair(sports_pair())]}), "utf-8")
    return [
        "--selection",
        str(selection),
        "--archive",
        str(tmp_path / "archive"),
        "--kalshi-url",
        origin,
        "--us-url",
        origin,
        "--window-start",
        "2020-01-01T00:00:00Z",
        "--window-end",
        "2090-01-01T00:00:00Z",
        "--rounds",
        "2",
        "--interval-seconds",
        "0",
        "--max-skew-ms",
        "10000",
    ]


def test_repeated_real_capture_replay_and_process(api_server: ApiServer, tmp_path: Path) -> None:
    serve_sports(api_server, 2)
    base = inputs(tmp_path, api_server.url)
    first, second = tmp_path / "capture.json", tmp_path / "replay.json"
    assert main([*base, "--mode", "capture", "--out", str(first)], stand_in_pin) == 0
    assert len(api_server.requests) == 8
    assert main([*base, "--mode", "replay", "--out", str(second)], stand_in_pin) == 0
    assert first.read_bytes() == second.read_bytes()
    report = read_document(first)
    assert report["rounds"] == 2
    assert report["orders_submitted"] == 0
    assert len(require_list(report, "snapshots")) == 2
    assert len(require_list(report, "sources")) == 9
    record = decode_run_record(read_document(run_record_sidecar(first)))
    assert record["payload_digest"] == sha256(first.read_bytes()).hexdigest()
    assert record["observations"] == ({"name": "snapshots", "value": 2.0},)
    third = tmp_path / "process.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.sports_capture",
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
    assert completed.stdout == f"Report: {third}\n"
    assert read_document(third)["snapshots"] == report["snapshots"]
    for existing in (first, tmp_path / "sidecar-only.json"):
        run_record_sidecar(existing).write_text("{}", "utf-8")
        with pytest.raises(FileExistsError, match="SPORTS-009"):
            main([*base, "--mode", "capture", "--out", str(existing)], stand_in_pin)
    assert len(api_server.requests) == 8
    envelope = next((tmp_path / "archive" / "0" / "kalshi").glob("*.json"))
    document = read_document(envelope)
    document["body"] = "{}"
    envelope.write_text(dump_json_str(document), "utf-8")
    failed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.sports_capture",
            *base,
            "--mode",
            "replay",
            "--out",
            str(tmp_path / "bad.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert failed.returncode == 1
    assert failed.stderr.splitlines()[-1] == (
        "ValueError: [MARKET-015] archived response hash mismatch"
    )
    assert not (tmp_path / "bad.json").exists()
    assert len(api_server.requests) == 8


@pytest.mark.parametrize(
    "key,value",
    [
        ("--rounds", "0"),
        ("--rounds", "1001"),
        ("--interval-seconds", "-1"),
        ("--interval-seconds", "61"),
        ("--max-skew-ms", "0"),
        ("--window-end", "2010-01-01T00:00:00Z"),
    ],
)
def test_invalid_settings_before_io(tmp_path: Path, key: str, value: str) -> None:
    base = inputs(tmp_path, "http://127.0.0.1:1")
    base[base.index(key) + 1] = value
    with pytest.raises(ValueError, match="SPORTS-007"):
        main([*base, "--mode", "capture", "--out", str(tmp_path / "out.json")], stand_in_pin)
    assert not (tmp_path / "archive").exists()


@pytest.mark.parametrize("kind", ["empty", "duplicate_kalshi", "duplicate_us"])
def test_invalid_universe_before_io(tmp_path: Path, kind: str) -> None:
    base = inputs(tmp_path, "http://127.0.0.1:1")
    rows: list[JSONValue] = []
    if kind != "empty":
        first = encode_sports_pair(sports_pair())
        second = dict(first)
        if kind == "duplicate_us":
            second["kalshi_ticker"] = "KXNHLGAME-26OCT13NJDET-DET"
        rows = [first, second]
    (tmp_path / "selection.json").write_text(dump_json_str({"pairs": rows}), "utf-8")
    with pytest.raises(ValueError, match="SPORTS-007"):
        main([*base, "--mode", "capture", "--out", str(tmp_path / "out.json")], stand_in_pin)
    assert not (tmp_path / "archive").exists()


def test_entry_refuses_late_pinning() -> None:
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.sports_capture", run_name="__main__")
