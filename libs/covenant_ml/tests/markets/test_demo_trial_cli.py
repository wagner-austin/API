"""Execute all demo CLI actions through real crypto, HTTP, files and child processes."""

from __future__ import annotations

import runpy
import subprocess
import sys
from collections.abc import Generator
from functools import partial
from pathlib import Path

import pytest
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from scripts.kalshi_demo_trial import main

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.demo_trial import build_demo_trial
from covenant_ml.markets.submission_evidence import decode_submission_timing
from covenant_ml.markets.validation import read_document
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._demo_trial_fixtures import serve_reconciliation, serve_submission
from ._kalshi_fixtures import KEY_ID, private_pem, serve_account, signing_key
from ._server import ApiServer, api_server

__all__ = ["api_server"]


@pytest.fixture()
def demo_cli(api_server: ApiServer) -> Generator[ApiServer, None, None]:
    original = _test_hooks.demo_trial_factory
    _test_hooks.demo_trial_factory = partial(build_demo_trial, origin=api_server.url)
    yield api_server
    _test_hooks.demo_trial_factory = original


def test_cli_end_to_end_and_real_restart(
    demo_cli: ApiServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    key, _ = signing_key()
    identity, pem = tmp_path / "key-id", tmp_path / "key.pem"
    identity.write_text(KEY_ID, encoding="utf-8")
    pem.write_bytes(private_pem(key))
    root = tmp_path / "trial"
    base = ["--key-id-file", str(identity), "--private-key-file", str(pem), "--root", str(root)]
    serve_account(demo_cli)
    assert (
        main([*base, "prepare", "--ticker", "MARKET-1", "--limit-price", "9800"], stand_in_pin) == 0
    )
    assert (
        narrow_json_to_dict(load_json_str(capsys.readouterr().out))["mode"] == "demo_single_order"
    )
    serve_submission(demo_cli, root)
    assert main([*base, "submit"], stand_in_pin) == 0
    report = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    assert report["confirmed_orders"] == 1
    assert main([*base, "status"], stand_in_pin) == 0
    assert narrow_json_to_dict(load_json_str(capsys.readouterr().out)) == report
    assert main([*base, "evidence"], stand_in_pin) == 0
    evidence = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    assert evidence["mode"] == "demo_submission_evidence"
    timing = decode_submission_timing(read_document(root / "submission-timing.json"))
    assert evidence["local_workflow_duration_ms"] == timing["returned_ms"] - timing["started_ms"]
    serve_reconciliation(demo_cli, root)
    assert main([*base, "reconcile"], stand_in_pin) == 0
    reconciled = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    assert reconciled["confirmed_orders"] == 1
    assert reconciled["exposure"] == 10000
    assert reconciled["live_ready"] is False
    result = subprocess.run(
        [sys.executable, "-m", "scripts.kalshi_demo_trial", *base, "status"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert narrow_json_to_dict(load_json_str(result.stdout)) == reconciled
    replay = subprocess.run(
        [sys.executable, "-m", "scripts.kalshi_demo_trial", *base, "evidence"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert narrow_json_to_dict(load_json_str(replay.stdout)) == evidence
    refused = subprocess.run(
        [sys.executable, "-m", "scripts.kalshi_demo_trial", *base, "submit"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert refused.returncode == 1
    assert refused.stdout == ""
    assert "[DEMO-TRIAL-005]" in refused.stderr.splitlines()[-1]
    assert len(demo_cli.bodies) == 1
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.kalshi_demo_trial", run_name="__main__")
