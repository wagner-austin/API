"""Credential-file CLI with real cryptography, sockets and saved account evidence."""

from __future__ import annotations

import runpy
import subprocess
import sys
from collections.abc import Generator
from functools import partial
from pathlib import Path

import pytest
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import load_json_str, narrow_json_to_dict, require_dict
from scripts.kalshi_demo_preflight import main

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.kalshi_demo import DemoAccountClient, decode_preflight
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._kalshi_fixtures import KEY_ID, private_pem, serve_account, signing_key
from ._server import ApiServer, api_server

__all__ = ["api_server"]


@pytest.fixture()
def local_demo(api_server: ApiServer) -> Generator[ApiServer, None, None]:
    original = _test_hooks.demo_account_factory
    _test_hooks.demo_account_factory = partial(DemoAccountClient, origin=api_server.url)
    yield api_server
    _test_hooks.demo_account_factory = original


def test_cli_real_reads_and_exclusive_evidence(
    local_demo: ApiServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    key, signer = signing_key()
    key_id, pem, output = (tmp_path / name for name in ("key-id", "private.pem", "evidence.json"))
    key_id.write_text(KEY_ID + "\n", encoding="utf-8")
    pem.write_bytes(private_pem(key))
    args = ["--key-id-file", str(key_id), "--private-key-file", str(pem), "--output", str(output)]
    serve_account(local_demo)
    assert main(args, stand_in_pin) == 0
    text = output.read_text("utf-8")
    assert capsys.readouterr().out == text
    report = narrow_json_to_dict(load_json_str(text))
    evidence = decode_preflight(require_dict(report, "evidence"))
    assert evidence["credential_fingerprint"] == signer.fingerprint
    assert evidence["balance_microdollars"] == 20_000_099
    assert report == {
        "mode": "demo_account_preflight",
        "origin": "https://external-api.demo.kalshi.co",
        "subaccount": 0,
        "orders_sent": 0,
        "live_ready": False,
        "execution_ready": False,
        "evidence": require_dict(report, "evidence"),
    }
    serve_account(local_demo)
    with pytest.raises(FileExistsError):
        main(args, stand_in_pin)
    assert output.read_text("utf-8") == text
    assert capsys.readouterr().out == ""
    local_demo.documents = {}
    with pytest.raises(ValueError, match=r"KALSHI-DEMO-003.*HTTP 404"):
        main([*args[:-1], str(tmp_path / "failed.json")], stand_in_pin)
    assert not (tmp_path / "failed.json").exists()
    assert capsys.readouterr().out == ""


def test_real_process_rejects_missing_credentials(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.kalshi_demo_preflight",
            "--key-id-file",
            str(tmp_path / "absent"),
            "--private-key-file",
            str(tmp_path / "absent.pem"),
            "--output",
            str(tmp_path / "no-report.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.splitlines()[-1].startswith("FileNotFoundError:")
    assert not (tmp_path / "no-report.json").exists()
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.kalshi_demo_preflight", run_name="__main__")
