"""Real CLI processes acquire and replay official documents with bound evidence."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from scripts.cpi_release import main

from covenant_ml.markets.bls_collection import RELEASE_PATH
from covenant_ml.markets.provenance import implementation_digest
from covenant_ml.markets.reporting import read_verified_market_report
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._server import ApiServer, api_server
from .test_bls_collection import PUBLICATION
from .test_bls_records import release_document
from .test_bls_release import native_html

__all__ = ["api_server"]


def arguments(root: Path, url: str) -> list[str]:
    """Declare the exact target, publication and implementation for the CLI."""
    return [
        "--archive",
        str(root / "capture"),
        "--source-url",
        url,
        "--target-month",
        "2026-08",
        "--published-at",
        PUBLICATION,
        "--implementation-sha256",
        implementation_digest(),
    ]


def test_cli_capture_and_separate_process_replay(api_server: ApiServer, tmp_path: Path) -> None:
    api_server.text_documents[RELEASE_PATH] = [native_html()]
    args = arguments(tmp_path, api_server.url)
    first, second, third = (tmp_path / name for name in ("first.json", "second.json", "third.json"))
    assert main([*args, "--mode", "capture", "--out", str(first)], stand_in_pin) == 0
    assert main([*args, "--mode", "replay", "--out", str(second)], stand_in_pin) == 0
    assert first.read_bytes() == second.read_bytes()
    report, evidence, digest = read_verified_market_report(first, "TEST")
    assert report["release"] == release_document()
    assert report["orders_submitted"] == 0
    assert len(evidence) == 3
    assert len(digest) == 64
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.cpi_release",
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
    assert process.stdout == f"CPI release: {third}\nRun record: {third}.runrecord.json\n"
    assert read_verified_market_report(third, "TEST")[0]["release"] == report["release"]
    with pytest.raises(FileExistsError, match="CPI-012"):
        main([*args, "--mode", "replay", "--out", str(first)], stand_in_pin)
    first.unlink()
    with pytest.raises(FileExistsError, match="CPI-012"):
        main([*args, "--mode", "replay", "--out", str(first)], stand_in_pin)
    assert api_server.requests == [RELEASE_PATH + "?"]


@pytest.mark.parametrize("failure", ["http", "implementation", "stale"])
def test_real_cli_failure_is_nonzero_and_no_report(
    api_server: ApiServer, tmp_path: Path, failure: str
) -> None:
    args = arguments(tmp_path, api_server.url)
    if failure == "implementation":
        args[-1] = "0" * 64
    elif failure == "stale":
        args[args.index("--target-month") + 1] = "2026-09"
        api_server.text_documents[RELEASE_PATH] = [native_html()]
    output = tmp_path / "failed.json"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.cpi_release",
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
    if failure == "http":
        assert process.stderr.splitlines()[-2:] == [
            "httpx.HTTPStatusError: Client error '404 Not Found' for url "
            f"'{api_server.url}{RELEASE_PATH}?'",
            "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404",
        ]
    elif failure == "implementation":
        assert process.stderr.splitlines()[-1] == (
            "ValueError: [CPI-011] implementation differs from the declared deployment"
        )
        with pytest.raises(ValueError, match="CPI-011"):
            main([*args, "--mode", "capture", "--out", str(output)], stand_in_pin)
    else:
        assert process.stderr.splitlines()[-1] == (
            "ValueError: [CPI-005] target headline or publication header is inconsistent"
        )
    assert output.exists() is False
    assert Path(str(output) + ".runrecord.json").exists() is False
    assert api_server.requests == ([] if failure == "implementation" else [RELEASE_PATH + "?"])
    assert (tmp_path / "capture").exists() is (failure == "stale")


def test_entry_rejects_late_arithmetic_pin() -> None:
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.cpi_release", run_name="__main__")
