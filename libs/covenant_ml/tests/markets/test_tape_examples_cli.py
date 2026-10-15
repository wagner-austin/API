"""Exercise report provenance, filesystem failures and the real examples CLI."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from scripts.public_tape_examples import main

from covenant_ml.markets.provenance import market_fingerprint
from covenant_ml.markets.reporting import read_verified_market_report, write_market_report
from tests.benchmarking._cli_fixtures import stand_in_pin

from .test_public_tape import END
from .test_tape_examples import example_report


def test_real_examples_cli_and_integrity(tmp_path: Path) -> None:
    report = tmp_path / "input.json"
    source = tmp_path / "source.txt"
    source.write_text("actual evidence bytes", "utf-8")
    write_market_report(
        example_report(),
        market_fingerprint(stand_in_pin(), lambda name: None),
        report,
        (source,),
        "kalshi-public-tape-v1",
        "test",
        (),
    )
    adjacent = tmp_path / "adjacent.json"
    empty = example_report()
    empty["start_ts_inclusive"], empty["end_ts_exclusive"] = END, END + 60
    empty["trades"], empty["markets"] = [], []
    for name in ("trade_count", "contract_count", "event_count", "quantity_centi_contracts"):
        empty[name] = 0
    write_market_report(
        empty,
        market_fingerprint(stand_in_pin(), lambda name: None),
        adjacent,
        (source,),
        "kalshi-public-tape-v1",
        "adjacent",
        (),
    )
    output = tmp_path / "summary.json"
    arguments = ["--reports", str(report), str(adjacent), "--out", str(output)]
    assert main(arguments, stand_in_pin) == 0
    summary, paths, digest = read_verified_market_report(output, "TAPE-TEST")
    assert (summary["trade_count"], summary["window_count"], len(paths), len(digest)) == (
        3,
        2,
        7,
        64,
    )
    child_output = tmp_path / "child.json"
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.public_tape_examples",
            "--reports",
            str(report),
            str(adjacent),
            "--out",
            str(child_output),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert child.stdout == ""
    child_report, _, _ = read_verified_market_report(child_output, "TAPE-TEST")
    assert child_report["windows"] == summary["windows"]
    with pytest.raises(FileExistsError, match="MARKET-027"):
        main(arguments, stand_in_pin)
    with pytest.raises(ValueError, match=r"TAPE-009.*overlap"):
        main(
            ["--reports", str(report), str(report), "--out", str(tmp_path / "overlap.json")],
            stand_in_pin,
        )
    source.write_text("changed evidence", "utf-8")
    with pytest.raises(ValueError, match=r"TAPE-009.*changed"):
        main(["--reports", str(report), "--out", str(tmp_path / "changed.json")], stand_in_pin)
    with (
        pytest.warns(RuntimeWarning, match="scripts.public_tape_examples.*found in sys.modules"),
        pytest.raises(NativeLibrariesAlreadyLoadedError),
    ):
        runpy.run_module("scripts.public_tape_examples", run_name="__main__")
