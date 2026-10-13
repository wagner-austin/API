"""Actual CLI file workflows and subprocess success/failure twins."""

from __future__ import annotations

import runpy
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict
from scripts.execution_rehearsal import main

from covenant_ml.markets.execution_records import encode_intent
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._execution_fixtures import intent, native_order, settlement


def test_cli_full_local_lifecycle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "risk.sqlite"
    base = ["--ledger", str(path)]
    assert main([*base, "init"], stand_in_pin) == 0
    report = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    assert report["mode"] == "local_rehearsal"
    assert report["inspection_only_order_payload"] is None
    created = int(datetime.now(UTC).timestamp() * 1000) - 1000
    row = intent(created=created)
    proposal, receipt, final = (
        tmp_path / name for name in ("intent.json", "order.json", "final.json")
    )
    proposal.write_text(dump_json_str(encode_intent(row)), encoding="utf-8")
    receipt.write_text(dump_json_str({"order": native_order(row)}), encoding="utf-8")
    final.write_text(dump_json_str(settlement(row)), encoding="utf-8")
    assert main([*base, "reserve", "--intent", str(proposal)], stand_in_pin) == 0
    report = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    assert report["exposure"] == 10000
    assert report["orders_sent"] == 0
    assert report["live_ready"] is False
    assert report["issues"] == ["EXECUTION-UNRESOLVED"]
    assert main([*base, "receipt", "--order", str(receipt)], stand_in_pin) == 0
    report = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    assert report["exposure"] == 10000
    assert report["issues"] == []
    assert (
        main(
            [
                *base,
                "settlement",
                "--client-order-id",
                row["client_order_id"],
                "--evidence",
                str(final),
            ],
            stand_in_pin,
        )
        == 0
    )
    report = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    assert report["gross_realized_loss"] == 10000
    assert report["exposure"] == 0
    assert main([*base, "halt", "--reason", "finished rehearsal"], stand_in_pin) == 0
    report = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    assert report["issues"] == ["EXECUTION-HALTED"]
    assert main([*base, "status"], stand_in_pin) == 0
    assert narrow_json_to_dict(load_json_str(capsys.readouterr().out)) == report


def test_subprocess_persistence_and_nonzero_failure(tmp_path: Path) -> None:
    path = tmp_path / "process.sqlite"
    base = [sys.executable, "-m", "scripts.execution_rehearsal", "--ledger", str(path)]
    initialized = subprocess.run(
        [*base, "init"], capture_output=True, text=True, check=True, timeout=30
    )
    expected = narrow_json_to_dict(load_json_str(initialized.stdout))
    checked = subprocess.run(
        [*base, "status"], capture_output=True, text=True, check=True, timeout=30
    )
    assert narrow_json_to_dict(load_json_str(checked.stdout)) == expected
    failed = subprocess.run(
        [*base, "init"], capture_output=True, text=True, check=False, timeout=30
    )
    assert failed.returncode == 1
    assert failed.stderr.splitlines()[-1].split(":", 1)[0] == "FileExistsError"
    assert failed.stdout == ""
    checked = subprocess.run(
        [*base, "status"], capture_output=True, text=True, check=True, timeout=30
    )
    assert narrow_json_to_dict(load_json_str(checked.stdout)) == expected


def test_entry_refuses_late_numeric_pin() -> None:
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.execution_rehearsal", run_name="__main__")
