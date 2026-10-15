"""Actual file boundaries, child-process fits and corrupted-source refusal."""

from __future__ import annotations

import runpy
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import dump_json_str
from platform_core.run_record import run_record_sidecar
from scripts.interval_comparison import main

from covenant_ml.markets.provenance import market_fingerprint
from covenant_ml.markets.reporting import read_verified_market_report, write_market_report
from covenant_ml.markets.validation import read_document
from tests.benchmarking._cli_fixtures import stand_in_pin

from .test_interval_comparison import comparison_documents


def test_actual_command_replays_and_refuses_invalid_evidence(tmp_path: Path) -> None:
    cohort, quotes, plan = comparison_documents()
    fingerprint = market_fingerprint(stand_in_pin(), config_test_hooks.get_env)
    cohort_path, quote_path = tmp_path / "cohort.json", tmp_path / "quotes.json"
    write_market_report(cohort, fingerprint, cohort_path, (), "test-cohort", "test", ())
    cohort_hash = sha256(cohort_path.read_bytes()).hexdigest()
    quotes["cohort_sha256"] = cohort_hash
    write_market_report(quotes, fingerprint, quote_path, (cohort_path,), "test-quotes", "test", ())
    quote_hash = sha256(quote_path.read_bytes()).hexdigest()
    plan.update(cohort_sha256=cohort_hash, quotes_sha256=quote_hash)
    plan_path, output = tmp_path / "plan.json", tmp_path / "results.json"
    plan_path.write_text(dump_json_str(plan), encoding="utf-8")
    base = ["--cohort", str(cohort_path), "--quotes", str(quote_path), "--plan", str(plan_path)]
    assert main([*base, "--out", str(output)], stand_in_pin) == 0
    report, paths, _ = read_verified_market_report(output, "IFORE-TEST")
    assert report["evaluation_count"] == 20
    assert set(paths) == {
        cohort_path,
        run_record_sidecar(cohort_path),
        quote_path,
        run_record_sidecar(quote_path),
        plan_path,
        output,
        run_record_sidecar(output),
    }
    replay = tmp_path / "replay.json"
    child = subprocess.run(
        [sys.executable, "-m", "scripts.interval_comparison", *base, "--out", str(replay)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert child.stdout == f"Report: {replay}\nRun record: {run_record_sidecar(replay)}\n"
    assert read_document(replay) == report
    with pytest.raises(FileExistsError, match="MARKET-027"):
        main([*base, "--out", str(output)], stand_in_pin)
    replay.unlink()
    with pytest.raises(FileExistsError, match="MARKET-027"):
        main([*base, "--out", str(replay)], stand_in_pin)
    for field in ("cohort_sha256", "quotes_sha256"):
        plan_path.write_text(dump_json_str(dict(plan, **{field: "wrong"})), encoding="utf-8")
        with pytest.raises(ValueError, match="IFORE-005"):
            main([*base, "--out", str(tmp_path / "invalid.json")], stand_in_pin)
        assert not (tmp_path / "invalid.json").exists()
    plan_path.write_text(dump_json_str(plan), encoding="utf-8")
    cohort_path.write_text("{}", encoding="utf-8")
    failed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.interval_comparison",
            *base,
            "--out",
            str(tmp_path / "failed.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert failed.returncode == 1
    assert failed.stderr.splitlines()[-1] == (
        "ValueError: [IFORE-005] report differs from its RunRecord"
    )
    assert not (tmp_path / "failed.json").exists()
    with (
        pytest.warns(RuntimeWarning, match="found in sys.modules"),
        pytest.raises(NativeLibrariesAlreadyLoadedError),
    ):
        runpy.run_module("scripts.interval_comparison", run_name="__main__")
