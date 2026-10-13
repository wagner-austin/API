"""Real acquisition, offline CLI execution and tampered-evidence rejection."""

from __future__ import annotations

import runpy
import subprocess
import sys
from contextlib import closing
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_int,
    require_list,
    require_str,
)
from scripts.forecast_calibration import main

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.archive import ResponseArchive
from covenant_ml.markets.calibration_evidence import load_calibration_evidence
from covenant_ml.markets.collection import MarketCollector
from covenant_ml.markets.forecast_collection import collect_forecast_rows
from covenant_ml.markets.forecast_evaluation import evaluate_forecasts
from covenant_ml.markets.forecast_records import decode_comparison_plan, encode_comparison_plan
from covenant_ml.markets.provenance import market_fingerprint
from covenant_ml.markets.reporting import read_verified_market_report, write_market_report
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._forecast_fixtures import comparison_plan, serve_comparison
from ._server import ApiServer, api_server

__all__ = ["api_server"]


def _write(report: JSONObject, paths: tuple[Path, ...], output: Path) -> None:
    write_market_report(
        report,
        market_fingerprint(stand_in_pin(), config_test_hooks.get_env),
        output,
        paths,
        "calibration-test",
        "fixture",
        (),
    )


def _capture(server: ApiServer, root: Path) -> Path:
    serve_comparison(server)
    document = encode_comparison_plan(comparison_plan())
    document["train_end"] = "2026-04-14"
    plan = decode_comparison_plan(document)
    with closing(_test_hooks.client_factory(10.0, None)) as client:
        market = ResponseArchive(
            root / "markets", server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        forecast = ResponseArchive(
            root / "forecasts", server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        rows, counts = collect_forecast_rows(
            plan, MarketCollector(market.capture), forecast.capture, print
        )
    assert counts == {"selected_contracts": 96, "included": 96}
    output = root / "original.json"
    sources = (*market.root.glob("*.json"), *forecast.root.glob("*.json"))
    _write(evaluate_forecasts(rows, plan), sources, output)
    return output


def test_offline_cli_reproduces_and_real_subprocess_fails_on_missing_evidence(
    api_server: ApiServer, tmp_path: Path
) -> None:
    source = _capture(api_server, tmp_path)
    request_count = len(api_server.requests)
    output = tmp_path / "calibrated.json"
    args = [
        "--report",
        str(source),
        "--markets-archive",
        str(tmp_path / "markets"),
        "--market-url",
        api_server.url,
        "--out",
        str(output),
    ]
    assert main(args, stand_in_pin) == 0
    report, paths, _ = read_verified_market_report(output, "TEST")
    assert len(paths) == 158
    assert set(require_dict(report, "city_models")) == {"KXHIGHTBOS", "KXHIGHPHIL", "KXHIGHTDC"}
    second = tmp_path / "repeat.json"
    assert main([*args[:-1], str(second)], stand_in_pin) == 0
    assert output.read_bytes() == second.read_bytes()
    third = tmp_path / "subprocess.json"
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.forecast_calibration", *args[:-1], str(third)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.stdout == f"Report: {third}\nRun record: {third}.runrecord.json\n"
    subprocess_report, _, _ = read_verified_market_report(third, "TEST")
    assert subprocess_report["city_models"] == report["city_models"]
    assert subprocess_report["city_calibrated"] == report["city_calibrated"]
    source.unlink()
    failed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.forecast_calibration",
            *args[:-1],
            str(tmp_path / "missing.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert failed.returncode == 1
    assert failed.stdout == ""
    assert failed.stderr.splitlines()[-1] == (
        f"FileNotFoundError: [Errno 2] No such file or directory: {str(source)!r}"
    )
    assert not (tmp_path / "missing.json").exists()
    assert len(api_server.requests) == request_count
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.forecast_calibration", run_name="__main__")


def test_rejects_unbound_and_inconsistent_archived_evidence(
    api_server: ApiServer, tmp_path: Path
) -> None:
    source = _capture(api_server, tmp_path)
    report, paths, _ = read_verified_market_report(source, "TEST")
    original_training = require_list(report, "training_rows")
    original_row = narrow_json_to_dict(original_training[0])
    with closing(_test_hooks.client_factory(10.0, None)) as client:
        archive = ResponseArchive(
            tmp_path / "markets", api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        unbound = tmp_path / "unbound.json"
        _write(report, (), unbound)
        with pytest.raises(ValueError, match="CALIBRATION-005"):
            load_calibration_evidence(unbound, archive)
        changes: tuple[tuple[str, JSONValue], ...] = (
            ("ticker", require_str(original_row, "ticker") + "-missing"),
            ("event_day", "2026-04-04"),
            ("decision_ms", require_int(original_row, "decision_ms") + 1),
            ("settled_ms", require_int(original_row, "settled_ms") + 1),
            ("yes", 1 - require_int(original_row, "yes")),
            ("settlement_rule", "Other station"),
            ("weather_p", 0.9),
        )
        for index, (field, value) in enumerate(changes):
            row = dict(narrow_json_to_dict(original_training[0]))
            row[field] = value
            training: list[JSONValue] = [row, *original_training[1:]]
            altered: JSONObject = dict(report, training_rows=training)
            path = tmp_path / f"altered-{index}.json"
            _write(altered, paths[2:], path)
            with pytest.raises(ValueError, match="CALIBRATION-006"):
                load_calibration_evidence(path, archive)
        # Re-hashing a forged report does not make changed original source bytes valid.
        paths[2].write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="CALIBRATION-004"):
            load_calibration_evidence(source, archive)


def test_unsupported_native_bracket_is_refused(api_server: ApiServer, tmp_path: Path) -> None:
    source = _capture(api_server, tmp_path)
    report, paths, _ = read_verified_market_report(source, "TEST")
    # Mutate both inventory tiers consistently, then bind their actual new bytes.
    for path in (tmp_path / "markets").glob("*.json"):
        envelope = narrow_json_to_dict(load_json_str(path.read_text("utf-8")))
        native = narrow_json_to_dict(load_json_str(require_str(envelope, "body")))
        if "markets" in native:
            for raw in require_list(native, "markets"):
                narrow_json_to_dict(raw)["strike_type"] = "unsupported"
            body = dump_json_str(native)
            envelope["body"], envelope["digest"] = body, sha256(body.encode()).hexdigest()
            path.write_text(dump_json_str(envelope), encoding="utf-8")
    changed = tmp_path / "unsupported.json"
    _write(report, paths[2:], changed)
    with closing(_test_hooks.client_factory(10.0, None)) as client:
        archive = ResponseArchive(
            tmp_path / "markets", api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        with pytest.raises(ValueError, match="CALIBRATION-006"):
            load_calibration_evidence(changed, archive)
