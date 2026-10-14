"""Real HTTP acquisition, offline CLI reproduction and corrupted-evidence failure."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import dump_json_str, narrow_json_to_dict, require_list
from platform_core.run_record import run_record_sidecar
from scripts.inflation_experiment import main

from covenant_ml.markets.inflation_selection import inflation_selection
from covenant_ml.markets.validation import read_document
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._inflation_experiment_fixtures import feed, histories, market
from ._server import ApiServer, api_server

__all__ = ["api_server"]


def test_real_capture_replay_and_corruption(api_server: ApiServer, tmp_path: Path) -> None:
    history_path, inventory_path, current_path = (
        tmp_path / name for name in ("history.json", "inventory.json", "current.json")
    )
    history_path.write_text(dump_json_str(feed()), "utf-8")
    inventory_path.write_text(
        dump_json_str({"markets": [market(i) for i in (1, 3)], "cursor": ""}), "utf-8"
    )
    current_path.write_text(dump_json_str({"markets": [market(2)], "cursor": ""}), "utf-8")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Sixty earlier errors; exact midnight; one contract per release.", "utf-8")
    selected, _ = inflation_selection(tuple(market(i) for i in (1, 2, 3)), histories())
    for index, (spec, _) in enumerate(selected):
        prefix = "/series/KXCPIYOY/markets/" if index == 1 else "/historical/markets/"
        price_field = "close_dollars" if index == 1 else "close"
        api_server.documents[prefix + spec["ticker"] + "/candlesticks"] = [
            {
                "ticker": spec["ticker"],
                "candlesticks": []
                if index == 2
                else [
                    {
                        "end_period_ts": spec["decision_ms"] // 1000,
                        "yes_bid": {price_field: "0.4900"},
                        "yes_ask": {price_field: "0.5100"},
                    }
                ],
            }
        ]
    archive = tmp_path / "candles"
    args = [
        "--history",
        str(history_path),
        "--historical-inventory",
        str(inventory_path),
        "--current-inventory",
        str(current_path),
        "--protocol",
        str(protocol),
        "--archive",
        str(archive),
        "--kalshi-url",
        api_server.url,
    ]
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    assert main([*args, "--mode", "capture", "--out", str(first)], stand_in_pin) == 0
    assert main([*args, "--mode", "replay", "--out", str(second)], stand_in_pin) == 0
    assert first.read_bytes() == second.read_bytes()
    report = read_document(first)
    assert report["native_contracts"] == 3
    assert report["selected_contracts"] == 3
    assert report["eligible_contracts"] == 2
    assert report["hypothetical_profit_units"] == 9200
    assert report["orders_submitted"] == 0
    coverage = tuple(narrow_json_to_dict(value) for value in require_list(report, "coverage"))
    assert tuple(row["quote_status"] for row in coverage) == (
        "eligible",
        "eligible",
        "missing_candle",
    )
    assert len(api_server.requests) == 3
    assert tuple(row["quote_partition"] for row in coverage) == (
        "current",
        "historical",
        "historical",
    )
    for output in (first, tmp_path / "sidecar.json"):
        run_record_sidecar(output).write_text("{}", "utf-8")
        with pytest.raises(FileExistsError, match="INFLATION-016"):
            main([*args, "--mode", "replay", "--out", str(output)], stand_in_pin)
    process_output = tmp_path / "process.json"
    command = [
        sys.executable,
        "-m",
        "scripts.inflation_experiment",
        *args,
        "--mode",
        "replay",
        "--out",
        str(process_output),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    assert completed.stdout == ""
    assert read_document(process_output)["decisions"] == report["decisions"]
    envelope = next(archive.glob("*.json"))
    document = read_document(envelope)
    document["body"] = "{}"
    envelope.write_text(dump_json_str(document), "utf-8")
    bad = tmp_path / "bad.json"
    command[-1] = str(bad)
    failed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
    assert failed.returncode == 1
    assert (
        failed.stderr.splitlines()[-1] == "ValueError: [MARKET-015] archived response hash mismatch"
    )
    assert not bad.exists()


def test_entry_requires_early_pinning() -> None:
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.inflation_experiment", run_name="__main__")
