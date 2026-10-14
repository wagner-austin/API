"""Complete comparison through real HTTP capture, offline replay and subprocesses."""

from __future__ import annotations

import runpy
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import dump_json_str, narrow_json_to_dict, require_list
from platform_core.run_record import run_record_sidecar
from scripts.sports_cross_venue import main

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.archive import ResponseArchive
from covenant_ml.markets.sports_history import historical_quote, historical_selection
from covenant_ml.markets.validation import read_document
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._rating_fixtures import game
from ._server import ApiServer, api_server
from ._strategy_fixtures import candle_payload, observation
from ._us_history_fixtures import history, mapping, us_game

__all__ = ["api_server"]


def test_real_capture_replay_and_failure(api_server: ApiServer, tmp_path: Path) -> None:
    stamps = ("26JAN01", "26MAR01", "26MAR02", "26MAR03", "26MAR04")
    days = tuple(observation(stamp)[0]["day"] for stamp in stamps)
    api_server.documents["/historical/markets"] = [
        {
            "markets": [row for stamp in stamps for row in game(stamp)],
            "cursor": "",
        }
    ]
    for stamp in stamps:
        payload = candle_payload(stamp)
        if stamp == "26MAR04":
            payload["candlesticks"] = []
        api_server.documents[
            "/historical/markets/" + observation(stamp)[0]["ticker"] + "/candlesticks"
        ] = [payload]
    k_root = tmp_path / "kalshi"
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        inventory, candles = (
            ResponseArchive(
                k_root / name, api_server.url, client, _test_hooks.clock, _test_hooks.pause
            )
            for name in ("inventory", "candles")
        )
        specs, _ = historical_selection(inventory.capture)
        for spec in specs:
            historical_quote(spec, candles.capture)
    api_server.documents["/v1/markets"] = [
        {"markets": [us_game(day) for day in days if day != "2026-03-03"]},
        {"markets": []},
    ]
    api_server.documents["/v1/price-history"] = [
        history(day) for day in days if day != "2026-03-03"
    ]
    team_map = tmp_path / "teams.json"
    team_map.write_text(dump_json_str(mapping()), "utf-8")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Fixed midpoint; no fit; one-minute freshness.\n", "utf-8")
    u_root = tmp_path / "us"
    args = [
        "--kalshi-archive",
        str(k_root),
        "--us-archive",
        str(u_root),
        "--kalshi-url",
        api_server.url,
        "--us-url",
        api_server.url,
        "--team-map",
        str(team_map),
        "--protocol",
        str(protocol),
    ]
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    assert (
        main(
            [
                *args,
                "--inventory-mode",
                "capture",
                "--history-mode",
                "capture",
                "--out",
                str(first),
            ],
            stand_in_pin,
        )
        == 0
    )
    replay = [*args, "--inventory-mode", "replay", "--history-mode", "replay"]
    assert main([*replay, "--out", str(second)], stand_in_pin) == 0
    assert first.read_bytes() == second.read_bytes()
    report = read_document(first)
    assert report["training_games"] == 1
    assert report["evaluation_games"] == 2
    assert report["trades"] == 2
    assert report["orders_submitted"] == 0
    assert report["hypothetical_profit_units"] == -6800
    coverage = tuple(narrow_json_to_dict(value) for value in require_list(report, "coverage"))
    assert tuple(row["paired_eligible"] for row in coverage) == (True, True, True, False, False)
    assert coverage[3] == {
        "ticker": specs[3]["ticker"],
        "status": "unmatched_game",
        "kalshi_quote_status": "eligible",
        "paired_eligible": False,
    }
    assert coverage[4]["kalshi_quote_status"] == "missing_candle"
    assert len(api_server.requests) == 12
    for output in (first, tmp_path / "sidecar.json"):
        run_record_sidecar(output).write_text("{}", "utf-8")
        with pytest.raises(FileExistsError, match="CROSS-009"):
            main([*replay, "--out", str(output)], stand_in_pin)
    third = tmp_path / "process.json"
    command = [sys.executable, "-m", "scripts.sports_cross_venue", *replay, "--out", str(third)]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    assert completed.stdout == f"Report: {third}\n"
    assert read_document(third)["decisions"] == report["decisions"]
    envelope = next((u_root / "history").glob("*.json"))
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
        runpy.run_module("scripts.sports_cross_venue", run_name="__main__")
