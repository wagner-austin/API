"""Full experiment CLI, verified source hashes and reproducible economic output."""

from __future__ import annotations

import runpy
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
    require_list,
)
from platform_core.run_record import decode_run_record
from scripts.market_research import main

from covenant_ml.markets.codecs import encode_rule
from covenant_ml.markets.experiment import decode_plan, encode_plan, evaluate_panel, write_report
from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
from covenant_ml.markets.runner import AcquisitionMode, build_collector, collect_panel
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._fixtures import history, rule
from ._server import ApiServer, api_server
from .test_acquisition import SERIES, TICKER, candle_document, market_document, trade_document

__all__ = ["api_server"]


def plan_document() -> JSONObject:
    return {
        "name": "test-replication",
        "series": [SERIES],
        "timezone": "America/New_York",
        "start": "2026-01-01",
        "end": "2026-01-14",
        "rule": encode_rule(rule()),
        "friction": 100,
        "protocol_digest": sha256(b"frozen test protocol").hexdigest(),
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("series", []),
        ("series", [SERIES, SERIES]),
        ("end", "2025-01-01"),
        ("protocol_digest", "x" * 64),
        ("protocol_digest", "abc"),
    ],
)
def test_plan_refuses_invalid_contract(key: str, value: JSONValue) -> None:
    document = plan_document()
    document[key] = value
    with pytest.raises(ValueError, match="MARKET-025"):
        decode_plan(document)


def test_plan_roundtrip_frozen_sensitivities_and_panel_refusals() -> None:
    plan = decode_plan(plan_document())
    assert decode_plan(encode_plan(plan)) == plan
    histories = [
        history(series=SERIES, ticker="a"),
        history(series=SERIES, ticker="b", event_day="2026-01-02"),
    ]
    report = evaluate_panel(histories, plan)
    arms = [narrow_json_to_dict(row) for row in require_list(report, "results")]
    assert [row["through"] for row in arms] == [200, 100, 0]
    assert [row["candidate_gate"] for row in arms] == [True, True, True]
    assert require_dict(arms[0], "summary")["pnl"] == 1800
    with pytest.raises(ValueError, match="MARKET-026"):
        evaluate_panel([history()], plan)
    document = plan_document()
    document["rule"] = encode_rule(rule(through=0))
    with pytest.raises(ValueError, match="MARKET-026"):
        evaluate_panel(histories, decode_plan(document))


def test_report_hash_covers_every_byte_and_refuses_existing_sidecar(tmp_path: Path) -> None:
    plan = decode_plan(plan_document())
    report = evaluate_panel([history(series=SERIES)], plan)
    fingerprint = market_fingerprint(stand_in_pin(), config_test_hooks.get_env)
    output = tmp_path / "result.json"
    sidecar = write_report(report, plan, fingerprint, output, [])
    record = decode_run_record(narrow_json_to_dict(load_json_str(sidecar.read_text("utf-8"))))
    assert record["payload_digest"] == sha256(output.read_bytes()).hexdigest()
    assert record["observations"] == tuple(
        {"name": f"through_{threshold}_pnl", "value": 0.09} for threshold in (0, 100, 200)
    )
    with pytest.raises(FileExistsError, match="MARKET-027"):
        write_report(report, plan, fingerprint, output, [])
    output.unlink()
    with pytest.raises(FileExistsError, match="MARKET-027"):
        write_report(report, plan, fingerprint, output, [])


def test_complete_cli_capture_resume_and_replay_match_actual_http_evidence(
    tmp_path: Path,
    api_server: ApiServer,
) -> None:
    api_server.documents["/historical/cutoff"] = [
        {"market_settled_ts": "2025-01-01T00:00:00Z", "trades_created_ts": "2025-01-01T00:00:00Z"}
    ]
    api_server.documents["/series/" + SERIES] = [
        {"series": {"ticker": SERIES, "fee_type": "quadratic", "fee_multiplier": 1}}
    ]
    second = market_document()
    other_ticker = SERIES + "-26JAN02-T41"
    second["ticker"] = other_ticker
    api_server.documents["/markets"] = [{"markets": [market_document(), second], "cursor": ""}]
    api_server.documents["/historical/markets"] = [{"markets": [], "cursor": ""}]
    api_server.documents["/series/" + SERIES + "/markets/" + TICKER + "/candlesticks"] = [
        {
            "ticker": TICKER,
            "candlesticks": [
                dict(candle_document(), end_period_ts=1767301200 - offset * 3600)
                for offset in range(6)
            ],
        }
    ]
    api_server.documents["/series/" + SERIES + "/markets/" + other_ticker + "/candlesticks"] = [
        {"ticker": other_ticker, "candlesticks": []}
    ]
    api_server.documents["/markets/trades"] = [
        {"trades": [trade_document()], "cursor": ""},
        {"trades": [], "cursor": ""},
    ]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(dump_json_str(plan_document()), "utf-8")
    protocol_path = tmp_path / "protocol.txt"
    protocol_path.write_bytes(b"frozen test protocol")
    results: list[JSONValue] = []
    for mode in ("resume", "resume", "replay"):
        output = tmp_path / f"result-{len(results)}.json"
        arguments = [
            "--plan",
            str(plan_path),
            "--protocol",
            str(protocol_path),
            "--archive",
            str(tmp_path / "raw"),
            "--out",
            str(output),
            "--workers",
            "1",
            "--mode",
            mode,
            "--base-url",
            api_server.url,
        ]
        assert main(arguments, stand_in_pin) == 0
        report = narrow_json_to_dict(load_json_str(output.read_text("utf-8")))
        assert report["implementation_sha256"] == implementation_digest()
        entries = require_list(require_dict(report, "pre_entry_regimes"), "contracts")
        assert [narrow_json_to_dict(row)["regime"] for row in entries] == [
            "low_flat",
            "unclassified",
        ]
        for value in require_list(report, "results"):
            groups = require_dict(narrow_json_to_dict(value), "by_regime")
            assert require_dict(groups, "low_flat")["markets"] == 1
            assert require_dict(groups, "unclassified")["markets"] == 1
        results.append(report["results"])
        assert len(require_list(report, "sources")) == 9
    assert results[0] == results[1] == results[2]
    assert len(api_server.requests) == 7
    protocol_path.write_bytes(b"changed after freezing")
    with pytest.raises(ValueError, match="MARKET-031"):
        main(arguments, stand_in_pin)
    collector, client = build_collector(tmp_path / "raw", api_server.url, AcquisitionMode.CAPTURE)
    with pytest.raises(ValueError, match="MARKET-028"):
        collect_panel(decode_plan(plan_document()), collector, 0, print)
    with pytest.raises(FileExistsError, match="MARKET-016"):
        collect_panel(decode_plan(plan_document()), collector, 1, print)
    client.close()


def test_module_entrypoint_refuses_late_pin_instead_of_claiming_false_determinism() -> None:
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.market_research", run_name="__main__")
