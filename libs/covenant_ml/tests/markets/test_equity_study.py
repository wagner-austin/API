"""Frozen ranking, calendar and provenance are verified before acquisition."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.config import config_test_hooks
from platform_core.json_utils import JSONObject, JSONValue, narrow_json_to_dict, require_list

from covenant_ml.markets.equity_study import (
    decode_equity_study,
    encode_equity_study,
    load_equity_study,
)
from covenant_ml.markets.provenance import market_fingerprint
from covenant_ml.markets.reporting import write_market_report
from tests.benchmarking._cli_fixtures import stand_in_pin


def study_document() -> JSONObject:
    """Return an explicit two-stock historical arithmetic design."""
    return {
        "snapshot_sha256": "a" * 64,
        "decision_at": "2020-08-28T00:00:00Z",
        "cohort": ["AAPL", "MSFT"],
        "selected": ["AAPL"],
        "expected_days": ["2020-08-28", "2020-08-31"],
        "costs": [0.002, 0.005],
    }


def snapshot_document() -> JSONObject:
    """Create a fully specified test snapshot, without inventing source prices."""
    rows: list[JSONValue] = []
    for index, symbol in enumerate(("AAPL", "MSFT")):
        rows.append(
            {
                "signal": {
                    "symbol": symbol,
                    "cik": index + 1,
                    "current_ratio": 2.0 - index,
                    "annual_roa": 0.2 - index / 10,
                },
                "liquidity_rank": 1 - index,
                "profitability_rank": 1 - index,
                "score": 2 - index * 2,
                "selected": index == 0,
                "portfolio_weight": 1.0 - index,
                "benchmark_weight": 0.5,
            }
        )
    return {
        "rows": rows,
        "selected_count": 1,
        "cohort_count": 2,
        "decision_at": "2020-08-28T00:00:00Z",
        "completed_at": "2020-08-27T00:00:00Z",
        "orders_submitted": 0,
        "advance_to_funded_trading": False,
        "round_trip_cost": 0.002,
        "stress_round_trip_cost": 0.005,
        "entry_date": "2020-08-28",
        "exit_date": "2020-08-31",
        "holding_bars": 1,
    }


def persist_snapshot(path: Path, document: JSONObject) -> JSONObject:
    """Persist the real report contract and return its hash-pinned calendar."""
    write_market_report(
        document,
        market_fingerprint(stand_in_pin(), config_test_hooks.get_env),
        path,
        (),
        "equity-test",
        "historical-arithmetic",
        (),
    )
    return {
        "snapshot_sha256": sha256(path.read_bytes()).hexdigest(),
        "expected_days": ["2020-08-28", "2020-08-31"],
    }


def test_study_roundtrip_and_verified_snapshot(tmp_path: Path) -> None:
    document = study_document()
    assert encode_equity_study(decode_equity_study(document)) == document
    path = tmp_path / "snapshot.json"
    calendar = persist_snapshot(path, snapshot_document())
    study, sources = load_equity_study(path, calendar)
    expected = dict(document, snapshot_sha256=calendar["snapshot_sha256"])
    assert encode_equity_study(study) == expected
    assert sources == (path, path.with_suffix(".json.runrecord.json"))


@pytest.mark.parametrize(
    "key,value",
    [
        ("snapshot_sha256", "short"),
        ("snapshot_sha256", "z" * 64),
        ("costs", []),
        ("costs", [0.1, 0.1]),
        ("costs", [float("inf")]),
        ("costs", [-0.1]),
        ("costs", [1.1]),
        ("decision_at", "2020-08-30T00:00:00Z"),
    ],
)
def test_invalid_study_fields(key: str, value: JSONValue) -> None:
    with pytest.raises(ValueError, match="EQUITY-016"):
        decode_equity_study(dict(study_document(), **{key: value}))


@pytest.mark.parametrize(
    "key,value",
    [
        ("cohort_count", 3),
        ("selected_count", 0),
        ("selected_count", 3),
        ("completed_at", "2020-08-28T00:00:00Z"),
        ("orders_submitted", 1),
        ("advance_to_funded_trading", True),
        ("entry_date", "2020-08-27"),
        ("exit_date", "2020-09-01"),
        ("holding_bars", 2),
    ],
)
def test_invalid_snapshot_fields(tmp_path: Path, key: str, value: JSONValue) -> None:
    path = tmp_path / "snapshot.json"
    calendar = persist_snapshot(path, dict(snapshot_document(), **{key: value}))
    with pytest.raises(ValueError, match="EQUITY-017"):
        load_equity_study(path, calendar)


@pytest.mark.parametrize(
    "key,value",
    [
        ("signal", {"symbol": "AAPL", "cik": 1, "current_ratio": 0.0, "annual_roa": 0.0}),
        ("liquidity_rank", 0),
        ("profitability_rank", 0),
        ("score", 0),
        ("selected", False),
        ("portfolio_weight", 0.5),
        ("benchmark_weight", 1.0),
    ],
)
def test_changed_ranking_or_weights(tmp_path: Path, key: str, value: JSONValue) -> None:
    document = snapshot_document()
    narrow_json_to_dict(require_list(document, "rows")[0])[key] = value
    path = tmp_path / "snapshot.json"
    calendar = persist_snapshot(path, document)
    with pytest.raises(ValueError, match="EQUITY-017"):
        load_equity_study(path, calendar)


def test_different_digest_and_changed_source_bytes(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    calendar = persist_snapshot(path, snapshot_document())
    with pytest.raises(ValueError, match="EQUITY-017"):
        load_equity_study(path, dict(calendar, snapshot_sha256="b" * 64))
    path.write_text(path.read_text("utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="EQUITY-017"):
        load_equity_study(path, calendar)
