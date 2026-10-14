"""Native contracts must agree with the declared release and settlement policy."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONValue, narrow_json_to_dict

from covenant_ml.markets.inflation_records import InflationMonth
from covenant_ml.markets.inflation_selection import (
    inflation_inventory,
    inflation_selection,
    inflation_target,
)

from ._inflation_experiment_fixtures import histories, market


def test_inventory_and_eligible_selection() -> None:
    first: JSONObject = {"markets": [market()], "cursor": ""}
    second: JSONObject = {"markets": [market(), market(2)], "cursor": ""}
    records = inflation_inventory((first, second))
    assert len(records) == 2
    selected, coverage = inflation_selection(records, histories())
    assert len(selected) == 2
    assert selected[0][1] == 60.5 / 61
    assert sorted(row[0]["day"] for row in selected) == ["2024-02-13", "2024-03-13"]
    assert [narrow_json_to_dict(value)["status"] for value in coverage] == ["eligible", "eligible"]
    assert inflation_target("CPIYOY-24JAN") == "2024-01"


@pytest.mark.parametrize("event", ["bad", "KXCPIYOY-24BAD"])
def test_bad_events(event: str) -> None:
    with pytest.raises(ValueError, match="INFLATION-011"):
        inflation_target(event)


@pytest.mark.parametrize("kind", ["cursor", "duplicate", "ticker", "overlap"])
def test_inventory_conflicts(kind: str) -> None:
    native = market()
    first: JSONObject = {"markets": [native], "cursor": "next" if kind == "cursor" else ""}
    second = market()
    if kind == "duplicate":
        first["markets"] = [native, native]
    elif kind == "ticker":
        native["ticker"] = "wrong"
    elif kind == "overlap":
        second["close_time"] = "2024-02-14T13:29:00Z"
    with pytest.raises(ValueError, match="INFLATION-012"):
        inflation_inventory((first, {"markets": [second], "cursor": ""}))


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("strike_type", "less", "unsupported_rules"),
        ("rules_primary", "wrong", "unsupported_rules"),
        ("status", "active", "not_binary_final"),
        ("result", "scalar", "not_binary_final"),
        ("close_time", "2024-02-14T13:29:00Z", "release_close_mismatch"),
        ("close_time", "2024-02-13T00:00:00Z", "release_close_mismatch"),
        ("open_time", "2024-02-13T00:01:00Z", "not_open_at_cutoff"),
    ],
)
def test_contract_exclusions(field: str, value: JSONValue, reason: str) -> None:
    native = market()
    native[field] = value
    selected, coverage = inflation_selection((native,), histories())
    assert selected == ()
    assert narrow_json_to_dict(coverage[0])["status"] == reason


@pytest.mark.parametrize("kind", ["strike", "ticker", "payout"])
def test_contract_inconsistency(kind: str) -> None:
    native = market()
    native[
        {"strike": "floor_strike", "ticker": "ticker", "payout": "settlement_value_dollars"}[kind]
    ] = 2.65 if kind == "strike" else "wrong" if kind == "ticker" else "0.0000"
    with pytest.raises(ValueError, match="INFLATION-013"):
        inflation_selection((native,), histories())


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("absent", "missing_target_history"),
        ("release", "missing_release"),
        ("forecast", "missing_forecast"),
        ("training", "insufficient_prior_errors"),
        ("settlement", "release_settlement_mismatch"),
        ("outside", "outside_evaluation"),
    ],
)
def test_history_exclusions(kind: str, reason: str) -> None:
    history = histories()
    native = market()
    current = history[-3]
    if kind == "absent":
        history = ()
    elif kind == "training":
        history = (current,)
    elif kind == "outside":
        native["event_ticker"] = "KXCPIYOY-23JAN"
    else:
        changed: InflationMonth = {
            "month": current["month"],
            "forecasts": () if kind == "forecast" else current["forecasts"],
            "actual": None
            if kind == "release"
            else {"day": "2024-02-13", "value": 2.5}
            if kind == "settlement"
            else current["actual"],
        }
        history = (*history[:-3], changed)
    selected, coverage = inflation_selection((native,), history)
    assert selected == ()
    assert narrow_json_to_dict(coverage[0])["status"] == reason
