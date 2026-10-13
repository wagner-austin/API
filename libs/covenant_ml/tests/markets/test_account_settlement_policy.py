"""Account credits enter the real journal reducer exactly once and survive replay."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, require_dict

from covenant_ml.markets.account_settlement import (
    decode_native_account_settlement,
    encode_account_settlement,
)
from covenant_ml.markets.execution_journal import (
    append_execution_event,
    initialize_execution_journal,
    read_execution_journal,
)
from covenant_ml.markets.execution_policy import apply_execution_event
from covenant_ml.markets.execution_records import decode_native_order, encode_intent, encode_receipt
from covenant_ml.markets.execution_state import (
    decode_execution_state,
    encode_execution_state,
    execution_report,
)

from ._account_settlement_fixtures import final_market, native_credit
from ._execution_fixtures import intent, native_order, settlement


def events() -> tuple[JSONObject, JSONObject, JSONObject]:
    row = intent()
    receipt = decode_native_order(native_order(row))
    credit = decode_native_account_settlement(native_credit(), final_market(), receipt, 120000)
    return (
        {"action": "reserve", "observed_ms": 1000, "data": encode_intent(row)},
        {
            "action": "receipt",
            "observed_ms": 1000,
            "client_order_id": row["client_order_id"],
            "data": encode_receipt(receipt),
        },
        {
            "action": "account_settlement",
            "observed_ms": 120000,
            "client_order_id": row["client_order_id"],
            "data": encode_account_settlement(credit),
        },
    )


def test_persisted_credit_and_repeated_replay(tmp_path: Path) -> None:
    path = tmp_path / "journal.sqlite"
    initialize_execution_journal(path)
    reserve, receipt, credit = events()
    for event in (reserve, receipt, credit):
        append_execution_event(path, event)
    state = read_execution_journal(path)
    assert decode_execution_state(encode_execution_state(state)) == state
    assert execution_report(state)["exposure"] == 0
    assert execution_report(state)["net_realized_profit"] == 0
    assert append_execution_event(path, credit) == state
    repeated = {**receipt, "observed_ms": 120000}
    assert append_execution_event(path, repeated) == state
    alternate: JSONObject = {
        "action": "settlement",
        "observed_ms": 120000,
        "client_order_id": intent()["client_order_id"],
        "data": settlement(intent()),
    }
    with pytest.raises(ValueError, match="EXECUTION-006"):
        append_execution_event(path, alternate)
    assert read_execution_journal(path) == state


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "resting",
        "public",
        "ticker",
        "quantity",
        "cost",
        "fees",
        "early",
        "future",
        "changed",
    ],
)
def test_contradictory_credit_never_mutates_journal(tmp_path: Path, failure: str) -> None:
    path = tmp_path / "journal.sqlite"
    initialize_execution_journal(path)
    reserve, receipt, credit = events()
    append_execution_event(path, reserve)
    if failure == "resting":
        require_dict(receipt, "data").update(status="resting", filled_count=50, remaining_count=50)
    if failure != "missing":
        append_execution_event(path, receipt)
    data = require_dict(credit, "data")
    if failure == "public":
        append_execution_event(
            path,
            {
                "action": "settlement",
                "observed_ms": 1001,
                "client_order_id": intent()["client_order_id"],
                "data": settlement(intent()),
            },
        )
    elif failure == "ticker":
        data["ticker"] = "OTHER"
    elif failure == "quantity":
        data.update(quantity=50, revenue=5000)
    elif failure in ("cost", "fees"):
        data[failure] = 1
    elif failure == "early":
        data["settled_ms"] = 999
    elif failure == "future":
        data["settled_ms"] = 120001
    elif failure == "changed":
        append_execution_event(path, credit)
        data.update(payout=0, revenue=0)
    before = read_execution_journal(path)
    with pytest.raises(ValueError, match="EXECUTION-011"):
        apply_execution_event(before, credit)
    with pytest.raises(ValueError, match="EXECUTION-011"):
        append_execution_event(path, credit)
    assert read_execution_journal(path) == before
