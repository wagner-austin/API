"""Adversarial native account evidence must fail before any journal transition."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, JSONValue, require_dict, require_list

from covenant_ml.markets.batch_account import account_snapshot_event
from covenant_ml.markets.execution_journal import execution_journal
from covenant_ml.markets.execution_policy import apply_execution_event
from covenant_ml.markets.execution_records import encode_intent
from covenant_ml.markets.execution_state import (
    ExecutionState,
    decode_execution_state,
    encode_execution_state,
)

from ._batch_fixtures import account_document, baseline_state, batch_intent
from ._execution_fixtures import empty_state, intent, settlement


def replace_field(
    document: JSONObject, section: str, index: int, key: str, value: JSONValue
) -> None:
    """Alter one native record using the same strict JSON validators as production."""
    row = require_dict({"row": require_list(document, section)[index]}, "row")
    row[key] = value


@pytest.mark.parametrize(
    ("section", "index", "key", "value", "code"),
    [
        ("orders", 1, "client_order_id", "00000000-0000-0000-0000-000000000009", "DEMO-BATCH-001"),
        (
            "orders",
            1,
            "order_id",
            "exchange-00000000-0000-0000-0000-000000000001",
            "DEMO-BATCH-001",
        ),
        ("orders", 1, "ticker", "MARKET-1", "DEMO-BATCH-001"),
        ("fills", 1, "fill_id", "fill-00000000-0000-0000-0000-000000000001", "DEMO-BATCH-001"),
        ("fills", 1, "order_id", "unknown", "DEMO-BATCH-001"),
        ("fills", 1, "fee_cost", "0.010001", "DEMO-TRIAL-003"),
        ("settlements", 0, "ticker", "unknown", "DEMO-BATCH-001"),
        ("markets", 0, "ticker", "unknown", "DEMO-BATCH-001"),
        ("positions", 0, "position_fp", "0.99", "DEMO-BATCH-003"),
    ],
)
def test_corrupt_account_field_leaves_sqlite_unchanged(
    tmp_path: Path, section: str, index: int, key: str, value: JSONValue, code: str
) -> None:
    first, second = intent(1, 120000), batch_intent(2)
    journal = tmp_path / "execution.sqlite"
    execution_journal().initialize(journal)
    execution_journal().transact(
        journal,
        lambda _: {"action": "reserve", "observed_ms": 120000, "data": encode_intent(first)},
    )
    state = execution_journal().read(journal)
    execution_journal().transact(
        journal,
        lambda _: account_snapshot_event(account_document((first,), (100,)), state, 20000099),
    )
    before = execution_journal().transact(
        journal,
        lambda _: {"action": "reserve", "observed_ms": 120000, "data": encode_intent(second)},
    )
    document = account_document((first, second), (100, 100))
    replace_field(document, section, index, key, value)
    with pytest.raises(ValueError, match=code):
        execution_journal().transact(
            journal, lambda _: account_snapshot_event(document, before, 20000099)
        )
    assert execution_journal().read(journal) == before


@pytest.mark.parametrize("field", ["orders", "fills", "settlements", "markets", "positions"])
def test_duplicate_native_records_are_not_silently_collapsed(field: str) -> None:
    state = baseline_state()
    second = batch_intent(2)
    state = apply_execution_event(
        state, {"action": "reserve", "observed_ms": 120000, "data": encode_intent(second)}
    )
    document = account_document((state["orders"][0]["intent"], second), (100, 100))
    rows = require_list(document, field)
    rows.append(rows[0])
    with pytest.raises(ValueError, match=r"DEMO-BATCH-00[13]"):
        account_snapshot_event(document, state, 20000099)


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"balance_dollars": "20.010099", "balance": 2001}, "DEMO-BATCH-002"),
        ({"balance": 2001}, "KALSHI-DEMO-006"),
        ({"portfolio_value": 1}, "DEMO-BATCH-003"),
        ({"updated_ts": 121}, "MARKET-001"),
    ],
)
def test_changed_cash_or_invalid_balance_is_rejected(changes: JSONObject, code: str) -> None:
    state = baseline_state()
    document = account_document((state["orders"][0]["intent"],), (100,))
    require_dict(document, "last_balance").update(changes)
    with pytest.raises(ValueError, match=code):
        account_snapshot_event(document, state, 20000099)


def test_disappeared_credit_stable_wrong_cash_empty_history_and_resting_order() -> None:
    state = baseline_state()
    row = state["orders"][0]["intent"]
    document = account_document((row,), (100,))
    document.update(settlements=[], markets=[])
    with pytest.raises(ValueError, match="previously credited"):
        account_snapshot_event(document, state, 20000099)
    document = account_document((row,), (100,))
    with pytest.raises(ValueError, match="cash or positions"):
        account_snapshot_event(document, state, 20000000)
    with pytest.raises(ValueError, match="DEMO-BATCH-001"):
        account_snapshot_event(account_document((), ()), empty_state(), 20000099)
    replace_field(document, "orders", 0, "status", "resting")
    replace_field(document, "orders", 0, "fill_count_fp", "0.00")
    replace_field(document, "orders", 0, "remaining_count_fp", "1.00")
    replace_field(document, "orders", 0, "taker_fill_cost_dollars", "0.0000")
    replace_field(document, "orders", 0, "taker_fees_dollars", "0.0000")
    with pytest.raises(ValueError, match="terminal orders"):
        account_snapshot_event(document, state, 20000099)


def test_public_only_settlement_cannot_substitute_for_account_credit() -> None:
    state = baseline_state()
    document = encode_execution_state(state)
    order = require_dict({"row": require_list(document, "orders")[0]}, "row")
    order["account_settlement"] = None
    order["settlement"] = settlement(state["orders"][0]["intent"], 10000)
    altered: ExecutionState = decode_execution_state(document)
    with pytest.raises(ValueError, match="authenticated credits"):
        account_snapshot_event(
            account_document((state["orders"][0]["intent"],), (100,)), altered, 20000099
        )
