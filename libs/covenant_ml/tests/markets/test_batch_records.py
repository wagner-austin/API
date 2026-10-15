"""Strict plan, account identity and immutable attempt validation without mocks."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, dump_json_str, require_dict, require_list

from covenant_ml.markets.batch_attempts import read_batch_attempts
from covenant_ml.markets.batch_records import (
    baseline_digest,
    decode_batch_plan,
    encode_batch_plan,
    require_batch_history,
)
from covenant_ml.markets.execution_policy import apply_execution_event
from covenant_ml.markets.execution_records import encode_intent
from covenant_ml.markets.execution_state import decode_execution_state, encode_execution_state
from covenant_ml.markets.execution_storage import persist_evidence

from ._batch_fixtures import baseline_state, batch_intent, batch_plan
from ._execution_fixtures import empty_state, intent


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"tickers": []}, "DEMO-BATCH-004"),
        ({"tickers": ["X"] * 2}, "DEMO-BATCH-004"),
        ({"tickers": [f"X{i}" for i in range(101)]}, "DEMO-BATCH-004"),
        ({"mode": "live"}, "DEMO-BATCH-004"),
        ({"limit_price": 10001}, "MARKET-001"),
        ({"expires_ms": 120000}, "MARKET-001"),
        ({"credential_fingerprint": "bad"}, "CAMPAIGN-001"),
        ({"initial_cash": 0}, "MARKET-001"),
    ],
)
def test_invalid_plans_are_rejected(changes: JSONObject, code: str) -> None:
    document = encode_batch_plan(batch_plan(baseline_state()))
    document.update(changes)
    with pytest.raises(ValueError, match=code):
        decode_batch_plan(document)


def test_plan_roundtrip_and_missing_or_uncredited_predecessor() -> None:
    state = baseline_state()
    plan = batch_plan(state)
    assert decode_batch_plan(encode_batch_plan(plan)) == plan
    require_batch_history(plan, state, "a" * 64)
    with pytest.raises(ValueError, match="DEMO-BATCH-005"):
        baseline_digest(empty_state())
    reserved = apply_execution_event(
        empty_state(), {"action": "reserve", "observed_ms": 1000, "data": encode_intent(intent())}
    )
    with pytest.raises(ValueError, match="DEMO-BATCH-005"):
        baseline_digest(reserved)
    with pytest.raises(ValueError, match="DEMO-BATCH-005"):
        require_batch_history(plan, state, "b" * 64)


@pytest.mark.parametrize(
    "changes",
    [
        {"baseline_digest": "b" * 64},
        {"tickers": ["MARKET-1"]},
        {"tickers": ["FOREIGN"]},
        {"limit_price": 499},
    ],
)
def test_changed_plan_cannot_accept_reserved_history(changes: JSONObject) -> None:
    state = baseline_state()
    plan = batch_plan(state)
    state = apply_execution_event(
        state, {"action": "reserve", "observed_ms": 120000, "data": encode_intent(batch_intent(2))}
    )
    require_batch_history(plan, state, "a" * 64)
    document = encode_batch_plan(plan)
    document.update(changes)
    with pytest.raises(ValueError, match="DEMO-BATCH-005"):
        require_batch_history(decode_batch_plan(document), state, "a" * 64)


def test_permanent_halt_refuses_batch_history() -> None:
    state = baseline_state()
    plan = batch_plan(state)
    halted = apply_execution_event(
        state, {"action": "halt", "observed_ms": 120000, "reason": "verified breach"}
    )
    with pytest.raises(ValueError, match="permanent policy breach"):
        require_batch_history(plan, halted, "a" * 64)


@pytest.mark.parametrize("repeat", [False, True])
def test_excess_or_duplicate_reserved_history_is_rejected(repeat: bool) -> None:
    state = baseline_state()
    plan = batch_plan(state, 1)
    reserved = apply_execution_event(
        state, {"action": "reserve", "observed_ms": 120000, "data": encode_intent(batch_intent(2))}
    )
    document = encode_execution_state(reserved)
    orders = require_list(document, "orders")
    if repeat:
        orders.append(orders[1])
    else:
        native_intent = require_dict(require_dict({"row": orders[1]}, "row"), "intent")
        native_intent.update(contracts=2, fee_reserve=9000)
    with pytest.raises(ValueError, match="DEMO-BATCH-005"):
        require_batch_history(plan, decode_execution_state(document), "a" * 64)


@pytest.mark.parametrize(
    "changes",
    [
        {"ticker": "FOREIGN"},
        {"contracts": 2, "fee_reserve": 9000},
        {"limit_price": 499},
        {"fee_reserve": 9400},
        {"created_ms": 119999, "expires_ms": 419999},
        {"expires_ms": 419999},
    ],
)
def test_attempt_manifest_cannot_drift(tmp_path: Path, changes: JSONObject) -> None:
    state = baseline_state()
    plan = batch_plan(state)
    path = tmp_path / "attempts" / "001"
    path.mkdir(parents=True)
    document = encode_intent(batch_intent(2))
    document.update(changes)
    persist_evidence(path / "intent.json", document)
    with pytest.raises(ValueError, match="DEMO-BATCH-006"):
        read_batch_attempts(tmp_path, plan, state)


def test_sequence_identity_and_reservations_are_bound_to_saved_attempts(tmp_path: Path) -> None:
    state = baseline_state()
    plan = batch_plan(state)
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    assert read_batch_attempts(tmp_path, plan, state) == ()
    first = attempts / "001"
    first.mkdir()
    with pytest.raises(FileNotFoundError):
        read_batch_attempts(tmp_path, plan, state)
    row = batch_intent(2)
    persist_evidence(first / "intent.json", encode_intent(row))
    assert read_batch_attempts(tmp_path, plan, state) == (row,)
    reserved = apply_execution_event(
        state, {"action": "reserve", "observed_ms": 120000, "data": encode_intent(row)}
    )
    assert read_batch_attempts(tmp_path, plan, reserved) == (row,)
    first.rename(attempts / "002")
    with pytest.raises(ValueError, match="sequence"):
        read_batch_attempts(tmp_path, plan, reserved)
    (attempts / "002").rename(first)
    document = encode_intent(row)
    document["client_order_id"] = batch_intent(3)["client_order_id"]
    (first / "intent.json").write_text(dump_json_str(document), encoding="utf-8")
    with pytest.raises(ValueError, match="persisted attempts"):
        read_batch_attempts(tmp_path, plan, reserved)
    second = attempts / "002"
    second.mkdir()
    persist_evidence(second / "intent.json", encode_intent(batch_intent(3)))
    with pytest.raises(ValueError, match="distinct persisted attempts"):
        read_batch_attempts(tmp_path, plan, state)
    with pytest.raises(ValueError, match="sequence"):
        read_batch_attempts(tmp_path, batch_plan(state, 1), state)
