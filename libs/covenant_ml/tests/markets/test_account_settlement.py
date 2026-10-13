"""Exact account credits require independent final market and acquisition evidence."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONValue

from covenant_ml.markets.account_settlement import (
    decode_account_settlement,
    decode_native_account_settlement,
    encode_account_settlement,
)
from covenant_ml.markets.execution_records import decode_native_order

from ._account_settlement_fixtures import final_market, native_credit
from ._execution_fixtures import intent, native_order


@pytest.mark.parametrize("payout", [0, 37, 100])
def test_native_credit_roundtrip(payout: int) -> None:
    receipt = decode_native_order(native_order(intent()))
    credit = decode_native_account_settlement(
        native_credit(payout=payout), final_market(payout=payout), receipt, 120000
    )
    assert encode_account_settlement(credit) == {
        "ticker": "MARKET-1",
        "quantity": 100,
        "cost": 9800,
        "fees": 200,
        "revenue": payout * 100,
        "payout": payout * 100,
        "settled_ms": 120000,
    }
    assert decode_account_settlement(encode_account_settlement(credit)) == credit


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("credit", "ticker", "OTHER"),
        ("market", "ticker", "OTHER"),
        ("credit", "event_ticker", "OTHER"),
        ("market", "status", "determined"),
        ("market", "result", "no"),
        ("market", "settlement_value_dollars", "0.9900"),
        ("credit", "settled_time", "1970-01-01T00:02:01Z"),
        ("market", "settlement_ts", "1970-01-01T00:00:00Z"),
        ("market", "settlement_ts", "1970-01-01T00:02:01Z"),
    ],
)
def test_inconsistent_final_evidence(target: str, field: str, value: JSONValue) -> None:
    credit, market = native_credit(), final_market()
    (credit if target == "credit" else market)[field] = value
    with pytest.raises(ValueError, match="ACCOUNT-SETTLEMENT-002"):
        decode_native_account_settlement(
            credit, market, decode_native_order(native_order(intent())), 120000
        )


@pytest.mark.parametrize("result", ["unknown", "no"])
def test_unsupported_or_contradictory_result(result: str) -> None:
    credit, market = native_credit(), final_market()
    credit["market_result"] = market["result"] = result
    with pytest.raises(ValueError, match="ACCOUNT-SETTLEMENT-002"):
        decode_native_account_settlement(
            credit, market, decode_native_order(native_order(intent())), 120000
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("no_count_fp", "0.01"),
        ("no_total_cost_dollars", "0.0001"),
        ("yes_total_cost_dollars", "0.9700"),
        ("fee_cost", "0.0199"),
    ],
)
def test_acquisition_mismatch(field: str, value: str) -> None:
    credit = native_credit()
    credit[field] = value
    with pytest.raises(ValueError, match="ACCOUNT-SETTLEMENT-003"):
        decode_native_account_settlement(
            credit, final_market(), decode_native_order(native_order(intent())), 120000
        )


def test_credit_arithmetic_never_rounds() -> None:
    receipt = decode_native_order(native_order(intent()))
    native = native_credit()
    native["revenue"] = 99
    with pytest.raises(ValueError, match="ACCOUNT-SETTLEMENT-001"):
        decode_native_account_settlement(native, final_market(), receipt, 120000)
    row = encode_account_settlement(
        decode_native_account_settlement(native_credit(), final_market(), receipt, 120000)
    )
    row.update(quantity=1, payout=3333, revenue=33)
    with pytest.raises(ValueError, match="ACCOUNT-SETTLEMENT-001"):
        decode_account_settlement(row)
    row["quantity"] = True
    with pytest.raises(TypeError):
        decode_account_settlement(row)
