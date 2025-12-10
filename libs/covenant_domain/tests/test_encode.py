"""Tests for covenant_domain.encode module."""

from __future__ import annotations

from covenant_domain.encode import (
    encode_covenant,
    encode_covenant_id,
    encode_covenant_result,
    encode_deal,
    encode_deal_id,
    encode_measurement,
)
from covenant_domain.models import (
    Covenant,
    CovenantId,
    CovenantResult,
    Deal,
    DealId,
    Measurement,
)


def test_encode_deal_id() -> None:
    deal_id = DealId(value="deal-uuid-123")
    result = encode_deal_id(deal_id)
    assert result == {"value": "deal-uuid-123"}


def test_encode_deal() -> None:
    deal = Deal(
        id=DealId(value="deal-uuid-123"),
        name="Acme Corp Loan",
        borrower="Acme Corporation",
        sector="Technology",
        region="North America",
        commitment_amount_cents=100_000_000_00,
        currency="USD",
        maturity_date_iso="2028-12-31",
    )
    result = encode_deal(deal)
    assert result["id"] == {"value": "deal-uuid-123"}
    assert result["name"] == "Acme Corp Loan"
    assert result["borrower"] == "Acme Corporation"
    assert result["sector"] == "Technology"
    assert result["region"] == "North America"
    assert result["commitment_amount_cents"] == 100_000_000_00
    assert result["currency"] == "USD"
    assert result["maturity_date_iso"] == "2028-12-31"


def test_encode_covenant_id() -> None:
    covenant_id = CovenantId(value="cov-uuid-456")
    result = encode_covenant_id(covenant_id)
    assert result == {"value": "cov-uuid-456"}


def test_encode_covenant() -> None:
    covenant = Covenant(
        id=CovenantId(value="cov-uuid-456"),
        deal_id=DealId(value="deal-uuid-123"),
        name="Debt to EBITDA",
        formula="total_debt / ebitda",
        threshold_value_scaled=3_500_000,
        threshold_direction="<=",
        frequency="QUARTERLY",
    )
    result = encode_covenant(covenant)
    assert result["id"] == {"value": "cov-uuid-456"}
    assert result["deal_id"] == {"value": "deal-uuid-123"}
    assert result["name"] == "Debt to EBITDA"
    assert result["formula"] == "total_debt / ebitda"
    assert result["threshold_value_scaled"] == 3_500_000
    assert result["threshold_direction"] == "<="
    assert result["frequency"] == "QUARTERLY"


def test_encode_measurement() -> None:
    measurement = Measurement(
        deal_id=DealId(value="deal-uuid-123"),
        period_start_iso="2024-01-01",
        period_end_iso="2024-03-31",
        metric_name="total_debt",
        metric_value_scaled=100_000_000_000_000,
    )
    result = encode_measurement(measurement)
    assert result["deal_id"] == {"value": "deal-uuid-123"}
    assert result["period_start_iso"] == "2024-01-01"
    assert result["period_end_iso"] == "2024-03-31"
    assert result["metric_name"] == "total_debt"
    assert result["metric_value_scaled"] == 100_000_000_000_000


def test_encode_covenant_result() -> None:
    covenant_result = CovenantResult(
        covenant_id=CovenantId(value="cov-uuid-456"),
        period_start_iso="2024-01-01",
        period_end_iso="2024-03-31",
        calculated_value_scaled=3_200_000,
        status="OK",
    )
    result = encode_covenant_result(covenant_result)
    assert result["covenant_id"] == {"value": "cov-uuid-456"}
    assert result["period_start_iso"] == "2024-01-01"
    assert result["period_end_iso"] == "2024-03-31"
    assert result["calculated_value_scaled"] == 3_200_000
    assert result["status"] == "OK"


def test_encode_covenant_result_near_breach() -> None:
    covenant_result = CovenantResult(
        covenant_id=CovenantId(value="cov-uuid-789"),
        period_start_iso="2024-04-01",
        period_end_iso="2024-06-30",
        calculated_value_scaled=3_400_000,
        status="NEAR_BREACH",
    )
    result = encode_covenant_result(covenant_result)
    assert result["status"] == "NEAR_BREACH"


def test_encode_covenant_result_breach() -> None:
    covenant_result = CovenantResult(
        covenant_id=CovenantId(value="cov-uuid-abc"),
        period_start_iso="2024-07-01",
        period_end_iso="2024-09-30",
        calculated_value_scaled=4_000_000,
        status="BREACH",
    )
    result = encode_covenant_result(covenant_result)
    assert result["status"] == "BREACH"


def test_encode_covenant_gte_direction() -> None:
    covenant = Covenant(
        id=CovenantId(value="cov-gte"),
        deal_id=DealId(value="deal-123"),
        name="Interest Coverage",
        formula="ebitda / interest_expense",
        threshold_value_scaled=2_000_000,
        threshold_direction=">=",
        frequency="ANNUAL",
    )
    result = encode_covenant(covenant)
    assert result["threshold_direction"] == ">="
    assert result["frequency"] == "ANNUAL"
