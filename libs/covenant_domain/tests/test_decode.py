"""Tests for covenant_domain.decode module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from covenant_domain.decode import (
    _require_covenant_status,
    _require_frequency,
    _require_threshold_direction,
    decode_covenant,
    decode_covenant_id,
    decode_covenant_result,
    decode_deal,
    decode_deal_id,
    decode_measurement,
)

# Tests for _require_covenant_status, _require_threshold_direction, _require_frequency
# These are domain-specific validators that remain in covenant_domain.
# The generic require_str, require_int, require_dict are tested in platform_core.


def test_require_covenant_status_ok() -> None:
    data: JSONObject = {"status": "OK"}
    result = _require_covenant_status(data, "status")
    assert result == "OK"


def test_require_covenant_status_near_breach() -> None:
    data: JSONObject = {"status": "NEAR_BREACH"}
    result = _require_covenant_status(data, "status")
    assert result == "NEAR_BREACH"


def test_require_covenant_status_breach() -> None:
    data: JSONObject = {"status": "BREACH"}
    result = _require_covenant_status(data, "status")
    assert result == "BREACH"


def test_require_covenant_status_invalid() -> None:
    data: JSONObject = {"status": "UNKNOWN"}
    with pytest.raises(JSONTypeError, match="Invalid CovenantStatus"):
        _require_covenant_status(data, "status")


def test_require_threshold_direction_lte() -> None:
    data: JSONObject = {"direction": "<="}
    result = _require_threshold_direction(data, "direction")
    assert result == "<="


def test_require_threshold_direction_gte() -> None:
    data: JSONObject = {"direction": ">="}
    result = _require_threshold_direction(data, "direction")
    assert result == ">="


def test_require_threshold_direction_invalid() -> None:
    data: JSONObject = {"direction": "<"}
    with pytest.raises(JSONTypeError, match="Invalid ThresholdDirection"):
        _require_threshold_direction(data, "direction")


def test_require_frequency_quarterly() -> None:
    data: JSONObject = {"freq": "QUARTERLY"}
    result = _require_frequency(data, "freq")
    assert result == "QUARTERLY"


def test_require_frequency_annual() -> None:
    data: JSONObject = {"freq": "ANNUAL"}
    result = _require_frequency(data, "freq")
    assert result == "ANNUAL"


def test_require_frequency_invalid() -> None:
    data: JSONObject = {"freq": "MONTHLY"}
    with pytest.raises(JSONTypeError, match="Invalid CovenantFrequency"):
        _require_frequency(data, "freq")


def test_decode_deal_id_valid() -> None:
    data: JSONObject = {"value": "deal-uuid-123"}
    result = decode_deal_id(data)
    assert result["value"] == "deal-uuid-123"


def test_decode_deal_valid() -> None:
    deal_id_dict: JSONObject = {"value": "deal-uuid-123"}
    data: JSONObject = {
        "id": deal_id_dict,
        "name": "Acme Corp Loan",
        "borrower": "Acme Corporation",
        "sector": "Technology",
        "region": "North America",
        "commitment_amount_cents": 100_000_000_00,
        "currency": "USD",
        "maturity_date_iso": "2028-12-31",
    }
    result = decode_deal(data)
    assert result["id"]["value"] == "deal-uuid-123"
    assert result["name"] == "Acme Corp Loan"
    assert result["borrower"] == "Acme Corporation"
    assert result["sector"] == "Technology"
    assert result["region"] == "North America"
    assert result["commitment_amount_cents"] == 100_000_000_00
    assert result["currency"] == "USD"
    assert result["maturity_date_iso"] == "2028-12-31"


def test_decode_covenant_id_valid() -> None:
    data: JSONObject = {"value": "cov-uuid-456"}
    result = decode_covenant_id(data)
    assert result["value"] == "cov-uuid-456"


def test_decode_covenant_valid() -> None:
    cov_id_dict: JSONObject = {"value": "cov-uuid-456"}
    deal_id_dict: JSONObject = {"value": "deal-uuid-123"}
    data: JSONObject = {
        "id": cov_id_dict,
        "deal_id": deal_id_dict,
        "name": "Debt to EBITDA",
        "formula": "total_debt / ebitda",
        "threshold_value_scaled": 3_500_000,
        "threshold_direction": "<=",
        "frequency": "QUARTERLY",
    }
    result = decode_covenant(data)
    assert result["id"]["value"] == "cov-uuid-456"
    assert result["deal_id"]["value"] == "deal-uuid-123"
    assert result["name"] == "Debt to EBITDA"
    assert result["formula"] == "total_debt / ebitda"
    assert result["threshold_value_scaled"] == 3_500_000
    assert result["threshold_direction"] == "<="
    assert result["frequency"] == "QUARTERLY"


def test_decode_measurement_valid() -> None:
    deal_id_dict: JSONObject = {"value": "deal-uuid-123"}
    data: JSONObject = {
        "deal_id": deal_id_dict,
        "period_start_iso": "2024-01-01",
        "period_end_iso": "2024-03-31",
        "metric_name": "total_debt",
        "metric_value_scaled": 100_000_000_000_000,
    }
    result = decode_measurement(data)
    assert result["deal_id"]["value"] == "deal-uuid-123"
    assert result["period_start_iso"] == "2024-01-01"
    assert result["period_end_iso"] == "2024-03-31"
    assert result["metric_name"] == "total_debt"
    assert result["metric_value_scaled"] == 100_000_000_000_000


def test_decode_covenant_result_valid() -> None:
    cov_id_dict: JSONObject = {"value": "cov-uuid-456"}
    data: JSONObject = {
        "covenant_id": cov_id_dict,
        "period_start_iso": "2024-01-01",
        "period_end_iso": "2024-03-31",
        "calculated_value_scaled": 3_200_000,
        "status": "OK",
    }
    result = decode_covenant_result(data)
    assert result["covenant_id"]["value"] == "cov-uuid-456"
    assert result["period_start_iso"] == "2024-01-01"
    assert result["period_end_iso"] == "2024-03-31"
    assert result["calculated_value_scaled"] == 3_200_000
    assert result["status"] == "OK"
