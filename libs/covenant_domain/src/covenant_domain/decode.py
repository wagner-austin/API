from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_dict,
    require_int,
    require_str,
)

from .models import Covenant, CovenantId, CovenantResult, Deal, DealId, Measurement


def _require_covenant_status(data: JSONObject, key: str) -> Literal["OK", "NEAR_BREACH", "BREACH"]:
    """Extract and validate CovenantStatus literal."""
    value = require_str(data, key)
    if value == "OK":
        return "OK"
    if value == "NEAR_BREACH":
        return "NEAR_BREACH"
    if value == "BREACH":
        return "BREACH"
    raise JSONTypeError(f"Invalid CovenantStatus: {value}")


def _require_threshold_direction(data: JSONObject, key: str) -> Literal["<=", ">="]:
    """Extract and validate ThresholdDirection literal."""
    value = require_str(data, key)
    if value == "<=":
        return "<="
    if value == ">=":
        return ">="
    raise JSONTypeError(f"Invalid ThresholdDirection: {value}")


def _require_frequency(data: JSONObject, key: str) -> Literal["QUARTERLY", "ANNUAL"]:
    """Extract and validate CovenantFrequency literal."""
    value = require_str(data, key)
    if value == "QUARTERLY":
        return "QUARTERLY"
    if value == "ANNUAL":
        return "ANNUAL"
    raise JSONTypeError(f"Invalid CovenantFrequency: {value}")


def decode_deal_id(data: JSONObject) -> DealId:
    """Decode DealId from JSON dict."""
    return DealId(value=require_str(data, "value"))


def decode_deal(data: JSONObject) -> Deal:
    """Decode Deal from JSON dict. Raises on invalid data."""
    return Deal(
        id=decode_deal_id(require_dict(data, "id")),
        name=require_str(data, "name"),
        borrower=require_str(data, "borrower"),
        sector=require_str(data, "sector"),
        region=require_str(data, "region"),
        commitment_amount_cents=require_int(data, "commitment_amount_cents"),
        currency=require_str(data, "currency"),
        maturity_date_iso=require_str(data, "maturity_date_iso"),
    )


def decode_covenant_id(data: JSONObject) -> CovenantId:
    """Decode CovenantId from JSON dict."""
    return CovenantId(value=require_str(data, "value"))


def decode_covenant(data: JSONObject) -> Covenant:
    """Decode Covenant from JSON dict. Raises on invalid data."""
    return Covenant(
        id=decode_covenant_id(require_dict(data, "id")),
        deal_id=decode_deal_id(require_dict(data, "deal_id")),
        name=require_str(data, "name"),
        formula=require_str(data, "formula"),
        threshold_value_scaled=require_int(data, "threshold_value_scaled"),
        threshold_direction=_require_threshold_direction(data, "threshold_direction"),
        frequency=_require_frequency(data, "frequency"),
    )


def decode_measurement(data: JSONObject) -> Measurement:
    """Decode Measurement from JSON dict. Raises on invalid data."""
    return Measurement(
        deal_id=decode_deal_id(require_dict(data, "deal_id")),
        period_start_iso=require_str(data, "period_start_iso"),
        period_end_iso=require_str(data, "period_end_iso"),
        metric_name=require_str(data, "metric_name"),
        metric_value_scaled=require_int(data, "metric_value_scaled"),
    )


def decode_covenant_result(data: JSONObject) -> CovenantResult:
    """Decode CovenantResult from JSON dict. Raises on invalid data."""
    return CovenantResult(
        covenant_id=decode_covenant_id(require_dict(data, "covenant_id")),
        period_start_iso=require_str(data, "period_start_iso"),
        period_end_iso=require_str(data, "period_end_iso"),
        calculated_value_scaled=require_int(data, "calculated_value_scaled"),
        status=_require_covenant_status(data, "status"),
    )


__all__ = [
    "decode_covenant",
    "decode_covenant_id",
    "decode_covenant_result",
    "decode_deal",
    "decode_deal_id",
    "decode_measurement",
]
