from __future__ import annotations

from platform_core.json_utils import JSONValue

from .models import Covenant, CovenantId, CovenantResult, Deal, DealId, Measurement


def encode_deal_id(deal_id: DealId) -> dict[str, JSONValue]:
    """Encode DealId to JSON-serializable dict."""
    result: dict[str, JSONValue] = {"value": deal_id["value"]}
    return result


def encode_deal(deal: Deal) -> dict[str, JSONValue]:
    """Encode Deal to JSON-serializable dict."""
    result: dict[str, JSONValue] = {
        "id": encode_deal_id(deal["id"]),
        "name": deal["name"],
        "borrower": deal["borrower"],
        "sector": deal["sector"],
        "region": deal["region"],
        "commitment_amount_cents": deal["commitment_amount_cents"],
        "currency": deal["currency"],
        "maturity_date_iso": deal["maturity_date_iso"],
    }
    return result


def encode_covenant_id(covenant_id: CovenantId) -> dict[str, JSONValue]:
    """Encode CovenantId to JSON-serializable dict."""
    result: dict[str, JSONValue] = {"value": covenant_id["value"]}
    return result


def encode_covenant(covenant: Covenant) -> dict[str, JSONValue]:
    """Encode Covenant to JSON-serializable dict."""
    result: dict[str, JSONValue] = {
        "id": encode_covenant_id(covenant["id"]),
        "deal_id": encode_deal_id(covenant["deal_id"]),
        "name": covenant["name"],
        "formula": covenant["formula"],
        "threshold_value_scaled": covenant["threshold_value_scaled"],
        "threshold_direction": covenant["threshold_direction"],
        "frequency": covenant["frequency"],
    }
    return result


def encode_measurement(measurement: Measurement) -> dict[str, JSONValue]:
    """Encode Measurement to JSON-serializable dict."""
    result: dict[str, JSONValue] = {
        "deal_id": encode_deal_id(measurement["deal_id"]),
        "period_start_iso": measurement["period_start_iso"],
        "period_end_iso": measurement["period_end_iso"],
        "metric_name": measurement["metric_name"],
        "metric_value_scaled": measurement["metric_value_scaled"],
    }
    return result


def encode_covenant_result(result_obj: CovenantResult) -> dict[str, JSONValue]:
    """Encode CovenantResult to JSON-serializable dict."""
    result: dict[str, JSONValue] = {
        "covenant_id": encode_covenant_id(result_obj["covenant_id"]),
        "period_start_iso": result_obj["period_start_iso"],
        "period_end_iso": result_obj["period_end_iso"],
        "calculated_value_scaled": result_obj["calculated_value_scaled"],
        "status": result_obj["status"],
    }
    return result


__all__ = [
    "encode_covenant",
    "encode_covenant_id",
    "encode_covenant_result",
    "encode_deal",
    "encode_deal_id",
    "encode_measurement",
]
