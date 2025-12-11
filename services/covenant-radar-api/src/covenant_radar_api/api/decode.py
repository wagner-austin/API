"""HTTP request body parsing for covenant-radar-api.

Parses raw request bytes into strictly-typed domain models.
Uses platform_core.json_utils and covenant_domain.decode functions.
No framework validation (e.g., Pydantic) - internal decoders only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypedDict

from covenant_domain import (
    Covenant,
    CovenantId,
    Deal,
    DealId,
    Measurement,
    decode_covenant,
    decode_covenant_id,
    decode_deal,
    decode_deal_id,
    decode_measurement,
)
from covenant_ml.types import TrainConfig
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    load_json_str,
    require_float,
    require_int,
    require_list,
    require_str,
)


class CreateDealRequest(TypedDict, total=True):
    """Request body for creating a new deal."""

    id: DealId
    name: str
    borrower: str
    sector: str
    region: str
    commitment_amount_cents: int
    currency: str
    maturity_date_iso: str


class UpdateDealRequest(TypedDict, total=True):
    """Request body for updating an existing deal."""

    name: str
    borrower: str
    sector: str
    region: str
    commitment_amount_cents: int
    currency: str
    maturity_date_iso: str


class CreateCovenantRequest(TypedDict, total=True):
    """Request body for creating a new covenant."""

    id: CovenantId
    deal_id: DealId
    name: str
    formula: str
    threshold_value_scaled: int
    threshold_direction: str
    frequency: str


class AddMeasurementsRequest(TypedDict, total=True):
    """Request body for adding measurements."""

    measurements: Sequence[Measurement]


class EvaluateRequest(TypedDict, total=True):
    """Request body for evaluating covenants for a deal and period."""

    deal_id: str
    period_start_iso: str
    period_end_iso: str
    tolerance_ratio_scaled: int


class PredictRequest(TypedDict, total=True):
    """Request body for predicting breach risk for a deal."""

    deal_id: str


class PredictResponse(TypedDict, total=True):
    """Response body for breach risk prediction."""

    deal_id: str
    probability: float
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"]


class TrainResponse(TypedDict, total=True):
    """Response body for training job submission."""

    job_id: str
    status: Literal["queued"]


def _parse_body_as_dict(body: bytes) -> JSONObject:
    """Parse request body as JSON dict. Raises on invalid JSON or non-dict."""
    raw = load_json_str(body.decode("utf-8"))
    if not isinstance(raw, dict):
        raise JSONTypeError("Request body must be a JSON object")
    return raw


def parse_deal_request(body: bytes) -> Deal:
    """Parse request body into Deal.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
    """
    data = _parse_body_as_dict(body)
    return decode_deal(data)


def parse_deal_id_request(body: bytes) -> DealId:
    """Parse request body into DealId.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
    """
    data = _parse_body_as_dict(body)
    return decode_deal_id(data)


def parse_update_deal_request(body: bytes, deal_id: DealId) -> Deal:
    """Parse update request body into Deal with provided ID.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
    """
    data = _parse_body_as_dict(body)
    return Deal(
        id=deal_id,
        name=require_str(data, "name"),
        borrower=require_str(data, "borrower"),
        sector=require_str(data, "sector"),
        region=require_str(data, "region"),
        commitment_amount_cents=require_int(data, "commitment_amount_cents"),
        currency=require_str(data, "currency"),
        maturity_date_iso=require_str(data, "maturity_date_iso"),
    )


def parse_covenant_request(body: bytes) -> Covenant:
    """Parse request body into Covenant.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
    """
    data = _parse_body_as_dict(body)
    return decode_covenant(data)


def parse_covenant_id_request(body: bytes) -> CovenantId:
    """Parse request body into CovenantId.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
    """
    data = _parse_body_as_dict(body)
    return decode_covenant_id(data)


def parse_measurements_request(body: bytes) -> list[Measurement]:
    """Parse request body into list of Measurements.

    Expects: {"measurements": [...]}

    Raises:
        JSONTypeError: Missing required field or invalid field type.
    """
    data = _parse_body_as_dict(body)
    raw_list = require_list(data, "measurements")
    result: list[Measurement] = []
    for item in raw_list:
        if not isinstance(item, dict):
            raise JSONTypeError("Each measurement must be a JSON object")
        result.append(decode_measurement(item))
    return result


def parse_evaluate_request(body: bytes) -> EvaluateRequest:
    """Parse request body into EvaluateRequest.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
    """
    data = _parse_body_as_dict(body)
    return EvaluateRequest(
        deal_id=require_str(data, "deal_id"),
        period_start_iso=require_str(data, "period_start_iso"),
        period_end_iso=require_str(data, "period_end_iso"),
        tolerance_ratio_scaled=require_int(data, "tolerance_ratio_scaled"),
    )


def parse_predict_request(body: bytes) -> PredictRequest:
    """Parse request body into PredictRequest.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
    """
    data = _parse_body_as_dict(body)
    return PredictRequest(deal_id=require_str(data, "deal_id"))


def parse_train_request(body: bytes) -> TrainConfig:
    """Parse request body into TrainConfig.

    Raises:
        JSONTypeError: Missing required field or invalid field type.
    """
    data = _parse_body_as_dict(body)
    return TrainConfig(
        learning_rate=require_float(data, "learning_rate"),
        max_depth=require_int(data, "max_depth"),
        n_estimators=require_int(data, "n_estimators"),
        subsample=require_float(data, "subsample"),
        colsample_bytree=require_float(data, "colsample_bytree"),
        random_state=require_int(data, "random_state"),
    )


__all__ = [
    "AddMeasurementsRequest",
    "CreateCovenantRequest",
    "CreateDealRequest",
    "EvaluateRequest",
    "PredictRequest",
    "PredictResponse",
    "TrainResponse",
    "UpdateDealRequest",
    "parse_covenant_id_request",
    "parse_covenant_request",
    "parse_deal_id_request",
    "parse_deal_request",
    "parse_evaluate_request",
    "parse_measurements_request",
    "parse_predict_request",
    "parse_train_request",
    "parse_update_deal_request",
]
