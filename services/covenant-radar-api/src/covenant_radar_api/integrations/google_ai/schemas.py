"""Schemas for Google AI (Gemini) integration.

TypedDict definitions for Gemini request/response data with encode/decode
functions following the standard platform pattern.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from typing import Literal, TypedDict, TypeGuard

from platform_core.json_utils import (
    JSONObject,
    require_float,
    require_int,
    require_str,
)
from platform_core.risk_tiers import require_risk_tier

EvaluationStatusValue = Literal["OK", "BREACH", "WARNING"]

VALID_EVALUATION_STATUSES: tuple[EvaluationStatusValue, ...] = ("OK", "BREACH", "WARNING")


def _require_evaluation_status(obj: JSONObject, key: str) -> EvaluationStatusValue:
    """Require an evaluation status field.

    Args:
        obj: JSON object to extract from.
        key: Key to look up.

    Returns:
        The validated evaluation status value.

    Raises:
        KeyError: If key is missing.
        TypeError: If value is not a string.
        ValueError: If value is not a valid evaluation status.
    """
    value = require_str(obj, key)
    if value == "OK":
        return "OK"
    if value == "BREACH":
        return "BREACH"
    if value == "WARNING":
        return "WARNING"
    msg = f"{key} must be one of {VALID_EVALUATION_STATUSES}, got {value!r}"
    raise ValueError(msg)


# =============================================================================
# Alert Context Schema
# =============================================================================


class AlertContext(TypedDict, total=True):
    """Context data for generating alert summaries.

    Contains all the information needed for Gemini to generate
    a human-readable alert summary.

    Fields:
        deal_id: Unique identifier for the deal.
        deal_name: Human-readable deal name.
        borrower_name: Name of the borrower.
        sector: Industry sector of the borrower.
        risk_probability: ML-predicted probability (0.0-1.0).
        risk_tier: Risk tier classification.
        evaluation_status: Deterministic evaluation result.
        breaches_count: Number of covenant breaches detected.
        covenants_evaluated: Total covenants evaluated.
        period_start: Period start date (ISO format).
        period_end: Period end date (ISO format).
    """

    deal_id: str
    deal_name: str
    borrower_name: str
    sector: str
    risk_probability: float
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    evaluation_status: Literal["OK", "BREACH", "WARNING"]
    breaches_count: int
    covenants_evaluated: int
    period_start: str
    period_end: str


def make_alert_context(
    deal_id: str,
    deal_name: str,
    borrower_name: str,
    sector: str,
    risk_probability: float,
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    evaluation_status: Literal["OK", "BREACH", "WARNING"],
    breaches_count: int,
    covenants_evaluated: int,
    period_start: str,
    period_end: str,
) -> AlertContext:
    """Create an AlertContext with validated fields.

    Args:
        deal_id: Unique identifier for the deal.
        deal_name: Human-readable deal name.
        borrower_name: Name of the borrower.
        sector: Industry sector of the borrower.
        risk_probability: ML-predicted probability (0.0-1.0).
        risk_tier: Risk tier classification.
        evaluation_status: Deterministic evaluation result.
        breaches_count: Number of covenant breaches detected.
        covenants_evaluated: Total covenants evaluated.
        period_start: Period start date (ISO format).
        period_end: Period end date (ISO format).

    Returns:
        AlertContext TypedDict.
    """
    return {
        "deal_id": deal_id,
        "deal_name": deal_name,
        "borrower_name": borrower_name,
        "sector": sector,
        "risk_probability": risk_probability,
        "risk_tier": risk_tier,
        "evaluation_status": evaluation_status,
        "breaches_count": breaches_count,
        "covenants_evaluated": covenants_evaluated,
        "period_start": period_start,
        "period_end": period_end,
    }


def encode_alert_context(context: AlertContext) -> JSONObject:
    """Encode AlertContext for JSON serialization.

    Args:
        context: AlertContext to encode.

    Returns:
        Dictionary safe for JSON serialization.
    """
    return {
        "deal_id": context["deal_id"],
        "deal_name": context["deal_name"],
        "borrower_name": context["borrower_name"],
        "sector": context["sector"],
        "risk_probability": context["risk_probability"],
        "risk_tier": context["risk_tier"],
        "evaluation_status": context["evaluation_status"],
        "breaches_count": context["breaches_count"],
        "covenants_evaluated": context["covenants_evaluated"],
        "period_start": context["period_start"],
        "period_end": context["period_end"],
    }


def decode_alert_context(data: JSONObject) -> AlertContext:
    """Decode AlertContext from JSON data.

    Args:
        data: Dictionary from JSON parsing.

    Returns:
        Validated AlertContext.

    Raises:
        KeyError: If required field is missing.
        TypeError: If field has wrong type.
        ValueError: If literal field has invalid value.
    """
    return {
        "deal_id": require_str(data, "deal_id"),
        "deal_name": require_str(data, "deal_name"),
        "borrower_name": require_str(data, "borrower_name"),
        "sector": require_str(data, "sector"),
        "risk_probability": require_float(data, "risk_probability"),
        "risk_tier": require_risk_tier(data, "risk_tier"),
        "evaluation_status": _require_evaluation_status(data, "evaluation_status"),
        "breaches_count": require_int(data, "breaches_count"),
        "covenants_evaluated": require_int(data, "covenants_evaluated"),
        "period_start": require_str(data, "period_start"),
        "period_end": require_str(data, "period_end"),
    }


def is_alert_context(data: JSONObject) -> TypeGuard[AlertContext]:
    """Check if data is a valid AlertContext.

    Args:
        data: Dictionary to validate.

    Returns:
        True if data is a valid AlertContext.
    """
    required_keys = {
        "deal_id",
        "deal_name",
        "borrower_name",
        "sector",
        "risk_probability",
        "risk_tier",
        "evaluation_status",
        "breaches_count",
        "covenants_evaluated",
        "period_start",
        "period_end",
    }
    if not required_keys.issubset(data.keys()):
        return False
    if data.get("risk_tier") not in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        return False
    return data.get("evaluation_status") in ("OK", "BREACH", "WARNING")


# =============================================================================
# Generate Alert Request Schema
# =============================================================================


class GenerateAlertRequest(TypedDict, total=True):
    """Request for generating an alert summary.

    Fields:
        context: Alert context with deal and risk information.
        model: Gemini model to use for generation.
        max_tokens: Maximum tokens in response.
    """

    context: AlertContext
    model: str
    max_tokens: int


def make_generate_alert_request(
    context: AlertContext,
    model: str,
    max_tokens: int,
) -> GenerateAlertRequest:
    """Create a GenerateAlertRequest.

    Args:
        context: Alert context with deal and risk information.
        model: Gemini model to use for generation.
        max_tokens: Maximum tokens in response.

    Returns:
        GenerateAlertRequest TypedDict.
    """
    return {
        "context": context,
        "model": model,
        "max_tokens": max_tokens,
    }


def encode_generate_alert_request(
    request: GenerateAlertRequest,
) -> JSONObject:
    """Encode GenerateAlertRequest for JSON serialization.

    Args:
        request: Request to encode.

    Returns:
        Dictionary safe for JSON serialization.
    """
    return {
        "context": encode_alert_context(request["context"]),
        "model": request["model"],
        "max_tokens": request["max_tokens"],
    }


def decode_generate_alert_request(
    data: JSONObject,
) -> GenerateAlertRequest:
    """Decode GenerateAlertRequest from JSON data.

    Args:
        data: Dictionary from JSON parsing.

    Returns:
        Validated GenerateAlertRequest.

    Raises:
        KeyError: If required field is missing.
        TypeError: If field has wrong type.
    """
    context_data = data["context"]
    if not isinstance(context_data, dict):
        msg = f"context must be dict, got {type(context_data).__name__}"
        raise TypeError(msg)
    model_value = data["model"]
    if not isinstance(model_value, str):
        msg = f"model must be str, got {type(model_value).__name__}"
        raise TypeError(msg)
    max_tokens_value = data["max_tokens"]
    if not isinstance(max_tokens_value, int):
        msg = f"max_tokens must be int, got {type(max_tokens_value).__name__}"
        raise TypeError(msg)
    return {
        "context": decode_alert_context(context_data),
        "model": model_value,
        "max_tokens": max_tokens_value,
    }


# =============================================================================
# Generate Alert Response Schema
# =============================================================================


class GenerateAlertResponse(TypedDict, total=True):
    """Response from alert summary generation.

    Fields:
        summary: Generated human-readable alert summary.
        input_tokens: Number of input tokens used.
        output_tokens: Number of output tokens generated.
        model: Model that generated the response.
        latency_ms: API call latency in milliseconds.
    """

    summary: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: int


def make_generate_alert_response(
    summary: str,
    input_tokens: int,
    output_tokens: int,
    model: str,
    latency_ms: int,
) -> GenerateAlertResponse:
    """Create a GenerateAlertResponse.

    Args:
        summary: Generated human-readable alert summary.
        input_tokens: Number of input tokens used.
        output_tokens: Number of output tokens generated.
        model: Model that generated the response.
        latency_ms: API call latency in milliseconds.

    Returns:
        GenerateAlertResponse TypedDict.
    """
    return {
        "summary": summary,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model,
        "latency_ms": latency_ms,
    }


def encode_generate_alert_response(
    response: GenerateAlertResponse,
) -> JSONObject:
    """Encode GenerateAlertResponse for JSON serialization.

    Args:
        response: Response to encode.

    Returns:
        Dictionary safe for JSON serialization.
    """
    return {
        "summary": response["summary"],
        "input_tokens": response["input_tokens"],
        "output_tokens": response["output_tokens"],
        "model": response["model"],
        "latency_ms": response["latency_ms"],
    }


def decode_generate_alert_response(
    data: JSONObject,
) -> GenerateAlertResponse:
    """Decode GenerateAlertResponse from JSON data.

    Args:
        data: Dictionary from JSON parsing.

    Returns:
        Validated GenerateAlertResponse.

    Raises:
        KeyError: If required field is missing.
        TypeError: If field has wrong type.
    """
    return {
        "summary": require_str(data, "summary"),
        "input_tokens": require_int(data, "input_tokens"),
        "output_tokens": require_int(data, "output_tokens"),
        "model": require_str(data, "model"),
        "latency_ms": require_int(data, "latency_ms"),
    }


def is_generate_alert_response(
    data: JSONObject,
) -> TypeGuard[GenerateAlertResponse]:
    """Check if data is a valid GenerateAlertResponse.

    Args:
        data: Dictionary to validate.

    Returns:
        True if data is a valid GenerateAlertResponse.
    """
    required_keys = {"summary", "input_tokens", "output_tokens", "model", "latency_ms"}
    return required_keys.issubset(data.keys())


__all__ = [
    "AlertContext",
    "GenerateAlertRequest",
    "GenerateAlertResponse",
    "decode_alert_context",
    "decode_generate_alert_request",
    "decode_generate_alert_response",
    "encode_alert_context",
    "encode_generate_alert_request",
    "encode_generate_alert_response",
    "is_alert_context",
    "is_generate_alert_response",
    "make_alert_context",
    "make_generate_alert_request",
    "make_generate_alert_response",
]
