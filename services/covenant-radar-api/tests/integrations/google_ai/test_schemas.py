"""Tests for Google AI integration schemas."""

from __future__ import annotations

from typing import Literal

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from covenant_radar_api.integrations.google_ai.schemas import (
    AlertContext,
    decode_alert_context,
    decode_generate_alert_request,
    decode_generate_alert_response,
    encode_alert_context,
    encode_generate_alert_request,
    encode_generate_alert_response,
    is_alert_context,
    is_generate_alert_response,
    make_alert_context,
    make_generate_alert_request,
    make_generate_alert_response,
)

# =============================================================================
# Test Fixtures
# =============================================================================


def make_valid_alert_context() -> AlertContext:
    """Create a valid AlertContext for testing."""
    return make_alert_context(
        deal_id="deal-001",
        deal_name="Acme Corp Loan",
        borrower_name="Acme Corporation",
        sector="Technology",
        risk_probability=0.85,
        risk_tier="CRITICAL",
        evaluation_status="BREACH",
        breaches_count=2,
        covenants_evaluated=5,
        period_start="2024-01-01",
        period_end="2024-03-31",
    )


def make_valid_alert_context_dict() -> JSONObject:
    """Create a valid alert context as a raw dict."""
    return {
        "deal_id": "deal-001",
        "deal_name": "Acme Corp Loan",
        "borrower_name": "Acme Corporation",
        "sector": "Technology",
        "risk_probability": 0.85,
        "risk_tier": "CRITICAL",
        "evaluation_status": "BREACH",
        "breaches_count": 2,
        "covenants_evaluated": 5,
        "period_start": "2024-01-01",
        "period_end": "2024-03-31",
    }


def _make_invalid_request_data(
    field: str,
    invalid_value: str | int,
) -> JSONObject:
    """Create request data with one field having an invalid type.

    This helper builds data dynamically to avoid static type checking
    while still testing runtime type validation.

    Args:
        field: The field to set to an invalid value.
        invalid_value: The invalid value to set.

    Returns:
        Request data dict with the specified invalid field.
    """
    base: JSONObject = {
        "context": make_valid_alert_context_dict(),
        "model": "gemini-2.5-flash",
        "max_tokens": 256,
    }
    base[field] = invalid_value
    return base


# =============================================================================
# AlertContext Tests
# =============================================================================


class TestMakeAlertContext:
    """Tests for make_alert_context factory."""

    def test_creates_alert_context(self) -> None:
        """Test that make_alert_context creates valid context."""
        context = make_valid_alert_context()
        assert context["deal_id"] == "deal-001"
        assert context["deal_name"] == "Acme Corp Loan"
        assert context["borrower_name"] == "Acme Corporation"
        assert context["sector"] == "Technology"
        assert context["risk_probability"] == 0.85
        assert context["risk_tier"] == "CRITICAL"
        assert context["evaluation_status"] == "BREACH"
        assert context["breaches_count"] == 2
        assert context["covenants_evaluated"] == 5
        assert context["period_start"] == "2024-01-01"
        assert context["period_end"] == "2024-03-31"

    def test_accepts_all_risk_tiers(self) -> None:
        """Test all valid risk tier values."""
        tiers: tuple[Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"], ...] = (
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL",
        )
        for tier in tiers:
            context = make_alert_context(
                deal_id="d1",
                deal_name="n",
                borrower_name="b",
                sector="s",
                risk_probability=0.5,
                risk_tier=tier,
                evaluation_status="OK",
                breaches_count=0,
                covenants_evaluated=1,
                period_start="2024-01-01",
                period_end="2024-03-31",
            )
            assert context["risk_tier"] == tier

    def test_accepts_all_evaluation_statuses(self) -> None:
        """Test all valid evaluation status values."""
        statuses: tuple[Literal["OK", "BREACH", "WARNING"], ...] = (
            "OK",
            "BREACH",
            "WARNING",
        )
        for status in statuses:
            context = make_alert_context(
                deal_id="d1",
                deal_name="n",
                borrower_name="b",
                sector="s",
                risk_probability=0.5,
                risk_tier="LOW",
                evaluation_status=status,
                breaches_count=0,
                covenants_evaluated=1,
                period_start="2024-01-01",
                period_end="2024-03-31",
            )
            assert context["evaluation_status"] == status


class TestEncodeAlertContext:
    """Tests for encode_alert_context."""

    def test_encodes_all_fields(self) -> None:
        """Test that all fields are encoded."""
        context = make_valid_alert_context()
        encoded = encode_alert_context(context)

        assert encoded["deal_id"] == "deal-001"
        assert encoded["deal_name"] == "Acme Corp Loan"
        assert encoded["borrower_name"] == "Acme Corporation"
        assert encoded["sector"] == "Technology"
        assert encoded["risk_probability"] == 0.85
        assert encoded["risk_tier"] == "CRITICAL"
        assert encoded["evaluation_status"] == "BREACH"
        assert encoded["breaches_count"] == 2
        assert encoded["covenants_evaluated"] == 5
        assert encoded["period_start"] == "2024-01-01"
        assert encoded["period_end"] == "2024-03-31"

    def test_returns_plain_dict_types(self) -> None:
        """Test that encoded result contains plain types."""
        context = make_valid_alert_context()
        encoded = encode_alert_context(context)

        # Verify all values are JSON-compatible primitives
        assert encoded["deal_id"] == "deal-001"
        assert encoded["risk_probability"] == 0.85
        assert encoded["breaches_count"] == 2


class TestDecodeAlertContext:
    """Tests for decode_alert_context."""

    def test_decodes_valid_data(self) -> None:
        """Test decoding valid alert context data."""
        data = make_valid_alert_context_dict()
        context = decode_alert_context(data)

        assert context["deal_id"] == "deal-001"
        assert context["risk_tier"] == "CRITICAL"
        assert context["evaluation_status"] == "BREACH"

    def test_raises_on_missing_field(self) -> None:
        """Test that missing field raises JSONTypeError."""

        data = make_valid_alert_context_dict()
        del data["deal_id"]

        with pytest.raises(JSONTypeError, match="deal_id"):
            decode_alert_context(data)

    def test_raises_on_invalid_risk_tier(self) -> None:
        """An undeclared tier is refused.

        JSONTypeError rather than the ValueError this package's own copy of
        the narrowing used to raise. There is one narrowing now, in
        platform_core.risk_tiers, and it reports the same way for the event
        decoder, the streaming decoder and this reader -- three paths that
        previously disagreed about the type of their own refusal.
        """
        data = make_valid_alert_context_dict()
        data["risk_tier"] = "INVALID"

        with pytest.raises(JSONTypeError, match="Field 'risk_tier' must be one of"):
            decode_alert_context(data)

    def test_raises_on_invalid_evaluation_status(self) -> None:
        """An undeclared status is refused.

        JSONTypeError rather than the ValueError this package's own copy of
        the narrowing raised. There is one narrowing now, in
        platform_core.evaluation_statuses, and it reports the same way for all
        three decoders that read this set.
        """
        data = make_valid_alert_context_dict()
        data["evaluation_status"] = "INVALID"

        with pytest.raises(JSONTypeError, match="Field 'evaluation_status' must be one of"):
            decode_alert_context(data)

    def test_decodes_all_risk_tier_values(self) -> None:
        """Test decoding all valid risk tier values."""
        for tier in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            data = make_valid_alert_context_dict()
            data["risk_tier"] = tier
            context = decode_alert_context(data)
            assert context["risk_tier"] == tier

    def test_decodes_all_evaluation_status_values(self) -> None:
        """Test decoding all valid evaluation status values."""
        for status in ("OK", "BREACH", "WARNING"):
            data = make_valid_alert_context_dict()
            data["evaluation_status"] = status
            context = decode_alert_context(data)
            assert context["evaluation_status"] == status


class TestIsAlertContext:
    """Tests for is_alert_context TypeGuard."""

    def test_returns_true_for_valid_context(self) -> None:
        """Test TypeGuard returns True for valid context."""
        data = make_valid_alert_context_dict()
        assert is_alert_context(data) is True

    def test_returns_false_for_missing_field(self) -> None:
        """Test TypeGuard returns False when field is missing."""
        data = make_valid_alert_context_dict()
        del data["deal_id"]
        assert is_alert_context(data) is False

    def test_returns_false_for_invalid_risk_tier(self) -> None:
        """Test TypeGuard returns False for invalid risk tier."""
        data = make_valid_alert_context_dict()
        data["risk_tier"] = "INVALID"
        assert is_alert_context(data) is False

    def test_returns_false_for_invalid_evaluation_status(self) -> None:
        """Test TypeGuard returns False for invalid evaluation status."""
        data = make_valid_alert_context_dict()
        data["evaluation_status"] = "INVALID"
        assert is_alert_context(data) is False


class TestAlertContextRoundTrip:
    """Tests for encode/decode round trip."""

    def test_roundtrip_preserves_data(self) -> None:
        """Test that encode then decode preserves all data."""
        original = make_valid_alert_context()
        encoded = encode_alert_context(original)
        decoded = decode_alert_context(encoded)

        assert decoded == original


# =============================================================================
# GenerateAlertRequest Tests
# =============================================================================


class TestMakeGenerateAlertRequest:
    """Tests for make_generate_alert_request factory."""

    def test_creates_request(self) -> None:
        """Test that factory creates valid request."""
        context = make_valid_alert_context()
        request = make_generate_alert_request(
            context=context,
            model="gemini-2.5-flash",
            max_tokens=256,
        )

        assert request["context"] == context
        assert request["model"] == "gemini-2.5-flash"
        assert request["max_tokens"] == 256


class TestEncodeGenerateAlertRequest:
    """Tests for encode_generate_alert_request."""

    def test_encodes_nested_context(self) -> None:
        """Test that nested context is encoded."""
        from platform_core.json_utils import narrow_json_to_dict

        context = make_valid_alert_context()
        request = make_generate_alert_request(context, "gemini-2.5-flash", 256)
        encoded = encode_generate_alert_request(request)

        # Access context and verify it's properly encoded
        context_encoded = narrow_json_to_dict(encoded["context"])
        assert context_encoded["deal_id"] == "deal-001"
        assert context_encoded["risk_probability"] == 0.85

    def test_encodes_model_and_max_tokens(self) -> None:
        """Test that model and max_tokens are encoded."""
        context = make_valid_alert_context()
        request = make_generate_alert_request(context, "gemini-2.5-flash", 512)
        encoded = encode_generate_alert_request(request)

        assert encoded["model"] == "gemini-2.5-flash"
        assert encoded["max_tokens"] == 512


class TestDecodeGenerateAlertRequest:
    """Tests for decode_generate_alert_request."""

    def test_decodes_valid_data(self) -> None:
        """Test decoding valid request data."""
        data: JSONObject = {
            "context": make_valid_alert_context_dict(),
            "model": "gemini-2.5-flash",
            "max_tokens": 256,
        }
        request = decode_generate_alert_request(data)

        assert request["model"] == "gemini-2.5-flash"
        assert request["max_tokens"] == 256
        assert request["context"]["deal_id"] == "deal-001"

    def test_raises_on_invalid_context_type(self) -> None:
        """Test that invalid context type raises TypeError."""
        # Build invalid data through helper to avoid type checker
        data = _make_invalid_request_data("context", "not a dict")
        with pytest.raises(TypeError, match="context must be dict"):
            decode_generate_alert_request(data)

    def test_raises_on_invalid_model_type(self) -> None:
        """Test that invalid model type raises TypeError."""
        # Build invalid data through helper to avoid type checker
        data = _make_invalid_request_data("model", 123)
        with pytest.raises(TypeError, match="model must be str"):
            decode_generate_alert_request(data)

    def test_raises_on_invalid_max_tokens_type(self) -> None:
        """Test that invalid max_tokens type raises TypeError."""
        # Build invalid data through helper to avoid type checker
        data = _make_invalid_request_data("max_tokens", "256")
        with pytest.raises(TypeError, match="max_tokens must be int"):
            decode_generate_alert_request(data)


class TestGenerateAlertRequestRoundTrip:
    """Tests for request encode/decode round trip."""

    def test_roundtrip_preserves_data(self) -> None:
        """Test that encode then decode preserves all data."""
        context = make_valid_alert_context()
        original = make_generate_alert_request(context, "gemini-2.5-flash", 256)
        encoded = encode_generate_alert_request(original)
        decoded = decode_generate_alert_request(encoded)

        assert decoded == original


# =============================================================================
# GenerateAlertResponse Tests
# =============================================================================


class TestMakeGenerateAlertResponse:
    """Tests for make_generate_alert_response factory."""

    def test_creates_response(self) -> None:
        """Test that factory creates valid response."""
        response = make_generate_alert_response(
            summary="High risk alert for Acme Corp.",
            input_tokens=150,
            output_tokens=25,
            model="gemini-2.5-flash",
            latency_ms=342,
        )

        assert response["summary"] == "High risk alert for Acme Corp."
        assert response["input_tokens"] == 150
        assert response["output_tokens"] == 25
        assert response["model"] == "gemini-2.5-flash"
        assert response["latency_ms"] == 342


class TestEncodeGenerateAlertResponse:
    """Tests for encode_generate_alert_response."""

    def test_encodes_all_fields(self) -> None:
        """Test that all fields are encoded."""
        response = make_generate_alert_response(
            summary="Alert summary",
            input_tokens=100,
            output_tokens=20,
            model="gemini-2.5-flash",
            latency_ms=250,
        )
        encoded = encode_generate_alert_response(response)

        assert encoded["summary"] == "Alert summary"
        assert encoded["input_tokens"] == 100
        assert encoded["output_tokens"] == 20
        assert encoded["model"] == "gemini-2.5-flash"
        assert encoded["latency_ms"] == 250


class TestDecodeGenerateAlertResponse:
    """Tests for decode_generate_alert_response."""

    def test_decodes_valid_data(self) -> None:
        """Test decoding valid response data."""
        data: JSONObject = {
            "summary": "Alert text",
            "input_tokens": 100,
            "output_tokens": 20,
            "model": "gemini-2.5-flash",
            "latency_ms": 200,
        }
        response = decode_generate_alert_response(data)

        assert response["summary"] == "Alert text"
        assert response["input_tokens"] == 100
        assert response["output_tokens"] == 20
        assert response["model"] == "gemini-2.5-flash"
        assert response["latency_ms"] == 200

    def test_raises_on_missing_field(self) -> None:
        """Test that missing field raises JSONTypeError."""

        data: JSONObject = {
            "summary": "Alert",
            "input_tokens": 100,
            # missing output_tokens
            "model": "gemini-2.5-flash",
            "latency_ms": 200,
        }
        with pytest.raises(JSONTypeError, match="output_tokens"):
            decode_generate_alert_response(data)


class TestIsGenerateAlertResponse:
    """Tests for is_generate_alert_response TypeGuard."""

    def test_returns_true_for_valid_response(self) -> None:
        """Test TypeGuard returns True for valid response."""
        data: JSONObject = {
            "summary": "Alert",
            "input_tokens": 100,
            "output_tokens": 20,
            "model": "gemini-2.5-flash",
            "latency_ms": 200,
        }
        assert is_generate_alert_response(data) is True

    def test_returns_false_for_missing_field(self) -> None:
        """Test TypeGuard returns False when field is missing."""
        data: JSONObject = {
            "summary": "Alert",
            "input_tokens": 100,
            # missing output_tokens
            "model": "gemini-2.5-flash",
            "latency_ms": 200,
        }
        assert is_generate_alert_response(data) is False


class TestGenerateAlertResponseRoundTrip:
    """Tests for response encode/decode round trip."""

    def test_roundtrip_preserves_data(self) -> None:
        """Test that encode then decode preserves all data."""
        original = make_generate_alert_response(
            summary="Alert summary text",
            input_tokens=150,
            output_tokens=30,
            model="gemini-2.5-flash",
            latency_ms=300,
        )
        encoded = encode_generate_alert_response(original)
        decoded = decode_generate_alert_response(encoded)

        assert decoded == original
