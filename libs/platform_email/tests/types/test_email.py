"""Tests for platform_email.types.email module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_email.types.email import (
    Email,
    EmailAddress,
    EmailListResult,
    _require_body_type,
    _require_dict_value,
    _require_email_address_tuple,
    _require_email_importance,
    decode_email,
    decode_email_address,
    decode_email_list_result,
    encode_email,
    encode_email_address,
    encode_email_list_result,
)

# =============================================================================
# _require_body_type tests
# =============================================================================


class TestRequireBodyType:
    """Tests for _require_body_type function."""

    def test_returns_text_for_text_value(self) -> None:
        """Test that 'text' value returns 'text' literal."""
        result = _require_body_type({"body_type": "text"}, "body_type")
        assert result == "text"

    def test_returns_html_for_html_value(self) -> None:
        """Test that 'html' value returns 'html' literal."""
        result = _require_body_type({"body_type": "html"}, "body_type")
        assert result == "html"

    def test_raises_for_invalid_value(self) -> None:
        """Test that invalid values raise JSONTypeError."""
        with pytest.raises(JSONTypeError) as exc_info:
            _require_body_type({"body_type": "markdown"}, "body_type")
        assert "must be text/html" in str(exc_info.value)
        assert "markdown" in str(exc_info.value)


# =============================================================================
# _require_email_importance tests
# =============================================================================


class TestRequireEmailImportance:
    """Tests for _require_email_importance function."""

    def test_returns_low_for_low_value(self) -> None:
        """Test that 'low' value returns 'low' literal."""
        result = _require_email_importance({"importance": "low"}, "importance")
        assert result == "low"

    def test_returns_normal_for_normal_value(self) -> None:
        """Test that 'normal' value returns 'normal' literal."""
        result = _require_email_importance({"importance": "normal"}, "importance")
        assert result == "normal"

    def test_returns_high_for_high_value(self) -> None:
        """Test that 'high' value returns 'high' literal."""
        result = _require_email_importance({"importance": "high"}, "importance")
        assert result == "high"

    def test_raises_for_invalid_value(self) -> None:
        """Test that invalid values raise JSONTypeError."""
        with pytest.raises(JSONTypeError) as exc_info:
            _require_email_importance({"importance": "urgent"}, "importance")
        assert "must be low/normal/high" in str(exc_info.value)
        assert "urgent" in str(exc_info.value)


# =============================================================================
# _require_dict_value tests
# =============================================================================


class TestRequireDictValue:
    """Tests for _require_dict_value function."""

    def test_returns_dict_for_dict_value(self) -> None:
        """Test that dict values pass through."""
        result = _require_dict_value({"key": "value"}, "test_context")
        assert result == {"key": "value"}

    def test_raises_for_string_value(self) -> None:
        """Test that string values raise JSONTypeError."""
        with pytest.raises(JSONTypeError) as exc_info:
            _require_dict_value("not a dict", "field_name")
        assert "field_name must be an object" in str(exc_info.value)

    def test_raises_for_list_value(self) -> None:
        """Test that list values raise JSONTypeError."""
        with pytest.raises(JSONTypeError) as exc_info:
            _require_dict_value(["item"], "my_field")
        assert "my_field must be an object" in str(exc_info.value)

    def test_raises_for_none_value(self) -> None:
        """Test that None values raise JSONTypeError."""
        with pytest.raises(JSONTypeError) as exc_info:
            _require_dict_value(None, "nullable_field")
        assert "nullable_field must be an object" in str(exc_info.value)


# =============================================================================
# EmailAddress tests
# =============================================================================


class TestEmailAddress:
    """Tests for EmailAddress encode/decode functions."""

    def test_encode_email_address(self) -> None:
        """Test encoding an EmailAddress to JSON."""
        addr = EmailAddress(address="user@example.com", name="Test User")
        result = encode_email_address(addr)
        assert result == {"address": "user@example.com", "name": "Test User"}

    def test_decode_email_address(self) -> None:
        """Test decoding an EmailAddress from JSON."""
        data: JSONObject = {"address": "user@example.com", "name": "Test User"}
        result = decode_email_address(data)
        assert result["address"] == "user@example.com"
        assert result["name"] == "Test User"

    def test_decode_email_address_raises_for_missing_address(self) -> None:
        """Test that missing address raises JSONTypeError."""
        data: JSONObject = {"name": "Test User"}
        with pytest.raises(JSONTypeError):
            decode_email_address(data)

    def test_decode_email_address_raises_for_missing_name(self) -> None:
        """Test that missing name raises JSONTypeError."""
        data: JSONObject = {"address": "user@example.com"}
        with pytest.raises(JSONTypeError):
            decode_email_address(data)

    def test_roundtrip(self) -> None:
        """Test encoding then decoding preserves data."""
        original = EmailAddress(address="roundtrip@test.com", name="Roundtrip")
        encoded = encode_email_address(original)
        decoded = decode_email_address(encoded)
        assert decoded == original


# =============================================================================
# _require_email_address_tuple tests
# =============================================================================


class TestRequireEmailAddressTuple:
    """Tests for _require_email_address_tuple function."""

    def test_decodes_empty_list(self) -> None:
        """Test decoding empty list returns empty tuple."""
        data: JSONObject = {"recipients": []}
        result = _require_email_address_tuple(data, "recipients")
        assert result == ()

    def test_decodes_single_address(self) -> None:
        """Test decoding single address."""
        data: JSONObject = {"to": [{"address": "user@test.com", "name": "User"}]}
        result = _require_email_address_tuple(data, "to")
        assert len(result) == 1
        assert result[0]["address"] == "user@test.com"

    def test_decodes_multiple_addresses(self) -> None:
        """Test decoding multiple addresses."""
        data: JSONObject = {
            "cc": [
                {"address": "a@test.com", "name": "A"},
                {"address": "b@test.com", "name": "B"},
            ]
        }
        result = _require_email_address_tuple(data, "cc")
        assert len(result) == 2
        assert result[0]["address"] == "a@test.com"
        assert result[1]["address"] == "b@test.com"

    def test_raises_for_invalid_item_type(self) -> None:
        """Test that non-dict items raise JSONTypeError."""
        data: JSONObject = {"to": ["not a dict"]}
        with pytest.raises(JSONTypeError) as exc_info:
            _require_email_address_tuple(data, "to")
        assert "to[0] must be an object" in str(exc_info.value)


# =============================================================================
# Email tests
# =============================================================================


def _make_test_email() -> Email:
    """Create a test email for testing."""
    return Email(
        id="email-123",
        thread_id="thread-456",
        folder_id="inbox",
        subject="Test Subject",
        body="Test body content",
        body_type="text",
        from_address=EmailAddress(address="sender@test.com", name="Sender"),
        to=(EmailAddress(address="recipient@test.com", name="Recipient"),),
        cc=(),
        bcc=(),
        sent_at="2024-01-15T10:00:00Z",
        received_at="2024-01-15T10:00:01Z",
        is_read=False,
        is_draft=False,
        has_attachments=False,
        importance="normal",
    )


class TestEmail:
    """Tests for Email encode/decode functions."""

    def test_encode_email(self) -> None:
        """Test encoding an Email to JSON."""
        email = _make_test_email()
        result = encode_email(email)

        assert result["id"] == "email-123"
        assert result["thread_id"] == "thread-456"
        assert result["folder_id"] == "inbox"
        assert result["subject"] == "Test Subject"
        assert result["body"] == "Test body content"
        assert result["body_type"] == "text"
        assert result["is_read"] is False
        assert result["is_draft"] is False
        assert result["has_attachments"] is False
        assert result["importance"] == "normal"

    def test_encode_email_with_multiple_recipients(self) -> None:
        """Test encoding email with multiple to/cc/bcc."""
        email = Email(
            id="email-multi",
            thread_id="thread-multi",
            folder_id="sent",
            subject="Multi-recipient",
            body="Body",
            body_type="html",
            from_address=EmailAddress(address="me@test.com", name="Me"),
            to=(
                EmailAddress(address="a@test.com", name="A"),
                EmailAddress(address="b@test.com", name="B"),
            ),
            cc=(EmailAddress(address="c@test.com", name="C"),),
            bcc=(EmailAddress(address="d@test.com", name="D"),),
            sent_at="2024-01-15T10:00:00Z",
            received_at="2024-01-15T10:00:00Z",
            is_read=True,
            is_draft=False,
            has_attachments=True,
            importance="high",
        )
        result = encode_email(email)

        # Verify by decoding back and checking typed result
        decoded = decode_email(result)
        assert len(decoded["to"]) == 2
        assert len(decoded["cc"]) == 1
        assert len(decoded["bcc"]) == 1

    def test_decode_email(self) -> None:
        """Test decoding an Email from JSON."""
        data: JSONObject = {
            "id": "email-decoded",
            "thread_id": "thread-decoded",
            "folder_id": "archive",
            "subject": "Decoded Subject",
            "body": "Decoded body",
            "body_type": "html",
            "from_address": {"address": "from@test.com", "name": "From"},
            "to": [{"address": "to@test.com", "name": "To"}],
            "cc": [],
            "bcc": [],
            "sent_at": "2024-01-20T12:00:00Z",
            "received_at": "2024-01-20T12:00:01Z",
            "is_read": True,
            "is_draft": True,
            "has_attachments": False,
            "importance": "low",
        }
        result = decode_email(data)

        assert result["id"] == "email-decoded"
        assert result["body_type"] == "html"
        assert result["importance"] == "low"
        assert result["is_read"] is True
        assert result["is_draft"] is True
        assert len(result["to"]) == 1

    def test_roundtrip(self) -> None:
        """Test encode then decode preserves data."""
        original = _make_test_email()
        encoded = encode_email(original)
        decoded = decode_email(encoded)
        assert decoded == original


# =============================================================================
# EmailListResult tests
# =============================================================================


class TestEmailListResult:
    """Tests for EmailListResult encode/decode functions."""

    def test_encode_email_list_result(self) -> None:
        """Test encoding EmailListResult to JSON."""
        original = EmailListResult(
            emails=(_make_test_email(),),
            next_page_token="token123",
        )
        encoded = encode_email_list_result(original)

        # Verify by decoding back and checking typed result
        decoded = decode_email_list_result(encoded)
        assert len(decoded["emails"]) == 1
        assert decoded["next_page_token"] == "token123"

    def test_encode_email_list_result_no_token(self) -> None:
        """Test encoding EmailListResult with no next page."""
        result = EmailListResult(emails=(), next_page_token=None)
        encoded = encode_email_list_result(result)

        assert encoded["emails"] == []
        assert encoded["next_page_token"] is None

    def test_decode_email_list_result(self) -> None:
        """Test decoding EmailListResult from JSON."""
        email_data = encode_email(_make_test_email())
        data: JSONObject = {
            "emails": [email_data],
            "next_page_token": "next-token",
        }
        result = decode_email_list_result(data)

        assert len(result["emails"]) == 1
        assert result["next_page_token"] == "next-token"

    def test_decode_email_list_result_empty(self) -> None:
        """Test decoding empty EmailListResult."""
        data: JSONObject = {"emails": [], "next_page_token": None}
        result = decode_email_list_result(data)

        assert result["emails"] == ()
        assert result["next_page_token"] is None

    def test_roundtrip(self) -> None:
        """Test encode then decode preserves data."""
        original = EmailListResult(
            emails=(_make_test_email(),),
            next_page_token="page2",
        )
        encoded = encode_email_list_result(original)
        decoded = decode_email_list_result(encoded)
        assert decoded == original
