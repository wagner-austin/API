"""Gmail provider: wire decoding helpers."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_core.json_utils import JSONObject, JSONValue

from platform_email.providers.gmail_decode import (
    _decode_body_content,
    _decode_folder_type,
    _decode_multipart_body,
    _decode_simple_body,
    _extract_labels,
    _get_header_value,
    _parse_email_address,
    _try_decode_base64,
)
from platform_email.testing import (
    reset_hooks,
)


@pytest.fixture(autouse=True)
def _reset_hooks_after_test() -> Generator[None, None, None]:
    """Reset hooks after each test."""
    yield
    reset_hooks()


class TestParseEmailAddress:
    """Tests for _parse_email_address helper."""

    def test_parses_name_and_email(self) -> None:
        """Test parsing 'Name <email@example.com>' format."""
        result = _parse_email_address("John Doe <john@example.com>")
        assert result["address"] == "john@example.com"
        assert result["name"] == "John Doe"

    def test_parses_email_only(self) -> None:
        """Test parsing email address only."""
        result = _parse_email_address("john@example.com")
        assert result["address"] == "john@example.com"
        assert result["name"] == ""

    def test_parses_empty_string(self) -> None:
        """Test parsing empty string."""
        result = _parse_email_address("")
        assert result["address"] == ""
        assert result["name"] == ""


class TestDecodeFolderType:
    """Tests for _decode_folder_type helper."""

    def test_inbox(self) -> None:
        """Test inbox folder type."""
        assert _decode_folder_type("INBOX") == "inbox"
        assert _decode_folder_type("inbox") == "inbox"

    def test_sent(self) -> None:
        """Test sent folder type."""
        assert _decode_folder_type("SENT") == "sent"

    def test_drafts(self) -> None:
        """Test drafts folder type."""
        assert _decode_folder_type("DRAFT") == "drafts"

    def test_trash(self) -> None:
        """Test trash folder type."""
        assert _decode_folder_type("TRASH") == "trash"

    def test_spam(self) -> None:
        """Test spam folder type."""
        assert _decode_folder_type("SPAM") == "spam"

    def test_archive(self) -> None:
        """Test archive folder type."""
        assert _decode_folder_type("ARCHIVE") == "archive"
        assert _decode_folder_type("ALL") == "archive"

    def test_custom(self) -> None:
        """Test custom folder type."""
        assert _decode_folder_type("MyLabel") == "custom"


class TestGetHeaderValue:
    """Tests for _get_header_value helper."""

    def test_finds_header(self) -> None:
        """Test finding a header value."""
        headers: list[JSONValue] = [
            {"name": "Subject", "value": "Test Subject"},
            {"name": "From", "value": "sender@test.com"},
        ]
        result = _get_header_value(headers, "Subject")
        assert result == "Test Subject"

    def test_case_insensitive(self) -> None:
        """Test case-insensitive header lookup."""
        headers: list[JSONValue] = [
            {"name": "SUBJECT", "value": "Test"},
        ]
        result = _get_header_value(headers, "subject")
        assert result == "Test"

    def test_returns_empty_for_missing(self) -> None:
        """Test that missing header returns empty string."""
        headers: list[JSONValue] = [
            {"name": "Other", "value": "Value"},
        ]
        result = _get_header_value(headers, "Subject")
        assert result == ""

    def test_skips_non_dict_items(self) -> None:
        """Test that non-dict items are skipped."""
        headers: list[JSONValue] = [
            "not a dict",
            {"name": "Subject", "value": "Test"},
        ]
        result = _get_header_value(headers, "Subject")
        assert result == "Test"


class TestTryDecodeBase64:
    """Tests for _try_decode_base64 helper."""

    def test_decodes_valid_base64(self) -> None:
        """Test decoding valid base64url string."""
        # "Hello" in base64url
        result = _try_decode_base64("SGVsbG8")
        assert result == "Hello"

    def test_returns_none_for_invalid(self) -> None:
        """Test that invalid base64 returns None."""
        result = _try_decode_base64("not-valid-base64!!!")
        assert result is None


class TestDecodeSimpleBody:
    """Tests for _decode_simple_body helper."""

    def test_decodes_text_body(self) -> None:
        """Test decoding text body."""
        payload: JSONObject = {
            "mimeType": "text/plain",
            "body": {"data": "SGVsbG8"},
        }
        result = _decode_simple_body(payload)
        if result is None:
            pytest.fail("Expected result but got None")
        body_content, body_type = result
        assert body_content == "Hello"
        assert body_type == "text"

    def test_decodes_html_body(self) -> None:
        """Test decoding HTML body."""
        payload: JSONObject = {
            "mimeType": "text/html",
            "body": {"data": "PHA-SGVsbG88L3A-"},
        }
        result = _decode_simple_body(payload)
        if result is None:
            pytest.fail("Expected result but got None")
        _, body_type = result
        assert body_type == "html"

    def test_returns_none_for_no_body(self) -> None:
        """Test that missing body returns None."""
        payload: JSONObject = {"mimeType": "text/plain"}
        result = _decode_simple_body(payload)
        assert result is None

    def test_returns_none_for_no_data(self) -> None:
        """Test that missing data returns None."""
        payload: JSONObject = {"mimeType": "text/plain", "body": {}}
        result = _decode_simple_body(payload)
        assert result is None

    def test_returns_none_for_non_dict_body(self) -> None:
        """Test that non-dict body returns None."""
        payload: JSONObject = {"mimeType": "text/plain", "body": "string"}
        result = _decode_simple_body(payload)
        assert result is None


class TestDecodeMultipartBody:
    """Tests for _decode_multipart_body helper."""

    def test_prefers_html(self) -> None:
        """Test that HTML is preferred over plain text."""
        parts: list[JSONValue] = [
            {"mimeType": "text/plain", "body": {"data": "UGxhaW4"}},
            {"mimeType": "text/html", "body": {"data": "SFRNTA"}},
        ]
        result = _decode_multipart_body(parts)
        if result is None:
            pytest.fail("Expected result but got None")
        _, body_type = result
        assert body_type == "html"

    def test_falls_back_to_text(self) -> None:
        """Test fallback to plain text."""
        parts: list[JSONValue] = [
            {"mimeType": "text/plain", "body": {"data": "SGVsbG8"}},
        ]
        result = _decode_multipart_body(parts)
        if result is None:
            pytest.fail("Expected result but got None")
        body_content, body_type = result
        assert body_content == "Hello"
        assert body_type == "text"

    def test_skips_non_dict_parts(self) -> None:
        """Test that non-dict parts are skipped."""
        parts: list[JSONValue] = [
            "not a dict",
            {"mimeType": "text/plain", "body": {"data": "SGVsbG8"}},
        ]
        result = _decode_multipart_body(parts)
        if result is None:
            pytest.fail("Expected result but got None")
        _, body_type = result
        assert body_type == "text"

    def test_returns_none_for_empty_parts(self) -> None:
        """Test that empty parts returns None."""
        result = _decode_multipart_body([])
        assert result is None

    def test_skips_parts_without_body(self) -> None:
        """Test that parts without body are skipped."""
        parts: list[JSONValue] = [
            {"mimeType": "text/plain"},
        ]
        result = _decode_multipart_body(parts)
        assert result is None

    def test_skips_parts_with_non_dict_body(self) -> None:
        """Test that parts with non-dict body are skipped."""
        parts: list[JSONValue] = [
            {"mimeType": "text/plain", "body": "string"},
        ]
        result = _decode_multipart_body(parts)
        assert result is None

    def test_skips_parts_without_data(self) -> None:
        """Test that parts without data are skipped."""
        parts: list[JSONValue] = [
            {"mimeType": "text/plain", "body": {}},
        ]
        result = _decode_multipart_body(parts)
        assert result is None

    def test_skips_parts_with_invalid_base64(self) -> None:
        """Test that parts with invalid base64 data are skipped."""
        parts: list[JSONValue] = [
            {"mimeType": "text/plain", "body": {"data": "!!!invalid-base64!!!"}},
            {"mimeType": "text/html", "body": {"data": "PGh0bWw-SGVsbG88L2h0bWw-"}},
        ]
        result = _decode_multipart_body(parts)
        if result is None:
            pytest.fail("Expected result but got None")
        _body, body_type = result
        # Should return the HTML body since the plain text part failed to decode
        assert body_type == "html"


class TestDecodeBodyContent:
    """Tests for _decode_body_content helper."""

    def test_uses_simple_body(self) -> None:
        """Test that simple body is used first."""
        payload: JSONObject = {
            "mimeType": "text/plain",
            "body": {"data": "SGVsbG8"},
        }
        body, body_type = _decode_body_content(payload)
        assert body == "Hello"
        assert body_type == "text"

    def test_falls_back_to_multipart(self) -> None:
        """Test fallback to multipart body."""
        payload: JSONObject = {
            "parts": [
                {"mimeType": "text/plain", "body": {"data": "SGVsbG8"}},
            ],
        }
        body, body_type = _decode_body_content(payload)
        assert body == "Hello"
        assert body_type == "text"

    def test_returns_empty_on_failure(self) -> None:
        """Test that empty is returned if no body found."""
        payload: JSONObject = {}
        body, body_type = _decode_body_content(payload)
        assert body == ""
        assert body_type == "text"


class TestExtractLabels:
    """Tests for _extract_labels helper."""

    def test_extracts_labels(self) -> None:
        """Test extracting labels from message data."""
        data: JSONObject = {"labelIds": ["INBOX", "IMPORTANT"]}
        result = _extract_labels(data)
        assert result == ["INBOX", "IMPORTANT"]

    def test_returns_empty_for_no_labels(self) -> None:
        """Test that empty list is returned for no labels."""
        data: JSONObject = {}
        result = _extract_labels(data)
        assert result == []

    def test_returns_empty_for_non_list(self) -> None:
        """Test that empty list is returned for non-list labelIds."""
        data: JSONObject = {"labelIds": "not a list"}
        result = _extract_labels(data)
        assert result == []

    def test_filters_non_strings(self) -> None:
        """Test that non-string labels are filtered."""
        data: JSONObject = {"labelIds": ["INBOX", 123, None, "SENT"]}
        result = _extract_labels(data)
        assert result == ["INBOX", "SENT"]
