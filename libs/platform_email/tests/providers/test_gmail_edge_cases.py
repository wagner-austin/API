"""Gmail provider: decode and client edge cases."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_core.errors import AppError, EmailErrorCode
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str

from platform_email.fake_hooks import (
    make_fake_http_get,
    make_fake_http_post,
    make_raising_http_post,
)
from platform_email.providers.gmail import (
    _GmailEmailClient,
)
from platform_email.providers.gmail_decode import (
    _check_has_attachments,
    _decode_body_content,
    _decode_multipart_body,
    _decode_simple_body,
    _parse_importance,
)
from platform_email.testing import (
    hooks,
    reset_hooks,
)
from platform_email.types import Attachment


@pytest.fixture(autouse=True)
def _reset_hooks_after_test() -> Generator[None, None, None]:
    """Reset hooks after each test."""
    yield
    reset_hooks()


class TestDecodeSimpleBodyEdgeCases:
    """Tests for edge cases in _decode_simple_body."""

    def test_returns_none_when_base64_decode_fails(self) -> None:
        """Test that invalid base64 data in body returns None."""
        payload: JSONObject = {
            "mimeType": "text/plain",
            "body": {"data": "!!!invalid-base64-data!!!"},
        }
        result = _decode_simple_body(payload)
        assert result is None


class TestCheckHasAttachments:
    """Tests for _check_has_attachments helper."""

    def test_returns_true_when_filename_present(self) -> None:
        """Test attachment detected when filename is present."""
        payload: JSONObject = {
            "parts": [
                {"mimeType": "text/plain", "body": {"data": "SGVsbG8"}},
                {"mimeType": "application/pdf", "filename": "document.pdf"},
            ],
        }
        result = _check_has_attachments(payload)
        assert result is True

    def test_returns_false_when_no_filename(self) -> None:
        """Test no attachment when no filenames present."""
        payload: JSONObject = {
            "parts": [
                {"mimeType": "text/plain", "body": {"data": "SGVsbG8"}},
                {"mimeType": "text/html", "body": {"data": "PGh0bWw-"}},
            ],
        }
        result = _check_has_attachments(payload)
        assert result is False

    def test_returns_false_for_none_payload(self) -> None:
        """Test returns False for None payload."""
        result = _check_has_attachments(None)
        assert result is False

    def test_returns_false_for_non_list_parts(self) -> None:
        """Test returns False when parts is not a list."""
        payload: JSONObject = {"parts": "not a list"}
        result = _check_has_attachments(payload)
        assert result is False


class TestParseImportance:
    """Tests for _parse_importance helper."""

    def test_returns_high_for_high_header(self) -> None:
        """Test high importance is returned for high header."""
        headers: list[JSONValue] = [
            {"name": "Importance", "value": "high"},
        ]
        result = _parse_importance(headers)
        assert result == "high"

    def test_returns_low_for_low_header(self) -> None:
        """Test low importance is returned for low header."""
        headers: list[JSONValue] = [
            {"name": "Importance", "value": "low"},
        ]
        result = _parse_importance(headers)
        assert result == "low"

    def test_returns_normal_for_missing_header(self) -> None:
        """Test normal importance for missing header."""
        headers: list[JSONValue] = []
        result = _parse_importance(headers)
        assert result == "normal"


class TestGmailEmailClientListEmailsPageToken:
    """Tests for page_token handling in list_emails."""

    def test_list_emails_with_page_token(self) -> None:
        """Test that page_token is sent to the API."""
        captured_urls: list[str] = []

        def capture_get(url: str, headers: dict[str, str]) -> str:
            captured_urls.append(url)
            return dump_json_str({"messages": []})

        hooks.http_get = capture_get

        client = _GmailEmailClient(access_token="token")
        client.list_emails(page_token="page_token_123")

        assert "pageToken=page_token_123" in captured_urls[0]


class TestGmailEmailClientMoveEmailErrors:
    """Tests for error handling in move_email."""

    def test_move_email_os_error(self) -> None:
        """Test OSError handling in move_email."""
        hooks.http_post = make_raising_http_post(OSError("Socket error"))

        client = _GmailEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.move_email(email_id="msg123", destination_folder_id="ARCHIVE")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR

    def test_move_email_invalid_json_response(self) -> None:
        """Test invalid JSON response in move_email."""
        hooks.http_post = make_fake_http_post("not json")

        client = _GmailEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.move_email(email_id="msg123", destination_folder_id="ARCHIVE")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR


class TestGmailEmailClientGetAttachmentDecodeError:
    """Tests for decode error handling in get_attachment."""

    def test_get_attachment_with_invalid_base64(self) -> None:
        """Test that invalid base64 is handled gracefully."""
        # The invalid data that can't be decoded should fall back to raw data
        response = dump_json_str(
            {
                "size": 100,
                "data": "!!!invalid-base64-that-cannot-be-decoded!!!",
            }
        )
        hooks.http_get = make_fake_http_get(response)

        client = _GmailEmailClient(access_token="token")
        attachment = client.get_attachment(email_id="msg123", attachment_id="att123")

        # Falls back to using the raw data when decode fails
        assert attachment["content_bytes"] == "!!!invalid-base64-that-cannot-be-decoded!!!"

    def test_get_attachment_with_empty_data(self) -> None:
        """Test that empty data returns None content_bytes."""
        response = dump_json_str({"size": 0, "data": ""})
        hooks.http_get = make_fake_http_get(response)

        client = _GmailEmailClient(access_token="token")
        attachment = client.get_attachment(email_id="msg123", attachment_id="att123")

        # Empty data should result in None content_bytes
        assert attachment["content_bytes"] is None


class TestGmailEmailClientListEmailsEdgeCases:
    """Tests for edge cases in list_emails."""

    def test_list_emails_with_non_dict_message_refs(self) -> None:
        """Test list_emails when message references are not dicts."""
        response = dump_json_str(
            {
                "messages": [
                    "not a dict",
                    123,
                ],
            }
        )
        hooks.http_get = make_fake_http_get(response)

        client = _GmailEmailClient(access_token="token")
        result = client.list_emails()

        # Should return empty result (non-dict messages are skipped)
        assert result["emails"] == ()

    def test_list_emails_with_missing_message_id(self) -> None:
        """Test list_emails when message dict has no id."""
        response = dump_json_str(
            {
                "messages": [
                    {"threadId": "thread123"},  # Missing id
                ],
            }
        )
        hooks.http_get = make_fake_http_get(response)

        client = _GmailEmailClient(access_token="token")
        result = client.list_emails()

        # Should return empty result (messages without id are skipped)
        assert result["emails"] == ()


class TestGmailSendEmailWithAttachments:
    """Tests for send_email with attachments edge cases."""

    def test_send_email_with_attachment_without_content(self) -> None:
        """Test sending email where attachment has no content_bytes."""

        def fake_post(url: str, headers: dict[str, str], body: str) -> str:
            return dump_json_str({"id": "msg123", "threadId": "thread123"})

        def fake_get(url: str, headers: dict[str, str]) -> str:
            return dump_json_str(
                {
                    "id": "msg123",
                    "threadId": "thread123",
                    "labelIds": ["SENT"],
                    "payload": {
                        "headers": [
                            {"name": "Subject", "value": "Test"},
                            {"name": "From", "value": "me@test.com"},
                        ],
                    },
                }
            )

        hooks.http_post = fake_post
        hooks.http_get = fake_get

        client = _GmailEmailClient(access_token="token")
        # Attachment with None content_bytes
        attachment = Attachment(
            id="att1",
            name="doc.pdf",
            content_type="application/pdf",
            size=0,
            content_bytes=None,
        )
        email = client.send_email(
            to=("r@test.com",),
            subject="Test",
            body="Body",
            attachments=(attachment,),
        )

        assert email["id"] == "msg123"


class TestGmailDecodeBodyContentEdgeCases:
    """Tests for _decode_body_content edge cases."""

    def test_extract_headers_with_non_list(self) -> None:
        """Test _extract_headers when headers is not a list."""
        from platform_email.providers.gmail_decode import (
            _extract_headers,
        )

        data: JSONObject = {
            "payload": {
                "headers": "not a list",
            },
        }
        headers = _extract_headers(data)
        assert headers == []

    def test_decode_body_content_multipart_returns_none(self) -> None:
        """Test _decode_body_content when multipart body returns None."""
        payload: JSONObject = {
            "mimeType": "multipart/alternative",
            "parts": [
                {"mimeType": "application/octet-stream", "body": {}},
            ],
        }
        result = _decode_body_content(payload)
        # Should fall through to return empty string
        assert result == ("", "text")


class TestGmailCheckHasAttachmentsLoopBranch:
    """Tests for _check_has_attachments loop edge cases."""

    def test_parts_with_non_dict_items_only(self) -> None:
        """Test when all parts are non-dict."""
        payload: JSONObject = {
            "parts": ["not dict", 123, None],
        }
        result = _check_has_attachments(payload)
        assert result is False


class TestGmailDecodeMultipartOnlyUnknownMimeTypes:
    """Test multipart with only unknown mime types."""

    def test_multipart_with_only_unknown_types(self) -> None:
        """Test multipart body with only unknown mime types."""
        parts: list[JSONValue] = [
            {"mimeType": "application/pdf", "body": {"data": "SGVsbG8"}},
            {"mimeType": "image/png", "body": {"data": "SGVsbG8"}},
        ]
        result = _decode_multipart_body(parts)
        # Neither text/html nor text/plain, so should return None
        assert result is None
