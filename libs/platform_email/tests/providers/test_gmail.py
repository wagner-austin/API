"""Tests for platform_email.providers.gmail module."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_core.errors import AppError, EmailErrorCode
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str

from platform_email.providers.gmail import (
    _check_has_attachments,
    _decode_body_content,
    _decode_folder_type,
    _decode_multipart_body,
    _decode_simple_body,
    _extract_labels,
    _get_header_value,
    _GmailEmailClient,
    _parse_email_address,
    _parse_importance,
    _try_decode_base64,
)
from platform_email.testing import (
    FakeHTTPError,
    hooks,
    make_fake_http_delete,
    make_fake_http_get,
    make_fake_http_post,
    make_raising_http_delete,
    make_raising_http_get,
    make_raising_http_post,
    reset_hooks,
)
from platform_email.types import Attachment


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


class TestGmailEmailClientInit:
    """Tests for _GmailEmailClient initialization."""

    def test_init_stores_access_token(self) -> None:
        """Test that init stores access token."""
        client = _GmailEmailClient(access_token="test_token")
        headers = client._headers()
        assert headers["Authorization"] == "Bearer test_token"


class TestGmailEmailClientGetEmail:
    """Tests for _GmailEmailClient.get_email()."""

    def test_get_email_success(self) -> None:
        """Test successful email retrieval."""
        response = dump_json_str(
            {
                "id": "msg123",
                "threadId": "thread456",
                "labelIds": ["INBOX"],
                "payload": {
                    "headers": [
                        {"name": "Subject", "value": "Test Subject"},
                        {"name": "From", "value": "sender@test.com"},
                        {"name": "To", "value": "me@test.com"},
                        {"name": "Date", "value": "2024-01-01T10:00:00Z"},
                    ],
                    "mimeType": "text/plain",
                    "body": {"data": "SGVsbG8"},
                },
            }
        )
        hooks.http_get = make_fake_http_get(response)

        client = _GmailEmailClient(access_token="token")
        email = client.get_email(email_id="msg123")

        assert email["id"] == "msg123"
        assert email["subject"] == "Test Subject"

    def test_get_email_connection_error(self) -> None:
        """Test connection error handling."""
        hooks.http_get = make_raising_http_get(ConnectionError("Network down"))

        client = _GmailEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.get_email(email_id="msg123")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR

    def test_get_email_http_404_error(self) -> None:
        """Test HTTP 404 error handling."""
        error = FakeHTTPError(404, "Not found")
        hooks.http_get = make_raising_http_get(error)

        client = _GmailEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.get_email(email_id="nonexistent")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_NOT_FOUND

    def test_get_email_invalid_json(self) -> None:
        """Test invalid JSON response handling."""
        hooks.http_get = make_fake_http_get("not json")

        client = _GmailEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.get_email(email_id="msg123")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR


class TestGmailEmailClientSendEmail:
    """Tests for _GmailEmailClient.send_email()."""

    def test_send_email_success(self) -> None:
        """Test successful email sending."""
        post_response = dump_json_str(
            {
                "id": "sent123",
                "threadId": "thread1",
                "labelIds": ["SENT"],
            }
        )
        get_response = dump_json_str(
            {
                "id": "sent123",
                "threadId": "thread1",
                "labelIds": ["SENT"],
                "payload": {
                    "headers": [
                        {"name": "Subject", "value": "Test Subject"},
                        {"name": "From", "value": "me@test.com"},
                        {"name": "To", "value": "recipient@test.com"},
                    ],
                    "mimeType": "text/plain",
                    "body": {"data": "VGVzdCBib2R5"},
                },
            }
        )
        hooks.http_post = make_fake_http_post(post_response)
        hooks.http_get = make_fake_http_get(get_response)

        client = _GmailEmailClient(access_token="token")
        email = client.send_email(
            to=("recipient@test.com",),
            subject="Test Subject",
            body="Test body",
        )

        assert email["id"] == "sent123"
        assert email["subject"] == "Test Subject"

    def test_send_email_with_cc_bcc(self) -> None:
        """Test sending email with CC and BCC."""
        post_response = dump_json_str(
            {
                "id": "sent123",
                "threadId": "thread1",
                "labelIds": ["SENT"],
            }
        )
        get_response = dump_json_str(
            {
                "id": "sent123",
                "threadId": "thread1",
                "labelIds": ["SENT"],
                "payload": {
                    "headers": [
                        {"name": "Subject", "value": "Subject"},
                        {"name": "From", "value": "me@test.com"},
                        {"name": "To", "value": "to@test.com"},
                        {"name": "Cc", "value": "cc@test.com"},
                        {"name": "Bcc", "value": "bcc@test.com"},
                    ],
                    "mimeType": "text/plain",
                    "body": {"data": "Qm9keQ"},
                },
            }
        )
        hooks.http_post = make_fake_http_post(post_response)
        hooks.http_get = make_fake_http_get(get_response)

        client = _GmailEmailClient(access_token="token")
        email = client.send_email(
            to=("to@test.com",),
            subject="Subject",
            body="Body",
            cc=("cc@test.com",),
            bcc=("bcc@test.com",),
        )

        assert len(email["to"]) == 1
        assert len(email["cc"]) == 1
        assert len(email["bcc"]) == 1

    def test_send_email_html_body(self) -> None:
        """Test sending email with HTML body."""
        post_response = dump_json_str(
            {
                "id": "sent123",
                "threadId": "thread1",
                "labelIds": ["SENT"],
            }
        )
        get_response = dump_json_str(
            {
                "id": "sent123",
                "threadId": "thread1",
                "labelIds": ["SENT"],
                "payload": {
                    "headers": [
                        {"name": "Subject", "value": "HTML Email"},
                        {"name": "From", "value": "me@test.com"},
                        {"name": "To", "value": "recipient@test.com"},
                    ],
                    "mimeType": "text/html",
                    "body": {"data": "PHA-SGVsbG88L3A-"},
                },
            }
        )
        hooks.http_post = make_fake_http_post(post_response)
        hooks.http_get = make_fake_http_get(get_response)

        client = _GmailEmailClient(access_token="token")
        email = client.send_email(
            to=("recipient@test.com",),
            subject="HTML Email",
            body="<p>Hello</p>",
            body_type="html",
        )

        assert email["body_type"] == "html"

    def test_send_email_with_attachments(self) -> None:
        """Test sending email with attachments."""
        post_response = dump_json_str(
            {
                "id": "sent123",
                "threadId": "thread1",
                "labelIds": ["SENT"],
            }
        )
        get_response = dump_json_str(
            {
                "id": "sent123",
                "threadId": "thread1",
                "labelIds": ["SENT"],
                "payload": {
                    "headers": [
                        {"name": "Subject", "value": "With Attachment"},
                        {"name": "From", "value": "me@test.com"},
                        {"name": "To", "value": "recipient@test.com"},
                    ],
                    "mimeType": "multipart/mixed",
                    "parts": [
                        {"mimeType": "text/plain", "body": {"data": "U2VlIGF0dGFjaGVk"}},
                        {"filename": "file.txt", "mimeType": "text/plain"},
                    ],
                },
            }
        )
        hooks.http_post = make_fake_http_post(post_response)
        hooks.http_get = make_fake_http_get(get_response)

        attachment = Attachment(
            id="att1",
            name="file.txt",
            content_type="text/plain",
            size=100,
            content_bytes="SGVsbG8=",
        )

        client = _GmailEmailClient(access_token="token")
        email = client.send_email(
            to=("recipient@test.com",),
            subject="With Attachment",
            body="See attached",
            attachments=(attachment,),
        )

        assert email["has_attachments"] is True

    def test_send_email_connection_error(self) -> None:
        """Test connection error handling."""
        hooks.http_post = make_raising_http_post(ConnectionError("Network down"))

        client = _GmailEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.send_email(to=("r@test.com",), subject="Test", body="Body")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR


class TestGmailEmailClientListEmails:
    """Tests for _GmailEmailClient.list_emails()."""

    def test_list_emails_success(self) -> None:
        """Test listing emails."""
        # First call returns message list, subsequent calls get each message
        call_count = [0]

        def fake_get(url: str, headers: dict[str, str]) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return dump_json_str(
                    {
                        "messages": [
                            {"id": "msg1", "threadId": "t1"},
                            {"id": "msg2", "threadId": "t2"},
                        ],
                    }
                )
            # Individual message fetches
            msg_id = "msg1" if "msg1" in url else "msg2"
            return dump_json_str(
                {
                    "id": msg_id,
                    "threadId": "t1",
                    "labelIds": ["INBOX"],
                    "payload": {
                        "headers": [
                            {"name": "Subject", "value": f"Email {msg_id}"},
                            {"name": "From", "value": "a@test.com"},
                        ],
                        "mimeType": "text/plain",
                        "body": {"data": "SGVsbG8"},
                    },
                }
            )

        hooks.http_get = fake_get

        client = _GmailEmailClient(access_token="token")
        result = client.list_emails()

        assert len(result["emails"]) == 2

    def test_list_emails_with_folder(self) -> None:
        """Test listing emails from specific folder/label."""
        response = dump_json_str({"messages": []})

        captured_url: list[str] = []

        def capture_get(url: str, headers: dict[str, str]) -> str:
            captured_url.append(url)
            return response

        hooks.http_get = capture_get

        client = _GmailEmailClient(access_token="token")
        client.list_emails(folder_id="INBOX")

        # Gmail uses q=label:INBOX query format
        assert "q=label" in captured_url[0]
        assert "INBOX" in captured_url[0]

    def test_list_emails_with_query(self) -> None:
        """Test listing emails with search query."""
        response = dump_json_str({"messages": []})
        hooks.http_get = make_fake_http_get(response)

        client = _GmailEmailClient(access_token="token")
        result = client.list_emails(query="test search")

        assert result["next_page_token"] is None

    def test_list_emails_with_pagination(self) -> None:
        """Test pagination token handling."""
        response = dump_json_str(
            {
                "messages": [],
                "nextPageToken": "token123",
            }
        )
        hooks.http_get = make_fake_http_get(response)

        client = _GmailEmailClient(access_token="token")
        result = client.list_emails()

        assert result["next_page_token"] == "token123"

    def test_list_emails_empty_result(self) -> None:
        """Test empty result."""
        response = dump_json_str({})
        hooks.http_get = make_fake_http_get(response)

        client = _GmailEmailClient(access_token="token")
        result = client.list_emails()

        assert result["emails"] == ()


class TestGmailEmailClientSearchEmails:
    """Tests for _GmailEmailClient.search_emails()."""

    def test_search_emails_returns_tuple(self) -> None:
        """Test that search returns tuple of emails."""
        response = dump_json_str({"messages": []})
        hooks.http_get = make_fake_http_get(response)

        client = _GmailEmailClient(access_token="token")
        result = client.search_emails(query="test")

        assert result == ()


class TestGmailEmailClientCreateDraft:
    """Tests for _GmailEmailClient.create_draft()."""

    def test_create_draft_success(self) -> None:
        """Test creating a draft."""
        response = dump_json_str(
            {
                "id": "draft123",
                "message": {"id": "msg123", "threadId": "t1"},
            }
        )
        hooks.http_post = make_fake_http_post(response)

        client = _GmailEmailClient(access_token="token")
        draft = client.create_draft(
            to=("recipient@test.com",),
            subject="Draft Subject",
            body="Draft body",
        )

        assert draft["id"] == "draft123"
        assert draft["subject"] == "Draft Subject"

    def test_create_draft_with_cc_bcc(self) -> None:
        """Test creating draft with CC and BCC."""
        response = dump_json_str(
            {
                "id": "draft123",
                "message": {"id": "msg123", "threadId": "t1"},
            }
        )
        hooks.http_post = make_fake_http_post(response)

        client = _GmailEmailClient(access_token="token")
        draft = client.create_draft(
            to=("to@test.com",),
            subject="Draft",
            body="Body",
            cc=("cc@test.com",),
            bcc=("bcc@test.com",),
        )

        assert len(draft["cc"]) == 1
        assert len(draft["bcc"]) == 1


class TestGmailEmailClientSendDraft:
    """Tests for _GmailEmailClient.send_draft()."""

    def test_send_draft_success(self) -> None:
        """Test sending a draft."""
        # First call gets the draft, second call sends
        call_count = [0]

        def fake_get(url: str, headers: dict[str, str]) -> str:
            return dump_json_str(
                {
                    "id": "draft123",
                    "message": {
                        "id": "msg123",
                        "threadId": "t1",
                        "labelIds": ["DRAFT"],
                        "payload": {
                            "headers": [
                                {"name": "Subject", "value": "Draft Subject"},
                                {"name": "From", "value": "me@test.com"},
                                {"name": "To", "value": "r@test.com"},
                            ],
                            "mimeType": "text/plain",
                            "body": {"data": "Qm9keQ"},
                        },
                    },
                }
            )

        def fake_post(url: str, headers: dict[str, str], body: str) -> str:
            call_count[0] += 1
            return dump_json_str(
                {
                    "id": "msg123",
                    "threadId": "t1",
                    "labelIds": ["SENT"],
                }
            )

        hooks.http_get = fake_get
        hooks.http_post = fake_post

        client = _GmailEmailClient(access_token="token")
        email = client.send_draft(draft_id="draft123")

        assert email["is_draft"] is False


class TestGmailEmailClientReplyToEmail:
    """Tests for _GmailEmailClient.reply_to_email()."""

    def test_reply_to_email_success(self) -> None:
        """Test replying to an email."""

        def fake_get(url: str, headers: dict[str, str]) -> str:
            # Return different responses for original vs reply message
            if "reply123" in url:
                return dump_json_str(
                    {
                        "id": "reply123",
                        "threadId": "t1",
                        "labelIds": ["SENT"],
                        "payload": {
                            "headers": [
                                {"name": "Subject", "value": "Re: Original Subject"},
                                {"name": "From", "value": "me@test.com"},
                                {"name": "To", "value": "sender@test.com"},
                            ],
                            "mimeType": "text/plain",
                            "body": {"data": "TXkgcmVwbHk"},
                        },
                    }
                )
            return dump_json_str(
                {
                    "id": "original123",
                    "threadId": "t1",
                    "labelIds": ["INBOX"],
                    "payload": {
                        "headers": [
                            {"name": "Subject", "value": "Original Subject"},
                            {"name": "From", "value": "sender@test.com"},
                            {"name": "To", "value": "me@test.com"},
                            {"name": "Message-ID", "value": "<orig@example.com>"},
                        ],
                        "mimeType": "text/plain",
                        "body": {"data": "SGVsbG8"},
                    },
                }
            )

        def fake_post(url: str, headers: dict[str, str], body: str) -> str:
            return dump_json_str(
                {
                    "id": "reply123",
                    "threadId": "t1",
                    "labelIds": ["SENT"],
                }
            )

        hooks.http_get = fake_get
        hooks.http_post = fake_post

        client = _GmailEmailClient(access_token="token")
        reply = client.reply_to_email(email_id="original123", body="My reply")

        assert reply["subject"] == "Re: Original Subject"

    def test_reply_all(self) -> None:
        """Test reply all."""

        def fake_get(url: str, headers: dict[str, str]) -> str:
            # Return different responses for original vs reply message
            if "reply123" in url:
                return dump_json_str(
                    {
                        "id": "reply123",
                        "threadId": "t1",
                        "labelIds": ["SENT"],
                        "payload": {
                            "headers": [
                                {"name": "Subject", "value": "Re: Original"},
                                {"name": "From", "value": "me@test.com"},
                                {"name": "To", "value": "sender@test.com"},
                                {"name": "Cc", "value": "me@test.com, cc@test.com"},
                            ],
                            "mimeType": "text/plain",
                            "body": {"data": "UmVwbHk"},
                        },
                    }
                )
            return dump_json_str(
                {
                    "id": "original123",
                    "threadId": "t1",
                    "labelIds": ["INBOX"],
                    "payload": {
                        "headers": [
                            {"name": "Subject", "value": "Original"},
                            {"name": "From", "value": "sender@test.com"},
                            {"name": "To", "value": "me@test.com"},
                            {"name": "Cc", "value": "cc@test.com"},
                        ],
                        "mimeType": "text/plain",
                        "body": {"data": "SGVsbG8"},
                    },
                }
            )

        def fake_post(url: str, headers: dict[str, str], body: str) -> str:
            return dump_json_str(
                {
                    "id": "reply123",
                    "threadId": "t1",
                    "labelIds": ["SENT"],
                }
            )

        hooks.http_get = fake_get
        hooks.http_post = fake_post

        client = _GmailEmailClient(access_token="token")
        reply = client.reply_to_email(email_id="original123", body="Reply", reply_all=True)

        # Check that CC has at least the original CC recipient
        assert reply["cc"][0]["address"] == "me@test.com"
        assert reply["cc"][1]["address"] == "cc@test.com"


class TestGmailEmailClientDeleteEmail:
    """Tests for _GmailEmailClient.delete_email()."""

    def test_delete_email_to_trash(self) -> None:
        """Test moving email to trash."""
        response = dump_json_str(
            {
                "id": "msg123",
                "labelIds": ["TRASH"],
            }
        )
        hooks.http_post = make_fake_http_post(response)

        client = _GmailEmailClient(access_token="token")
        client.delete_email(email_id="msg123")

        # No exception means success

    def test_delete_email_permanent(self) -> None:
        """Test permanent deletion."""
        hooks.http_delete = make_fake_http_delete()

        client = _GmailEmailClient(access_token="token")
        client.delete_email(email_id="msg123", permanent=True)

        # No exception means success

    def test_delete_email_connection_error(self) -> None:
        """Test connection error on permanent delete."""
        hooks.http_delete = make_raising_http_delete(ConnectionError("Failed"))

        client = _GmailEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.delete_email(email_id="msg123", permanent=True)
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR

    def test_delete_email_http_error(self) -> None:
        """Test HTTP error on permanent delete."""
        error = FakeHTTPError(404, "Not found")
        hooks.http_delete = make_raising_http_delete(error)

        client = _GmailEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.delete_email(email_id="nonexistent", permanent=True)
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_NOT_FOUND


class TestGmailEmailClientMoveEmail:
    """Tests for _GmailEmailClient.move_email()."""

    def test_move_email_success(self) -> None:
        """Test moving an email."""
        response = dump_json_str(
            {
                "id": "msg123",
                "threadId": "t1",
                "labelIds": ["ARCHIVE"],
                "payload": {
                    "headers": [
                        {"name": "Subject", "value": "Moved email"},
                        {"name": "From", "value": "sender@test.com"},
                    ],
                    "mimeType": "text/plain",
                    "body": {"data": "SGVsbG8"},
                },
            }
        )
        hooks.http_post = make_fake_http_post(response)

        client = _GmailEmailClient(access_token="token")
        email = client.move_email(email_id="msg123", destination_folder_id="ARCHIVE")

        assert email["folder_id"] == "ARCHIVE"


class TestGmailEmailClientListFolders:
    """Tests for _GmailEmailClient.list_folders()."""

    def test_list_folders_success(self) -> None:
        """Test listing folders/labels."""
        response = dump_json_str(
            {
                "labels": [
                    {
                        "id": "INBOX",
                        "name": "INBOX",
                        "type": "system",
                        "messagesUnread": 5,
                        "messagesTotal": 100,
                    },
                    {
                        "id": "SENT",
                        "name": "SENT",
                        "type": "system",
                        "messagesUnread": 0,
                        "messagesTotal": 50,
                    },
                ],
            }
        )
        hooks.http_get = make_fake_http_get(response)

        client = _GmailEmailClient(access_token="token")
        folders = client.list_folders()

        assert len(folders) == 2
        assert folders[0]["name"] == "INBOX"
        assert folders[0]["folder_type"] == "inbox"
        assert folders[0]["unread_count"] == 5

    def test_list_folders_empty(self) -> None:
        """Test listing folders when none exist."""
        response = dump_json_str({"labels": []})
        hooks.http_get = make_fake_http_get(response)

        client = _GmailEmailClient(access_token="token")
        folders = client.list_folders()

        assert folders == ()

    def test_list_folders_non_list_value(self) -> None:
        """Test handling non-list value."""
        response = dump_json_str({"labels": "not a list"})
        hooks.http_get = make_fake_http_get(response)

        client = _GmailEmailClient(access_token="token")
        folders = client.list_folders()

        assert folders == ()

    def test_list_folders_skips_non_dict(self) -> None:
        """Test that non-dict labels are skipped."""
        response = dump_json_str(
            {
                "labels": [
                    {"id": "INBOX", "name": "INBOX"},
                    "not a dict",
                ],
            }
        )
        hooks.http_get = make_fake_http_get(response)

        client = _GmailEmailClient(access_token="token")
        folders = client.list_folders()

        assert len(folders) == 1


class TestGmailEmailClientGetAttachment:
    """Tests for _GmailEmailClient.get_attachment()."""

    def test_get_attachment_success(self) -> None:
        """Test getting an attachment."""
        response = dump_json_str(
            {
                "size": 1024,
                "data": "SGVsbG8gV29ybGQ",
            }
        )
        hooks.http_get = make_fake_http_get(response)

        client = _GmailEmailClient(access_token="token")
        attachment = client.get_attachment(email_id="msg123", attachment_id="att123")

        assert attachment["id"] == "att123"
        # Gmail returns base64url, code converts to standard base64 (adds padding)
        assert attachment["content_bytes"] == "SGVsbG8gV29ybGQ="


class TestGmailEmailClientErrorHandling:
    """Tests for error handling in various paths."""

    def test_handle_error_label_not_found(self) -> None:
        """Test label/folder not found error."""
        error = FakeHTTPError(404, "Label not found")
        hooks.http_get = make_raising_http_get(error)

        client = _GmailEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client._get("/gmail/v1/users/me/labels/nonexistent")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.FOLDER_NOT_FOUND

    def test_handle_error_draft_not_found(self) -> None:
        """Test draft not found error."""
        error = FakeHTTPError(404, "Draft not found")
        hooks.http_get = make_raising_http_get(error)

        client = _GmailEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client._get("/gmail/v1/users/me/drafts/nonexistent")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.DRAFT_NOT_FOUND

    def test_handle_error_generic_not_found(self) -> None:
        """Test generic 404 error."""
        error = FakeHTTPError(404, "Resource not found")
        hooks.http_get = make_raising_http_get(error)

        client = _GmailEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client._get("/gmail/v1/users/me/unknown")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_NOT_FOUND

    def test_handle_error_non_404(self) -> None:
        """Test non-404 HTTP error."""
        error = FakeHTTPError(500, "Internal server error")
        hooks.http_get = make_raising_http_get(error)

        client = _GmailEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.get_email(email_id="msg123")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR

    def test_post_invalid_json(self) -> None:
        """Test invalid JSON response on POST."""
        hooks.http_post = make_fake_http_post("not json")

        client = _GmailEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.create_draft(to=("r@test.com",), subject="Test", body="Body")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR

    def test_post_http_error(self) -> None:
        """Test HTTP error on POST."""
        error = FakeHTTPError(400, "Bad request")
        hooks.http_post = make_raising_http_post(error)

        client = _GmailEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.create_draft(to=("r@test.com",), subject="Test", body="Body")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR

    def test_os_error_without_http_protocol(self) -> None:
        """Test OSError that's not an HTTPErrorProtocol."""
        hooks.http_get = make_raising_http_get(OSError("Socket error"))

        client = _GmailEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.get_email(email_id="msg123")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR

    def test_post_os_error_without_http_protocol(self) -> None:
        """Test OSError on POST that's not an HTTPErrorProtocol."""
        hooks.http_post = make_raising_http_post(OSError("Socket error"))

        client = _GmailEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.send_email(to=("r@test.com",), subject="Test", body="Body")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR

    def test_delete_os_error_without_http_protocol(self) -> None:
        """Test OSError on DELETE that's not an HTTPErrorProtocol."""
        hooks.http_delete = make_raising_http_delete(OSError("Socket error"))

        client = _GmailEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.delete_email(email_id="msg123", permanent=True)
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR


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
        from platform_email.providers.gmail import _extract_headers

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
