"""Gmail provider: read, send, list, search, drafts."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_core.errors import AppError, EmailErrorCode
from platform_core.json_utils import dump_json_str

from platform_email.fake_hooks import (
    make_fake_http_get,
    make_fake_http_post,
    make_raising_http_get,
    make_raising_http_post,
)
from platform_email.providers.gmail import (
    _GmailEmailClient,
)
from platform_email.testing import (
    FakeHTTPError,
    hooks,
    reset_hooks,
)
from platform_email.types import Attachment


@pytest.fixture(autouse=True)
def _reset_hooks_after_test() -> Generator[None, None, None]:
    """Reset hooks after each test."""
    yield
    reset_hooks()


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
