"""Outlook provider: decoding and reads."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_core.errors import AppError, EmailErrorCode
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue, dump_json_str

from platform_email.fake_hooks import (
    make_fake_http_get,
    make_fake_http_send,
    make_raising_http_get,
    make_raising_http_send,
)
from platform_email.providers.outlook import (
    _OutlookEmailClient,
)
from platform_email.providers.outlook_decode import (
    _decode_email_address,
    _decode_folder_type,
    _decode_message,
    _decode_recipients,
)
from platform_email.testing import (
    FakeHTTPError,
    hooks,
    reset_hooks,
)
from platform_email.types import Attachment, BodyType, EmailImportance, FolderType


@pytest.fixture(autouse=True)
def _reset_hooks_after_test() -> Generator[None, None, None]:
    """Reset hooks after each test."""
    yield
    reset_hooks()


class TestDecodeEmailAddress:
    """Tests for _decode_email_address helper."""

    def test_decodes_valid_email_address(self) -> None:
        """Test decoding valid email address."""
        data: JSONObject = {"emailAddress": {"address": "test@example.com", "name": "Test User"}}
        result = _decode_email_address(data)
        assert result["address"] == "test@example.com"
        assert result["name"] == "Test User"

    def test_returns_empty_for_missing_email_address(self) -> None:
        """Test that missing emailAddress returns empty values."""
        data: JSONObject = {}
        result = _decode_email_address(data)
        assert result["address"] == ""
        assert result["name"] == ""

    def test_returns_empty_for_non_dict_email_address(self) -> None:
        """Test that non-dict emailAddress returns empty values."""
        data: JSONObject = {"emailAddress": "not a dict"}
        result = _decode_email_address(data)
        assert result["address"] == ""
        assert result["name"] == ""


class TestDecodeRecipients:
    """Tests for _decode_recipients helper."""

    def test_decodes_multiple_recipients(self) -> None:
        """Test decoding multiple recipients."""
        items: list[JSONValue] = [
            {"emailAddress": {"address": "a@test.com", "name": "A"}},
            {"emailAddress": {"address": "b@test.com", "name": "B"}},
        ]
        result = _decode_recipients(items)
        assert len(result) == 2
        assert result[0]["address"] == "a@test.com"
        assert result[1]["address"] == "b@test.com"

    def test_skips_non_dict_items(self) -> None:
        """Test that non-dict items are skipped."""
        items: list[JSONValue] = [
            {"emailAddress": {"address": "a@test.com", "name": "A"}},
            "not a dict",
            123,
        ]
        result = _decode_recipients(items)
        assert len(result) == 1

    def test_empty_list(self) -> None:
        """Test decoding empty list."""
        result = _decode_recipients([])
        assert result == ()


class TestDecodeFolderType:
    """Tests for _decode_folder_type helper."""

    def test_inbox(self) -> None:
        """Test inbox folder type."""
        assert _decode_folder_type("Inbox") is FolderType.INBOX
        assert _decode_folder_type("INBOX") is FolderType.INBOX

    def test_sent_items(self) -> None:
        """Test sent folder type."""
        assert _decode_folder_type("Sent Items") is FolderType.SENT
        assert _decode_folder_type("Sent") is FolderType.SENT

    def test_drafts(self) -> None:
        """Test drafts folder type."""
        assert _decode_folder_type("Drafts") is FolderType.DRAFTS

    def test_trash(self) -> None:
        """Test trash folder type."""
        assert _decode_folder_type("Deleted Items") is FolderType.TRASH
        assert _decode_folder_type("Trash") is FolderType.TRASH

    def test_spam(self) -> None:
        """Test spam folder type."""
        assert _decode_folder_type("Junk Email") is FolderType.SPAM
        assert _decode_folder_type("Spam") is FolderType.SPAM
        assert _decode_folder_type("Junk") is FolderType.SPAM

    def test_archive(self) -> None:
        """Test archive folder type."""
        assert _decode_folder_type("Archive") is FolderType.ARCHIVE

    def test_custom(self) -> None:
        """Test custom folder type."""
        assert _decode_folder_type("My Custom Folder") is FolderType.CUSTOM


def _graph_message(extra: JSONObject) -> JSONObject:
    """A Graph message with only an id, plus the fields a test names."""
    message: JSONObject = {"id": "msg"}
    message.update(extra)
    return message


class TestDecodeMessageVocabularies:
    """_decode_message narrows Graph's bodyType and importance enumerations strictly."""

    def test_absent_content_type_and_importance_read_as_text_and_normal(self) -> None:
        """A body without contentType is text, and a message without importance is normal."""
        email = _decode_message(_graph_message({"body": {"content": "hi"}}))
        assert email["body_type"] is BodyType.TEXT
        assert email["importance"] is EmailImportance.NORMAL

    def test_unknown_content_type_is_refused(self) -> None:
        """A contentType outside Graph's bodyType enumeration is refused by name."""
        with pytest.raises(
            JSONTypeError,
            match=r"^Invalid body.contentType 'markdown': must be one of 'text', 'html'$",
        ):
            _decode_message(_graph_message({"body": {"contentType": "markdown"}}))

    def test_unknown_importance_is_refused(self) -> None:
        """An importance outside Graph's enumeration is refused by name."""
        with pytest.raises(
            JSONTypeError,
            match=r"^Invalid importance 'urgent': must be one of 'low', 'normal', 'high'$",
        ):
            _decode_message(_graph_message({"importance": "urgent"}))


class TestOutlookEmailClientInit:
    """Tests for _OutlookEmailClient initialization."""

    def test_init_stores_access_token(self) -> None:
        """Test that init stores access token."""
        client = _OutlookEmailClient(access_token="test_token")
        headers = client._headers()
        assert headers["Authorization"] == "Bearer test_token"


class TestOutlookEmailClientGetEmail:
    """Tests for _OutlookEmailClient.get_email()."""

    def test_get_email_success(self) -> None:
        """Test successful email retrieval."""
        response = dump_json_str(
            {
                "id": "msg123",
                "conversationId": "conv456",
                "parentFolderId": "inbox",
                "subject": "Test Subject",
                "body": {"content": "Test body", "contentType": "text"},
                "from": {"emailAddress": {"address": "sender@test.com", "name": "Sender"}},
                "toRecipients": [{"emailAddress": {"address": "me@test.com", "name": "Me"}}],
                "ccRecipients": [],
                "bccRecipients": [],
                "sentDateTime": "2024-01-01T10:00:00Z",
                "receivedDateTime": "2024-01-01T10:00:01Z",
                "isRead": True,
                "isDraft": False,
                "hasAttachments": False,
                "importance": "normal",
            }
        )
        hooks.http_get = make_fake_http_get(response)

        client = _OutlookEmailClient(access_token="token")
        email = client.get_email(email_id="msg123")

        assert email["id"] == "msg123"
        assert email["subject"] == "Test Subject"
        assert email["from_address"]["address"] == "sender@test.com"

    def test_get_email_with_html_body(self) -> None:
        """Test email with HTML body."""
        response = dump_json_str(
            {
                "id": "msg123",
                "subject": "HTML Email",
                "body": {"content": "<p>HTML</p>", "contentType": "HTML"},
                "from": {"emailAddress": {"address": "sender@test.com"}},
                "toRecipients": [],
                "ccRecipients": [],
                "bccRecipients": [],
                "isRead": False,
                "isDraft": False,
                "hasAttachments": False,
                "importance": "high",
            }
        )
        hooks.http_get = make_fake_http_get(response)

        client = _OutlookEmailClient(access_token="token")
        email = client.get_email(email_id="msg123")

        assert email["body_type"] is BodyType.HTML
        assert email["importance"] is EmailImportance.HIGH

    def test_get_email_with_low_importance(self) -> None:
        """Test email with low importance."""
        response = dump_json_str(
            {
                "id": "msg123",
                "subject": "Low Priority",
                "body": {"content": "Low", "contentType": "text"},
                "from": {"emailAddress": {"address": "sender@test.com"}},
                "toRecipients": [],
                "ccRecipients": [],
                "bccRecipients": [],
                "isRead": False,
                "isDraft": False,
                "hasAttachments": False,
                "importance": "low",
            }
        )
        hooks.http_get = make_fake_http_get(response)

        client = _OutlookEmailClient(access_token="token")
        email = client.get_email(email_id="msg123")

        assert email["importance"] is EmailImportance.LOW

    def test_get_email_connection_error(self) -> None:
        """Test connection error handling."""
        hooks.http_get = make_raising_http_get(ConnectionError("Network down"))

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.get_email(email_id="msg123")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR

    def test_get_email_http_404_error(self) -> None:
        """Test HTTP 404 error handling."""
        error = FakeHTTPError(404, "Not found")
        hooks.http_get = make_raising_http_get(error)

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.get_email(email_id="nonexistent")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_NOT_FOUND

    def test_get_email_invalid_json(self) -> None:
        """Test invalid JSON response handling."""
        hooks.http_get = make_fake_http_get("not json")

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.get_email(email_id="msg123")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR


class TestOutlookEmailClientSendEmail:
    """Tests for _OutlookEmailClient.send_email()."""

    def test_send_email_success(self) -> None:
        """Test successful email sending."""
        hooks.http_post = make_fake_http_send("{}")

        client = _OutlookEmailClient(access_token="token")
        email = client.send_email(
            to=("recipient@test.com",),
            subject="Test Subject",
            body="Test body",
        )

        assert email["subject"] == "Test Subject"
        assert email["folder_id"] == "sentitems"
        assert email["is_draft"] is False

    def test_send_email_with_cc_bcc(self) -> None:
        """Test sending email with CC and BCC."""
        hooks.http_post = make_fake_http_send("{}")

        client = _OutlookEmailClient(access_token="token")
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
        hooks.http_post = make_fake_http_send("{}")

        client = _OutlookEmailClient(access_token="token")
        email = client.send_email(
            to=("recipient@test.com",),
            subject="HTML Email",
            body="<p>Hello</p>",
            body_type=BodyType.HTML,
        )

        assert email["body_type"] == "html"

    def test_send_email_with_attachments(self) -> None:
        """Test sending email with attachments."""
        hooks.http_post = make_fake_http_send("{}")

        attachment = Attachment(
            id="att1",
            name="file.txt",
            content_type="text/plain",
            size=100,
            content_bytes="SGVsbG8=",
        )

        client = _OutlookEmailClient(access_token="token")
        email = client.send_email(
            to=("recipient@test.com",),
            subject="With Attachment",
            body="See attached",
            attachments=(attachment,),
        )

        assert email["has_attachments"] is True

    def test_send_email_connection_error(self) -> None:
        """Test connection error handling."""
        hooks.http_post = make_raising_http_send(ConnectionError("Network down"))

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.send_email(to=("r@test.com",), subject="Test", body="Body")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR
