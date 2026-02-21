"""Tests for platform_email.providers.outlook module."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_core.errors import AppError, EmailErrorCode
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str

from platform_email.providers.outlook import (
    _decode_email_address,
    _decode_folder_type,
    _decode_importance,
    _decode_recipients,
    _OutlookEmailClient,
)
from platform_email.testing import (
    FakeHTTPError,
    hooks,
    make_fake_http_delete,
    make_fake_http_get,
    make_fake_http_patch,
    make_fake_http_post,
    make_raising_http_delete,
    make_raising_http_get,
    make_raising_http_patch,
    make_raising_http_post,
    reset_hooks,
)
from platform_email.types import Attachment


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
        assert _decode_folder_type("Inbox") == "inbox"
        assert _decode_folder_type("INBOX") == "inbox"

    def test_sent_items(self) -> None:
        """Test sent folder type."""
        assert _decode_folder_type("Sent Items") == "sent"
        assert _decode_folder_type("Sent") == "sent"

    def test_drafts(self) -> None:
        """Test drafts folder type."""
        assert _decode_folder_type("Drafts") == "drafts"

    def test_trash(self) -> None:
        """Test trash folder type."""
        assert _decode_folder_type("Deleted Items") == "trash"
        assert _decode_folder_type("Trash") == "trash"

    def test_spam(self) -> None:
        """Test spam folder type."""
        assert _decode_folder_type("Junk Email") == "spam"
        assert _decode_folder_type("Spam") == "spam"
        assert _decode_folder_type("Junk") == "spam"

    def test_archive(self) -> None:
        """Test archive folder type."""
        assert _decode_folder_type("Archive") == "archive"

    def test_custom(self) -> None:
        """Test custom folder type."""
        assert _decode_folder_type("My Custom Folder") == "custom"


class TestDecodeImportance:
    """Tests for _decode_importance helper."""

    def test_html_body_type(self) -> None:
        """Test html body type."""
        assert _decode_importance("html") == "html"

    def test_text_body_type(self) -> None:
        """Test text body type."""
        assert _decode_importance("text") == "text"
        assert _decode_importance(None) == "text"


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

        assert email["body_type"] == "html"
        assert email["importance"] == "high"

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

        assert email["importance"] == "low"

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
        hooks.http_post = make_fake_http_post("{}")

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
        hooks.http_post = make_fake_http_post("{}")

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
        hooks.http_post = make_fake_http_post("{}")

        client = _OutlookEmailClient(access_token="token")
        email = client.send_email(
            to=("recipient@test.com",),
            subject="HTML Email",
            body="<p>Hello</p>",
            body_type="html",
        )

        assert email["body_type"] == "html"

    def test_send_email_with_attachments(self) -> None:
        """Test sending email with attachments."""
        hooks.http_post = make_fake_http_post("{}")

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
        hooks.http_post = make_raising_http_post(ConnectionError("Network down"))

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.send_email(to=("r@test.com",), subject="Test", body="Body")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR


class TestOutlookEmailClientListEmails:
    """Tests for _OutlookEmailClient.list_emails()."""

    def test_list_emails_success(self) -> None:
        """Test listing emails."""
        response = dump_json_str(
            {
                "value": [
                    {
                        "id": "msg1",
                        "subject": "Email 1",
                        "body": {"content": "Body 1"},
                        "from": {"emailAddress": {"address": "a@test.com"}},
                        "toRecipients": [],
                        "ccRecipients": [],
                        "bccRecipients": [],
                        "isRead": True,
                        "isDraft": False,
                        "hasAttachments": False,
                    },
                    {
                        "id": "msg2",
                        "subject": "Email 2",
                        "body": {"content": "Body 2"},
                        "from": {"emailAddress": {"address": "b@test.com"}},
                        "toRecipients": [],
                        "ccRecipients": [],
                        "bccRecipients": [],
                        "isRead": False,
                        "isDraft": False,
                        "hasAttachments": False,
                    },
                ],
            }
        )
        hooks.http_get = make_fake_http_get(response)

        client = _OutlookEmailClient(access_token="token")
        result = client.list_emails()

        assert len(result["emails"]) == 2
        assert result["emails"][0]["id"] == "msg1"

    def test_list_emails_with_folder(self) -> None:
        """Test listing emails from specific folder."""
        response = dump_json_str({"value": []})

        captured_url: list[str] = []

        def capture_get(url: str, headers: dict[str, str]) -> str:
            captured_url.append(url)
            return response

        hooks.http_get = capture_get

        client = _OutlookEmailClient(access_token="token")
        client.list_emails(folder_id="inbox")

        assert "mailFolders/inbox/messages" in captured_url[0]

    def test_list_emails_with_query(self) -> None:
        """Test listing emails with search query."""
        response = dump_json_str({"value": []})
        hooks.http_get = make_fake_http_get(response)

        client = _OutlookEmailClient(access_token="token")
        result = client.list_emails(query="test search")

        assert result["next_page_token"] is None

    def test_list_emails_with_pagination(self) -> None:
        """Test pagination token handling."""
        response = dump_json_str(
            {
                "value": [],
                "@odata.nextLink": "https://graph.microsoft.com/v1.0/me/messages?$skiptoken=abc123",
            }
        )
        hooks.http_get = make_fake_http_get(response)

        client = _OutlookEmailClient(access_token="token")
        result = client.list_emails()

        assert result["next_page_token"] == "abc123"

    def test_list_emails_with_page_token(self) -> None:
        """Test using page token for continuation."""
        response = dump_json_str({"value": []})
        hooks.http_get = make_fake_http_get(response)

        client = _OutlookEmailClient(access_token="token")
        result = client.list_emails(page_token="prev_token")

        assert result["emails"] == ()

    def test_list_emails_skips_non_dict_messages(self) -> None:
        """Test that non-dict messages are skipped."""
        response = dump_json_str(
            {
                "value": [
                    {
                        "id": "msg1",
                        "subject": "Valid",
                        "body": {},
                        "from": {},
                        "toRecipients": [],
                        "ccRecipients": [],
                        "bccRecipients": [],
                        "isRead": False,
                        "isDraft": False,
                        "hasAttachments": False,
                    },
                    "not a dict",
                ],
            }
        )
        hooks.http_get = make_fake_http_get(response)

        client = _OutlookEmailClient(access_token="token")
        result = client.list_emails()

        assert len(result["emails"]) == 1


class TestOutlookEmailClientSearchEmails:
    """Tests for _OutlookEmailClient.search_emails()."""

    def test_search_emails_returns_tuple(self) -> None:
        """Test that search returns tuple of emails."""
        response = dump_json_str(
            {
                "value": [
                    {
                        "id": "msg1",
                        "subject": "Match",
                        "body": {"content": "Test"},
                        "from": {},
                        "toRecipients": [],
                        "ccRecipients": [],
                        "bccRecipients": [],
                        "isRead": False,
                        "isDraft": False,
                        "hasAttachments": False,
                    },
                ],
            }
        )
        hooks.http_get = make_fake_http_get(response)

        client = _OutlookEmailClient(access_token="token")
        result = client.search_emails(query="test")

        assert len(result) == 1


class TestOutlookEmailClientCreateDraft:
    """Tests for _OutlookEmailClient.create_draft()."""

    def test_create_draft_success(self) -> None:
        """Test creating a draft."""
        response = dump_json_str({"id": "draft123"})
        hooks.http_post = make_fake_http_post(response)

        client = _OutlookEmailClient(access_token="token")
        draft = client.create_draft(
            to=("recipient@test.com",),
            subject="Draft Subject",
            body="Draft body",
        )

        assert draft["id"] == "draft123"
        assert draft["subject"] == "Draft Subject"

    def test_create_draft_with_cc_bcc(self) -> None:
        """Test creating draft with CC and BCC."""
        response = dump_json_str({"id": "draft123"})
        hooks.http_post = make_fake_http_post(response)

        client = _OutlookEmailClient(access_token="token")
        draft = client.create_draft(
            to=("to@test.com",),
            subject="Draft",
            body="Body",
            cc=("cc@test.com",),
            bcc=("bcc@test.com",),
        )

        assert len(draft["cc"]) == 1
        assert len(draft["bcc"]) == 1


class TestOutlookEmailClientSendDraft:
    """Tests for _OutlookEmailClient.send_draft()."""

    def test_send_draft_success(self) -> None:
        """Test sending a draft."""
        # First call gets the draft, second call sends it
        call_count = [0]

        def fake_get(url: str, headers: dict[str, str]) -> str:
            return dump_json_str(
                {
                    "id": "draft123",
                    "conversationId": "conv1",
                    "subject": "Draft Subject",
                    "body": {"content": "Body"},
                    "from": {"emailAddress": {"address": "me@test.com"}},
                    "toRecipients": [{"emailAddress": {"address": "r@test.com"}}],
                    "ccRecipients": [],
                    "bccRecipients": [],
                    "isRead": False,
                    "isDraft": True,
                    "hasAttachments": False,
                    "importance": "normal",
                }
            )

        def fake_post(url: str, headers: dict[str, str], body: str) -> str:
            call_count[0] += 1
            return "{}"

        hooks.http_get = fake_get
        hooks.http_post = fake_post

        client = _OutlookEmailClient(access_token="token")
        email = client.send_draft(draft_id="draft123")

        assert email["is_draft"] is False
        assert email["folder_id"] == "sentitems"


class TestOutlookEmailClientReplyToEmail:
    """Tests for _OutlookEmailClient.reply_to_email()."""

    def test_reply_to_email_success(self) -> None:
        """Test replying to an email."""

        def fake_get(url: str, headers: dict[str, str]) -> str:
            return dump_json_str(
                {
                    "id": "original123",
                    "conversationId": "conv1",
                    "subject": "Original Subject",
                    "body": {"content": "Original body"},
                    "from": {"emailAddress": {"address": "sender@test.com"}},
                    "toRecipients": [],
                    "ccRecipients": [{"emailAddress": {"address": "cc@test.com"}}],
                    "bccRecipients": [],
                    "isRead": True,
                    "isDraft": False,
                    "hasAttachments": False,
                }
            )

        hooks.http_get = fake_get
        hooks.http_post = make_fake_http_post("{}")

        client = _OutlookEmailClient(access_token="token")
        reply = client.reply_to_email(email_id="original123", body="My reply")

        assert reply["subject"] == "Re: Original Subject"
        assert reply["to"][0]["address"] == "sender@test.com"

    def test_reply_all(self) -> None:
        """Test reply all."""

        def fake_get(url: str, headers: dict[str, str]) -> str:
            return dump_json_str(
                {
                    "id": "original123",
                    "conversationId": "conv1",
                    "subject": "Original",
                    "body": {"content": "Body"},
                    "from": {"emailAddress": {"address": "sender@test.com"}},
                    "toRecipients": [],
                    "ccRecipients": [{"emailAddress": {"address": "cc@test.com"}}],
                    "bccRecipients": [],
                    "isRead": True,
                    "isDraft": False,
                    "hasAttachments": False,
                }
            )

        captured_url: list[str] = []

        def capture_post(url: str, headers: dict[str, str], body: str) -> str:
            captured_url.append(url)
            return "{}"

        hooks.http_get = fake_get
        hooks.http_post = capture_post

        client = _OutlookEmailClient(access_token="token")
        reply = client.reply_to_email(email_id="original123", body="Reply", reply_all=True)

        assert "replyAll" in captured_url[0]
        assert len(reply["cc"]) == 1


class TestOutlookEmailClientDeleteEmail:
    """Tests for _OutlookEmailClient.delete_email()."""

    def test_delete_email_to_trash(self) -> None:
        """Test moving email to trash."""
        hooks.http_post = make_fake_http_post("{}")

        client = _OutlookEmailClient(access_token="token")
        client.delete_email(email_id="msg123")

        # No exception means success

    def test_delete_email_permanent(self) -> None:
        """Test permanent deletion."""
        hooks.http_delete = make_fake_http_delete()

        client = _OutlookEmailClient(access_token="token")
        client.delete_email(email_id="msg123", permanent=True)

        # No exception means success

    def test_delete_email_connection_error(self) -> None:
        """Test connection error on permanent delete."""
        hooks.http_delete = make_raising_http_delete(ConnectionError("Failed"))

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.delete_email(email_id="msg123", permanent=True)
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR

    def test_delete_email_http_error(self) -> None:
        """Test HTTP error on permanent delete."""
        error = FakeHTTPError(404, "Not found")
        hooks.http_delete = make_raising_http_delete(error)

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.delete_email(email_id="nonexistent", permanent=True)
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_NOT_FOUND


class TestOutlookEmailClientMoveEmail:
    """Tests for _OutlookEmailClient.move_email()."""

    def test_move_email_success(self) -> None:
        """Test moving an email."""
        response = dump_json_str(
            {
                "id": "msg123",
                "parentFolderId": "archive",
                "subject": "Moved email",
                "body": {"content": "Body"},
                "from": {},
                "toRecipients": [],
                "ccRecipients": [],
                "bccRecipients": [],
                "isRead": True,
                "isDraft": False,
                "hasAttachments": False,
            }
        )
        hooks.http_post = make_fake_http_post(response)

        client = _OutlookEmailClient(access_token="token")
        email = client.move_email(email_id="msg123", destination_folder_id="archive")

        assert email["folder_id"] == "archive"


class TestOutlookEmailClientListFolders:
    """Tests for _OutlookEmailClient.list_folders()."""

    def test_list_folders_success(self) -> None:
        """Test listing folders."""
        response = dump_json_str(
            {
                "value": [
                    {
                        "id": "folder1",
                        "displayName": "Inbox",
                        "unreadItemCount": 5,
                        "totalItemCount": 100,
                    },
                    {
                        "id": "folder2",
                        "displayName": "Sent Items",
                        "unreadItemCount": 0,
                        "totalItemCount": 50,
                    },
                ],
            }
        )
        hooks.http_get = make_fake_http_get(response)

        client = _OutlookEmailClient(access_token="token")
        folders = client.list_folders()

        assert len(folders) == 2
        assert folders[0]["name"] == "Inbox"
        assert folders[0]["folder_type"] == "inbox"
        assert folders[0]["unread_count"] == 5

    def test_list_folders_empty(self) -> None:
        """Test listing folders when none exist."""
        response = dump_json_str({"value": []})
        hooks.http_get = make_fake_http_get(response)

        client = _OutlookEmailClient(access_token="token")
        folders = client.list_folders()

        assert folders == ()

    def test_list_folders_non_list_value(self) -> None:
        """Test handling non-list value."""
        response = dump_json_str({"value": "not a list"})
        hooks.http_get = make_fake_http_get(response)

        client = _OutlookEmailClient(access_token="token")
        folders = client.list_folders()

        assert folders == ()

    def test_list_folders_skips_non_dict(self) -> None:
        """Test that non-dict folders are skipped."""
        response = dump_json_str(
            {
                "value": [
                    {"id": "f1", "displayName": "Inbox"},
                    "not a dict",
                ],
            }
        )
        hooks.http_get = make_fake_http_get(response)

        client = _OutlookEmailClient(access_token="token")
        folders = client.list_folders()

        assert len(folders) == 1


class TestOutlookEmailClientGetAttachment:
    """Tests for _OutlookEmailClient.get_attachment()."""

    def test_get_attachment_success(self) -> None:
        """Test getting an attachment."""
        response = dump_json_str(
            {
                "id": "att123",
                "name": "document.pdf",
                "contentType": "application/pdf",
                "size": 1024,
                "contentBytes": "SGVsbG8gV29ybGQ=",
            }
        )
        hooks.http_get = make_fake_http_get(response)

        client = _OutlookEmailClient(access_token="token")
        attachment = client.get_attachment(email_id="msg123", attachment_id="att123")

        assert attachment["id"] == "att123"
        assert attachment["name"] == "document.pdf"
        assert attachment["content_bytes"] == "SGVsbG8gV29ybGQ="


class TestOutlookEmailClientPatch:
    """Tests for _OutlookEmailClient._patch() method."""

    def test_patch_success(self) -> None:
        """Test successful PATCH request."""
        response = dump_json_str({"id": "msg123", "isRead": True})
        hooks.http_patch = make_fake_http_patch(response)

        client = _OutlookEmailClient(access_token="token")
        result = client._patch("/me/messages/msg123", {"isRead": True})

        assert result["isRead"] is True

    def test_patch_connection_error(self) -> None:
        """Test connection error on PATCH."""
        hooks.http_patch = make_raising_http_patch(ConnectionError("Failed"))

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client._patch("/me/messages/msg123", {"isRead": True})
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR

    def test_patch_http_error(self) -> None:
        """Test HTTP error on PATCH."""
        error = FakeHTTPError(404, "Not found")
        hooks.http_patch = make_raising_http_patch(error)

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client._patch("/me/messages/msg123", {"isRead": True})
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_NOT_FOUND

    def test_patch_invalid_json(self) -> None:
        """Test invalid JSON response on PATCH."""
        hooks.http_patch = make_fake_http_patch("not json")

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client._patch("/me/messages/msg123", {"isRead": True})
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR


class TestOutlookEmailClientErrorHandling:
    """Tests for error handling in various paths."""

    def test_handle_error_folder_not_found(self) -> None:
        """Test folder not found error.

        Note: Need to use a path containing 'folder' but not 'messages'
        since _handle_error checks 'messages' first.
        """
        error = FakeHTTPError(404, "Folder not found")
        hooks.http_get = make_raising_http_get(error)

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            # Use _get directly with a folder-only path
            client._get("/me/mailFolders/nonexistent")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.FOLDER_NOT_FOUND

    def test_handle_error_draft_not_found(self) -> None:
        """Test draft not found error.

        Note: Need to use a path containing 'draft' but not 'messages' or 'folder'.
        """
        error = FakeHTTPError(404, "Draft not found")
        hooks.http_get = make_raising_http_get(error)

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            # Use _get directly with a draft-only path
            client._get("/me/drafts/draft123")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.DRAFT_NOT_FOUND

    def test_handle_error_generic_not_found(self) -> None:
        """Test generic 404 error."""
        error = FakeHTTPError(404, "Resource not found")
        hooks.http_get = make_raising_http_get(error)

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client._get("/me/unknown")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_NOT_FOUND

    def test_handle_error_non_404(self) -> None:
        """Test non-404 HTTP error."""
        error = FakeHTTPError(500, "Internal server error")
        hooks.http_get = make_raising_http_get(error)

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.get_email(email_id="msg123")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR

    def test_post_invalid_json(self) -> None:
        """Test invalid JSON response on POST."""
        hooks.http_post = make_fake_http_post("not json")

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.create_draft(to=("r@test.com",), subject="Test", body="Body")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR

    def test_post_http_error(self) -> None:
        """Test HTTP error on POST."""
        error = FakeHTTPError(400, "Bad request")
        hooks.http_post = make_raising_http_post(error)

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.create_draft(to=("r@test.com",), subject="Test", body="Body")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR

    def test_get_os_error_without_http_protocol(self) -> None:
        """Test OSError on GET that's not an HTTPErrorProtocol."""
        hooks.http_get = make_raising_http_get(OSError("Socket error"))

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.get_email(email_id="msg123")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR

    def test_post_os_error_without_http_protocol(self) -> None:
        """Test OSError on POST that's not an HTTPErrorProtocol."""
        hooks.http_post = make_raising_http_post(OSError("Socket error"))

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.create_draft(to=("r@test.com",), subject="Test", body="Body")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR

    def test_patch_os_error_without_http_protocol(self) -> None:
        """Test OSError on PATCH that's not an HTTPErrorProtocol."""
        hooks.http_patch = make_raising_http_patch(OSError("Socket error"))

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client._patch("/me/messages/msg123", {"isRead": True})
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR

    def test_delete_os_error_without_http_protocol(self) -> None:
        """Test OSError on DELETE that's not an HTTPErrorProtocol."""
        hooks.http_delete = make_raising_http_delete(OSError("Socket error"))

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.delete_email(email_id="msg123", permanent=True)
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR


class TestOutlookEmailClientDecodeMessage:
    """Tests for _OutlookEmailClient._decode_message edge cases."""

    def test_decode_message_with_non_dict_from(self) -> None:
        """Test decoding message when from field is not a dict."""
        response = dump_json_str(
            {
                "id": "msg123",
                "subject": "Test",
                "body": {"content": "Body"},
                "from": "not a dict",  # Invalid format
                "toRecipients": [],
                "ccRecipients": [],
                "bccRecipients": [],
            }
        )
        hooks.http_get = make_fake_http_get(response)

        client = _OutlookEmailClient(access_token="token")
        email = client.get_email(email_id="msg123")

        # Should have empty from address
        assert email["from_address"]["address"] == ""
        assert email["from_address"]["name"] == ""

    def test_decode_message_with_non_dict_body(self) -> None:
        """Test decoding message when body field is not a dict."""
        response = dump_json_str(
            {
                "id": "msg123",
                "subject": "Test",
                "body": "not a dict",  # Invalid format
                "from": {"emailAddress": {"address": "test@example.com"}},
                "toRecipients": [],
                "ccRecipients": [],
                "bccRecipients": [],
            }
        )
        hooks.http_get = make_fake_http_get(response)

        client = _OutlookEmailClient(access_token="token")
        email = client.get_email(email_id="msg123")

        # Should have empty body
        assert email["body"] == ""
        assert email["body_type"] == "text"


class TestOutlookEmailClientListEmailsEdgeCases:
    """Tests for edge cases in list_emails."""

    def test_list_emails_with_non_list_value(self) -> None:
        """Test list_emails when API returns non-list value."""
        response = dump_json_str({"value": "not a list"})
        hooks.http_get = make_fake_http_get(response)

        client = _OutlookEmailClient(access_token="token")
        result = client.list_emails()

        # Should return empty result
        assert result["emails"] == ()
