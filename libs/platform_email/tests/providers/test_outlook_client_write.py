"""Outlook provider: send, list, search, drafts, replies."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_core.errors import AppError, EmailErrorCode
from platform_core.json_utils import dump_json_str

from platform_email.fake_hooks import (
    make_fake_http_delete,
    make_fake_http_get,
    make_fake_http_send,
    make_raising_http_delete,
)
from platform_email.providers.outlook import (
    _OutlookEmailClient,
)
from platform_email.testing import (
    FakeHTTPError,
    hooks,
    reset_hooks,
)


@pytest.fixture(autouse=True)
def _reset_hooks_after_test() -> Generator[None, None, None]:
    """Reset hooks after each test."""
    yield
    reset_hooks()


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
        hooks.http_post = make_fake_http_send(response)

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
        hooks.http_post = make_fake_http_send(response)

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
        hooks.http_post = make_fake_http_send("{}")

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
        hooks.http_post = make_fake_http_send("{}")

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
