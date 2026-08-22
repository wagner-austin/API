"""Gmail provider: replies, deletes, folders, attachments, errors."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_core.errors import AppError, EmailErrorCode
from platform_core.json_utils import dump_json_str

from platform_email.fake_hooks import (
    make_fake_http_delete,
    make_fake_http_get,
    make_fake_http_post,
    make_raising_http_delete,
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


@pytest.fixture(autouse=True)
def _reset_hooks_after_test() -> Generator[None, None, None]:
    """Reset hooks after each test."""
    yield
    reset_hooks()


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
