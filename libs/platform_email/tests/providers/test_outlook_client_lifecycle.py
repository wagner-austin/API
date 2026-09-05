"""Outlook provider: moves, folders, attachments, errors, edges."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_core.errors import AppError, EmailErrorCode
from platform_core.json_utils import dump_json_str

from platform_email.fake_hooks import (
    make_fake_http_get,
    make_fake_http_send,
    make_raising_http_delete,
    make_raising_http_get,
    make_raising_http_send,
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
        hooks.http_post = make_fake_http_send(response)

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
        hooks.http_patch = make_fake_http_send(response)

        client = _OutlookEmailClient(access_token="token")
        result = client._patch("/me/messages/msg123", {"isRead": True})

        assert result["isRead"] is True

    def test_patch_connection_error(self) -> None:
        """Test connection error on PATCH."""
        hooks.http_patch = make_raising_http_send(ConnectionError("Failed"))

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client._patch("/me/messages/msg123", {"isRead": True})
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR

    def test_patch_http_error(self) -> None:
        """Test HTTP error on PATCH."""
        error = FakeHTTPError(404, "Not found")
        hooks.http_patch = make_raising_http_send(error)

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client._patch("/me/messages/msg123", {"isRead": True})
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_NOT_FOUND

    def test_patch_invalid_json(self) -> None:
        """Test invalid JSON response on PATCH."""
        hooks.http_patch = make_fake_http_send("not json")

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
        hooks.http_post = make_fake_http_send("not json")

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.create_draft(to=("r@test.com",), subject="Test", body="Body")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR

    def test_post_http_error(self) -> None:
        """Test HTTP error on POST."""
        error = FakeHTTPError(400, "Bad request")
        hooks.http_post = make_raising_http_send(error)

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
        hooks.http_post = make_raising_http_send(OSError("Socket error"))

        client = _OutlookEmailClient(access_token="token")
        with pytest.raises(AppError) as exc_info:
            client.create_draft(to=("r@test.com",), subject="Test", body="Body")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_API_ERROR

    def test_patch_os_error_without_http_protocol(self) -> None:
        """Test OSError on PATCH that's not an HTTPErrorProtocol."""
        hooks.http_patch = make_raising_http_send(OSError("Socket error"))

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
