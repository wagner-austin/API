"""Google Gmail email client using Gmail API.

Implements EmailClientProtocol for Gmail API.
"""

from __future__ import annotations

import base64
import urllib.parse

from platform_core.errors import AppError, EmailErrorCode
from platform_core.json_utils import (
    InvalidJsonError,
    JSONObject,
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    optional_str,
    require_str,
)

from platform_email.config import GMAIL_API_BASE
from platform_email.providers.gmail_decode import (
    _create_mime_message,
    _decode_folder_type,
    _decode_message,
)
from platform_email.testing import HTTPErrorProtocol, hooks
from platform_email.types import (
    Attachment,
    BodyType,
    Draft,
    Email,
    EmailAddress,
    EmailListResult,
    Folder,
)


class _GmailEmailClient:
    """Google Gmail email client using Gmail API."""

    def __init__(self, *, access_token: str) -> None:
        """Initialize the client.

        Args:
            access_token: OAuth access token.
        """
        self._access_token = access_token
        self._user_id = "me"

    def _headers(self) -> dict[str, str]:
        """Get standard request headers.

        Returns:
            Headers dict with Authorization and Content-Type.
        """
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

    def _handle_error(self, status_code: int, message: str, context: str) -> None:
        """Handle HTTP error response.

        Args:
            status_code: HTTP status code.
            message: Error message.
            context: Context for error (URL path).

        Raises:
            AppError[EmailErrorCode]: Always raises.
        """
        if status_code == 404:
            if "messages" in context or "message" in context:
                msg = f"Email not found: {message}"
                raise AppError(EmailErrorCode.EMAIL_NOT_FOUND, msg, http_status=404)
            if "label" in context.lower():
                msg = f"Folder not found: {message}"
                raise AppError(EmailErrorCode.FOLDER_NOT_FOUND, msg, http_status=404)
            if "draft" in context.lower():
                msg = f"Draft not found: {message}"
                raise AppError(EmailErrorCode.DRAFT_NOT_FOUND, msg, http_status=404)
            msg = f"Resource not found: {message}"
            raise AppError(EmailErrorCode.EMAIL_NOT_FOUND, msg, http_status=404)
        msg = f"API error ({status_code}): {message}"
        raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=status_code)

    def _get(self, path: str) -> JSONObject:
        """Make a GET request to Gmail API.

        Args:
            path: API path (appended to base URL).

        Returns:
            Parsed JSON response.

        Raises:
            AppError[EmailErrorCode]: On request failure.
        """
        url = f"{GMAIL_API_BASE}{path}"
        try:
            response = hooks.http_get(url, self._headers())
        except ConnectionError as e:
            msg = f"Request failed: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e
        except OSError as e:
            if isinstance(e, HTTPErrorProtocol):
                status_code: int = e.code
                body_text: str = e.read().decode("utf-8")
                self._handle_error(status_code, body_text, path)
            msg = f"Request failed: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e

        try:
            raw_value = load_json_str(response)
            data = narrow_json_to_dict(raw_value)
        except (InvalidJsonError, JSONTypeError) as e:
            msg = f"Invalid response from API: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e

        return data

    def _post(self, path: str, body: JSONObject) -> JSONObject:
        """Make a POST request to Gmail API.

        Args:
            path: API path.
            body: Request body.

        Returns:
            Parsed JSON response.

        Raises:
            AppError[EmailErrorCode]: On request failure.
        """
        url = f"{GMAIL_API_BASE}{path}"
        body_str = dump_json_str(body)
        try:
            response = hooks.http_post(url, self._headers(), body_str)
        except ConnectionError as e:
            msg = f"Request failed: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e
        except OSError as e:
            if isinstance(e, HTTPErrorProtocol):
                status_code: int = e.code
                body_resp: str = e.read().decode("utf-8")
                self._handle_error(status_code, body_resp, path)
            msg = f"Request failed: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e

        try:
            raw_value = load_json_str(response)
            data = narrow_json_to_dict(raw_value)
        except (InvalidJsonError, JSONTypeError) as e:
            msg = f"Invalid response from API: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e

        return data

    def _delete(self, path: str) -> None:
        """Make a DELETE request to Gmail API.

        Args:
            path: API path.

        Raises:
            AppError[EmailErrorCode]: On request failure.
        """
        url = f"{GMAIL_API_BASE}{path}"
        try:
            hooks.http_delete(url, self._headers())
        except ConnectionError as e:
            msg = f"Delete request failed: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e
        except OSError as e:
            if isinstance(e, HTTPErrorProtocol):
                status_code: int = e.code
                body_resp: str = e.read().decode("utf-8")
                self._handle_error(status_code, body_resp, path)
            msg = f"Delete request failed: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e

    def send_email(
        self,
        *,
        to: tuple[str, ...],
        subject: str,
        body: str,
        body_type: BodyType = "text",
        cc: tuple[str, ...] = (),
        bcc: tuple[str, ...] = (),
        attachments: tuple[Attachment, ...] = (),
    ) -> Email:
        """Send an email.

        Args:
            to: Recipient email addresses.
            subject: Email subject.
            body: Email body content.
            body_type: Body content type (text or html).
            cc: CC recipient addresses.
            bcc: BCC recipient addresses.
            attachments: Email attachments.

        Returns:
            The sent Email.

        Raises:
            AppError[EmailErrorCode]: On send failure.
        """
        raw_message = _create_mime_message(
            to=to,
            subject=subject,
            body=body,
            body_type=body_type,
            cc=cc,
            bcc=bcc,
            attachments=attachments,
        )

        request_body: JSONObject = {"raw": raw_message}
        data = self._post(f"/users/{self._user_id}/messages/send", request_body)

        # Get the full message to return
        message_id = require_str(data, "id")
        return self.get_email(email_id=message_id)

    def get_email(self, *, email_id: str) -> Email:
        """Get a single email by ID.

        Args:
            email_id: ID of the email to retrieve.

        Returns:
            The Email.

        Raises:
            AppError[EmailErrorCode]: If email not found.
        """
        path = f"/users/{self._user_id}/messages/{email_id}?format=full"
        data = self._get(path)
        return _decode_message(data)

    def list_emails(
        self,
        *,
        folder_id: str | None = None,
        query: str | None = None,
        max_results: int = 50,
        page_token: str | None = None,
    ) -> EmailListResult:
        """List emails in a folder.

        Args:
            folder_id: Folder (label) to list from. None for all mail.
            query: Search query string.
            max_results: Maximum number of results.
            page_token: Token for pagination.

        Returns:
            EmailListResult with emails and next page token.
        """
        params: dict[str, str] = {"maxResults": str(max_results)}

        query_parts: list[str] = []
        if folder_id is not None:
            query_parts.append(f"label:{folder_id}")
        if query is not None:
            query_parts.append(query)
        if query_parts:
            params["q"] = " ".join(query_parts)
        if page_token is not None:
            params["pageToken"] = page_token

        path = f"/users/{self._user_id}/messages?" + urllib.parse.urlencode(params)
        data = self._get(path)

        messages_raw = data.get("messages")
        emails: list[Email] = []
        if isinstance(messages_raw, list):
            for msg_ref in messages_raw:
                if isinstance(msg_ref, dict):
                    msg_id = optional_str(msg_ref, "id")
                    if msg_id:
                        # Get full message
                        full_msg = self.get_email(email_id=msg_id)
                        emails.append(full_msg)

        return EmailListResult(
            emails=tuple(emails),
            next_page_token=optional_str(data, "nextPageToken"),
        )

    def search_emails(self, *, query: str, max_results: int = 50) -> tuple[Email, ...]:
        """Search emails.

        Args:
            query: Search query string.
            max_results: Maximum number of results.

        Returns:
            Tuple of matching emails.
        """
        result = self.list_emails(query=query, max_results=max_results)
        return result["emails"]

    def create_draft(
        self,
        *,
        to: tuple[str, ...],
        subject: str,
        body: str,
        body_type: BodyType = "text",
        cc: tuple[str, ...] = (),
        bcc: tuple[str, ...] = (),
    ) -> Draft:
        """Create a draft email.

        Args:
            to: Recipient email addresses.
            subject: Draft subject.
            body: Draft body content.
            body_type: Body content type.
            cc: CC recipient addresses.
            bcc: BCC recipient addresses.

        Returns:
            The created Draft.
        """
        raw_message = _create_mime_message(
            to=to,
            subject=subject,
            body=body,
            body_type=body_type,
            cc=cc,
            bcc=bcc,
            attachments=(),
        )

        request_body: JSONObject = {"message": {"raw": raw_message}}
        data = self._post(f"/users/{self._user_id}/drafts", request_body)

        draft_id = require_str(data, "id")

        to_addrs: list[EmailAddress] = []
        for addr in to:
            to_addrs.append(EmailAddress(address=addr, name=""))
        cc_addrs: list[EmailAddress] = []
        for addr in cc:
            cc_addrs.append(EmailAddress(address=addr, name=""))
        bcc_addrs: list[EmailAddress] = []
        for addr in bcc:
            bcc_addrs.append(EmailAddress(address=addr, name=""))

        return Draft(
            id=draft_id,
            subject=subject,
            body=body,
            body_type=body_type,
            to=tuple(to_addrs),
            cc=tuple(cc_addrs),
            bcc=tuple(bcc_addrs),
        )

    def send_draft(self, *, draft_id: str) -> Email:
        """Send a draft email.

        Args:
            draft_id: ID of the draft to send.

        Returns:
            The sent Email.

        Raises:
            AppError[EmailErrorCode]: If draft not found.
        """
        request_body: JSONObject = {"id": draft_id}
        data = self._post(f"/users/{self._user_id}/drafts/send", request_body)

        message_id = require_str(data, "id")
        return self.get_email(email_id=message_id)

    def reply_to_email(
        self,
        *,
        email_id: str,
        body: str,
        body_type: BodyType = "text",
        reply_all: bool = False,
    ) -> Email:
        """Reply to an email.

        Args:
            email_id: ID of the email to reply to.
            body: Reply body content.
            body_type: Body content type.
            reply_all: Whether to reply to all recipients.

        Returns:
            The sent reply Email.

        Raises:
            AppError[EmailErrorCode]: If email not found.
        """
        original = self.get_email(email_id=email_id)

        to_addrs = (original["from_address"]["address"],)
        cc_addrs: tuple[str, ...] = ()

        if reply_all:
            cc_list: list[str] = []
            for addr in original["to"]:
                cc_list.append(addr["address"])
            for addr in original["cc"]:
                cc_list.append(addr["address"])
            cc_addrs = tuple(cc_list)

        # Create reply message
        raw_message = _create_mime_message(
            to=to_addrs,
            subject=f"Re: {original['subject']}",
            body=body,
            body_type=body_type,
            cc=cc_addrs,
            bcc=(),
            attachments=(),
        )

        # Send as reply (with threadId)
        request_body: JSONObject = {
            "raw": raw_message,
            "threadId": original["thread_id"],
        }
        data = self._post(f"/users/{self._user_id}/messages/send", request_body)

        message_id = require_str(data, "id")
        return self.get_email(email_id=message_id)

    def delete_email(self, *, email_id: str, permanent: bool = False) -> None:
        """Delete an email.

        Args:
            email_id: ID of the email to delete.
            permanent: If True, permanently delete. If False, move to trash.

        Raises:
            AppError[EmailErrorCode]: If email not found.
        """
        if permanent:
            self._delete(f"/users/{self._user_id}/messages/{email_id}")
        else:
            self._post(f"/users/{self._user_id}/messages/{email_id}/trash", {})

    def move_email(self, *, email_id: str, destination_folder_id: str) -> Email:
        """Move an email to a folder (add label, remove others).

        Args:
            email_id: ID of the email to move.
            destination_folder_id: ID of the destination label.

        Returns:
            The moved Email.

        Raises:
            AppError[EmailErrorCode]: If email or folder not found.
        """
        # Modify labels: add new
        add_label_ids: list[JSONValue] = [destination_folder_id]

        request_body: JSONObject = {
            "addLabelIds": add_label_ids,
        }

        path = f"/users/{self._user_id}/messages/{email_id}/modify"
        headers = self._headers()
        url = f"{GMAIL_API_BASE}{path}"
        body_str = dump_json_str(request_body)

        try:
            response = hooks.http_post(url, headers, body_str)
        except OSError as e:
            msg = f"Request failed: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e

        try:
            raw_value = load_json_str(response)
            data = narrow_json_to_dict(raw_value)
        except (InvalidJsonError, JSONTypeError) as e:
            msg = f"Invalid response from API: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e

        return _decode_message(data, include_body=False)

    def list_folders(self) -> tuple[Folder, ...]:
        """List all email folders (labels).

        Returns:
            Tuple of all folders.
        """
        data = self._get(f"/users/{self._user_id}/labels")
        labels_raw = data.get("labels")
        if not isinstance(labels_raw, list):
            return ()

        folders: list[Folder] = []
        for label in labels_raw:
            if not isinstance(label, dict):
                continue
            label_name = optional_str(label, "name") or ""
            label_id = optional_str(label, "id") or ""

            # Get message counts if available
            messages_total = label.get("messagesTotal")
            messages_unread = label.get("messagesUnread")

            folders.append(
                Folder(
                    id=label_id,
                    name=label_name,
                    folder_type=_decode_folder_type(label_name),
                    unread_count=messages_unread if isinstance(messages_unread, int) else 0,
                    total_count=messages_total if isinstance(messages_total, int) else 0,
                )
            )

        return tuple(folders)

    def get_attachment(self, *, email_id: str, attachment_id: str) -> Attachment:
        """Get an email attachment with content.

        Args:
            email_id: ID of the email containing the attachment.
            attachment_id: ID of the attachment.

        Returns:
            Attachment with content_bytes populated.

        Raises:
            AppError[EmailErrorCode]: If email or attachment not found.
        """
        path = f"/users/{self._user_id}/messages/{email_id}/attachments/{attachment_id}"
        data = self._get(path)

        # Gmail returns data as base64url, convert to regular base64
        content_data = optional_str(data, "data")
        content_bytes: str | None = None
        if content_data:
            # Add padding and decode, then re-encode as standard base64
            try:
                decoded = base64.urlsafe_b64decode(content_data + "==")
                content_bytes = base64.b64encode(decoded).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                content_bytes = content_data

        size = data.get("size")

        return Attachment(
            id=attachment_id,
            name="",  # Name not returned by this endpoint
            content_type="application/octet-stream",
            size=size if isinstance(size, int) else 0,
            content_bytes=content_bytes,
        )


__all__ = [
    "_GmailEmailClient",
]
