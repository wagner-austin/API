"""Microsoft Outlook email client using Graph API.

Implements EmailClientProtocol for Microsoft Graph Mail API.
"""

from __future__ import annotations

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

from platform_email.config import OUTLOOK_API_BASE
from platform_email.testing import HTTPErrorProtocol, hooks
from platform_email.types import (
    Attachment,
    BodyType,
    Draft,
    Email,
    EmailAddress,
    EmailListResult,
    Folder,
    FolderType,
)


def _decode_email_address(data: JSONObject) -> EmailAddress:
    """Decode an email address from Graph API format.

    Args:
        data: JSON object with emailAddress field.

    Returns:
        EmailAddress.
    """
    email_addr_raw = data.get("emailAddress")
    if not isinstance(email_addr_raw, dict):
        return EmailAddress(address="", name="")
    return EmailAddress(
        address=optional_str(email_addr_raw, "address") or "",
        name=optional_str(email_addr_raw, "name") or "",
    )


def _decode_recipients(items: list[JSONValue]) -> tuple[EmailAddress, ...]:
    """Decode a list of recipients from Graph API format.

    Args:
        items: List of recipient JSON objects.

    Returns:
        Tuple of EmailAddress.
    """
    result: list[EmailAddress] = []
    for item in items:
        if isinstance(item, dict):
            result.append(_decode_email_address(item))
    return tuple(result)


def _decode_folder_type(display_name: str) -> FolderType:
    """Map Outlook folder display name to FolderType.

    Args:
        display_name: Folder display name from Graph API.

    Returns:
        FolderType literal.
    """
    lower_name = display_name.lower()
    if lower_name == "inbox":
        return "inbox"
    if lower_name in ("sent items", "sent"):
        return "sent"
    if lower_name == "drafts":
        return "drafts"
    if lower_name in ("deleted items", "trash"):
        return "trash"
    if lower_name in ("junk email", "spam", "junk"):
        return "spam"
    if lower_name == "archive":
        return "archive"
    return "custom"


def _decode_importance(value: str | None) -> BodyType:
    """Decode importance level.

    Args:
        value: Importance value from Graph API.

    Returns:
        EmailImportance literal.
    """
    # This is actually mapping body type not importance - fix
    if value == "html":
        return "html"
    return "text"


class _OutlookEmailClient:
    """Microsoft Outlook email client using Graph API."""

    def __init__(self, *, access_token: str) -> None:
        """Initialize the client.

        Args:
            access_token: OAuth access token.
        """
        self._access_token = access_token

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
            if "folder" in context.lower():
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
        """Make a GET request to Graph API.

        Args:
            path: API path (appended to base URL).

        Returns:
            Parsed JSON response.

        Raises:
            AppError[EmailErrorCode]: On request failure.
        """
        url = f"{OUTLOOK_API_BASE}{path}"
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
        """Make a POST request to Graph API.

        Args:
            path: API path.
            body: Request body.

        Returns:
            Parsed JSON response.

        Raises:
            AppError[EmailErrorCode]: On request failure.
        """
        url = f"{OUTLOOK_API_BASE}{path}"
        body_str = dump_json_str(body)
        try:
            response = hooks.http_post(url, self._headers(), body_str)
        except ConnectionError as e:
            msg = f"Request failed: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e
        except OSError as e:
            if isinstance(e, HTTPErrorProtocol):
                status_code = e.code
                resp_body: str = e.read().decode("utf-8")
                self._handle_error(status_code, resp_body, path)
            msg = f"Request failed: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e

        try:
            raw_value = load_json_str(response)
            data = narrow_json_to_dict(raw_value)
        except (InvalidJsonError, JSONTypeError) as e:
            msg = f"Invalid response from API: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e

        return data

    def _patch(self, path: str, body: JSONObject) -> JSONObject:
        """Make a PATCH request to Graph API.

        Args:
            path: API path.
            body: Request body.

        Returns:
            Parsed JSON response.

        Raises:
            AppError[EmailErrorCode]: On request failure.
        """
        url = f"{OUTLOOK_API_BASE}{path}"
        body_str = dump_json_str(body)
        try:
            response = hooks.http_patch(url, self._headers(), body_str)
        except ConnectionError as e:
            msg = f"Request failed: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e
        except OSError as e:
            if isinstance(e, HTTPErrorProtocol):
                status_code = e.code
                resp_body: str = e.read().decode("utf-8")
                self._handle_error(status_code, resp_body, path)
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
        """Make a DELETE request to Graph API.

        Args:
            path: API path.

        Raises:
            AppError[EmailErrorCode]: On request failure.
        """
        url = f"{OUTLOOK_API_BASE}{path}"
        try:
            hooks.http_delete(url, self._headers())
        except ConnectionError as e:
            msg = f"Delete request failed: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e
        except OSError as e:
            if isinstance(e, HTTPErrorProtocol):
                status_code: int = e.code
                resp_body: str = e.read().decode("utf-8")
                self._handle_error(status_code, resp_body, path)
            msg = f"Delete request failed: {e}"
            raise AppError(EmailErrorCode.EMAIL_API_ERROR, msg, http_status=500) from e

    def _decode_message(self, data: JSONObject) -> Email:
        """Decode a message from Graph API format.

        Args:
            data: JSON object representing a message.

        Returns:
            Email.
        """
        # Get sender
        from_raw = data.get("from")
        from_addr: EmailAddress
        if isinstance(from_raw, dict):
            from_addr = _decode_email_address(from_raw)
        else:
            from_addr = EmailAddress(address="", name="")

        # Get recipients
        to_raw = data.get("toRecipients")
        to_list: list[JSONValue] = to_raw if isinstance(to_raw, list) else []
        cc_raw = data.get("ccRecipients")
        cc_list: list[JSONValue] = cc_raw if isinstance(cc_raw, list) else []
        bcc_raw = data.get("bccRecipients")
        bcc_list: list[JSONValue] = bcc_raw if isinstance(bcc_raw, list) else []

        # Get body
        body_raw = data.get("body")
        body_content = ""
        body_type: BodyType = "text"
        if isinstance(body_raw, dict):
            body_content = optional_str(body_raw, "content") or ""
            content_type = optional_str(body_raw, "contentType") or "text"
            if content_type.lower() == "html":
                body_type = "html"

        # Get importance
        importance_raw = optional_str(data, "importance") or "normal"
        from platform_email.types.email import EmailImportance

        final_importance: EmailImportance
        if importance_raw == "low":
            final_importance = "low"
        elif importance_raw == "high":
            final_importance = "high"
        else:
            final_importance = "normal"

        return Email(
            id=require_str(data, "id"),
            thread_id=optional_str(data, "conversationId") or "",
            folder_id=optional_str(data, "parentFolderId") or "",
            subject=optional_str(data, "subject") or "",
            body=body_content,
            body_type=body_type,
            from_address=from_addr,
            to=_decode_recipients(to_list),
            cc=_decode_recipients(cc_list),
            bcc=_decode_recipients(bcc_list),
            sent_at=optional_str(data, "sentDateTime") or "",
            received_at=optional_str(data, "receivedDateTime") or "",
            is_read=data.get("isRead") is True,
            is_draft=data.get("isDraft") is True,
            has_attachments=data.get("hasAttachments") is True,
            importance=final_importance,
        )

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
        to_recipients: list[JSONValue] = []
        for addr in to:
            recipient: JSONObject = {"emailAddress": {"address": addr}}
            to_recipients.append(recipient)

        cc_recipients: list[JSONValue] = []
        for addr in cc:
            recipient_cc: JSONObject = {"emailAddress": {"address": addr}}
            cc_recipients.append(recipient_cc)

        bcc_recipients: list[JSONValue] = []
        for addr in bcc:
            recipient_bcc: JSONObject = {"emailAddress": {"address": addr}}
            bcc_recipients.append(recipient_bcc)

        content_type = "HTML" if body_type == "html" else "Text"

        message_body: JSONObject = {
            "subject": subject,
            "body": {
                "contentType": content_type,
                "content": body,
            },
            "toRecipients": to_recipients,
            "ccRecipients": cc_recipients,
            "bccRecipients": bcc_recipients,
        }

        # Add attachments if any
        if attachments:
            att_list: list[JSONValue] = []
            for att in attachments:
                att_obj: JSONObject = {
                    "@odata.type": "#microsoft.graph.fileAttachment",
                    "name": att["name"],
                    "contentType": att["content_type"],
                    "contentBytes": att["content_bytes"] or "",
                }
                att_list.append(att_obj)
            message_body["attachments"] = att_list

        request_body: JSONObject = {
            "message": message_body,
            "saveToSentItems": True,
        }

        # Send and get back the sent message
        self._post("/me/sendMail", request_body)

        # Since sendMail doesn't return the message, we need to get from sent items
        # For now, create a placeholder - in real usage you'd query sent items
        from_addr = EmailAddress(address="me@outlook.com", name="")

        to_addrs: list[EmailAddress] = []
        for addr in to:
            to_addrs.append(EmailAddress(address=addr, name=""))
        cc_addrs: list[EmailAddress] = []
        for addr in cc:
            cc_addrs.append(EmailAddress(address=addr, name=""))
        bcc_addrs: list[EmailAddress] = []
        for addr in bcc:
            bcc_addrs.append(EmailAddress(address=addr, name=""))

        return Email(
            id="sent_placeholder",
            thread_id="",
            folder_id="sentitems",
            subject=subject,
            body=body,
            body_type=body_type,
            from_address=from_addr,
            to=tuple(to_addrs),
            cc=tuple(cc_addrs),
            bcc=tuple(bcc_addrs),
            sent_at="",
            received_at="",
            is_read=True,
            is_draft=False,
            has_attachments=len(attachments) > 0,
            importance="normal",
        )

    def get_email(self, *, email_id: str) -> Email:
        """Get a single email by ID.

        Args:
            email_id: ID of the email to retrieve.

        Returns:
            The Email.

        Raises:
            AppError[EmailErrorCode]: If email not found.
        """
        path = f"/me/messages/{email_id}"
        data = self._get(path)
        return self._decode_message(data)

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
            folder_id: Folder to list from. None for all mail.
            query: Search query string.
            max_results: Maximum number of results.
            page_token: Token for pagination.

        Returns:
            EmailListResult with emails and next page token.
        """
        if folder_id is not None:
            base_path = f"/me/mailFolders/{folder_id}/messages"
        else:
            base_path = "/me/messages"

        params: dict[str, str] = {"$top": str(max_results)}

        if query is not None:
            params["$search"] = f'"{query}"'

        if page_token is not None:
            # page_token is a full URL for Graph API, use skiptoken
            params["$skiptoken"] = page_token

        path = base_path + "?" + urllib.parse.urlencode(params)
        data = self._get(path)

        messages_raw = data.get("value")
        emails: list[Email] = []
        if isinstance(messages_raw, list):
            for msg in messages_raw:
                if isinstance(msg, dict):
                    emails.append(self._decode_message(msg))

        next_link = optional_str(data, "@odata.nextLink")
        next_token: str | None = None
        if next_link is not None and "$skiptoken=" in next_link:
            # Extract skiptoken from next link
            start = next_link.find("$skiptoken=") + len("$skiptoken=")
            end = next_link.find("&", start)
            next_token = next_link[start:] if end == -1 else next_link[start:end]

        return EmailListResult(
            emails=tuple(emails),
            next_page_token=next_token,
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
        to_recipients: list[JSONValue] = []
        for addr in to:
            recipient: JSONObject = {"emailAddress": {"address": addr}}
            to_recipients.append(recipient)

        cc_recipients: list[JSONValue] = []
        for addr in cc:
            recipient_cc: JSONObject = {"emailAddress": {"address": addr}}
            cc_recipients.append(recipient_cc)

        bcc_recipients: list[JSONValue] = []
        for addr in bcc:
            recipient_bcc: JSONObject = {"emailAddress": {"address": addr}}
            bcc_recipients.append(recipient_bcc)

        content_type = "HTML" if body_type == "html" else "Text"

        request_body: JSONObject = {
            "subject": subject,
            "body": {
                "contentType": content_type,
                "content": body,
            },
            "toRecipients": to_recipients,
            "ccRecipients": cc_recipients,
            "bccRecipients": bcc_recipients,
        }

        data = self._post("/me/messages", request_body)

        # Decode recipients back
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
            id=require_str(data, "id"),
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
        # First get the draft to get its details
        draft_msg = self.get_email(email_id=draft_id)

        # Send the draft
        self._post(f"/me/messages/{draft_id}/send", {})

        # Return the message converted to sent email
        return Email(
            id=draft_id,
            thread_id=draft_msg["thread_id"],
            folder_id="sentitems",
            subject=draft_msg["subject"],
            body=draft_msg["body"],
            body_type=draft_msg["body_type"],
            from_address=draft_msg["from_address"],
            to=draft_msg["to"],
            cc=draft_msg["cc"],
            bcc=draft_msg["bcc"],
            sent_at="",  # Will be filled by server
            received_at="",
            is_read=True,
            is_draft=False,
            has_attachments=draft_msg["has_attachments"],
            importance=draft_msg["importance"],
        )

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

        request_body: JSONObject = {
            "comment": body,
        }

        action = "replyAll" if reply_all else "reply"
        self._post(f"/me/messages/{email_id}/{action}", request_body)

        # Return placeholder for sent reply
        return Email(
            id="reply_placeholder",
            thread_id=original["thread_id"],
            folder_id="sentitems",
            subject=f"Re: {original['subject']}",
            body=body,
            body_type=body_type,
            from_address=EmailAddress(address="me@outlook.com", name=""),
            to=(original["from_address"],),
            cc=original["cc"] if reply_all else (),
            bcc=(),
            sent_at="",
            received_at="",
            is_read=True,
            is_draft=False,
            has_attachments=False,
            importance="normal",
        )

    def delete_email(self, *, email_id: str, permanent: bool = False) -> None:
        """Delete an email.

        Args:
            email_id: ID of the email to delete.
            permanent: If True, permanently delete. If False, move to trash.

        Raises:
            AppError[EmailErrorCode]: If email not found.
        """
        if permanent:
            self._delete(f"/me/messages/{email_id}")
        else:
            # Move to deleted items folder
            body: JSONObject = {"destinationId": "deleteditems"}
            self._post(f"/me/messages/{email_id}/move", body)

    def move_email(self, *, email_id: str, destination_folder_id: str) -> Email:
        """Move an email to a folder.

        Args:
            email_id: ID of the email to move.
            destination_folder_id: ID of the destination folder.

        Returns:
            The moved Email.

        Raises:
            AppError[EmailErrorCode]: If email or folder not found.
        """
        body: JSONObject = {"destinationId": destination_folder_id}
        data = self._post(f"/me/messages/{email_id}/move", body)
        return self._decode_message(data)

    def list_folders(self) -> tuple[Folder, ...]:
        """List all email folders.

        Returns:
            Tuple of all folders.
        """
        data = self._get("/me/mailFolders")
        folders_raw = data.get("value")
        if not isinstance(folders_raw, list):
            return ()

        folders: list[Folder] = []
        for folder in folders_raw:
            if not isinstance(folder, dict):
                continue
            display_name = optional_str(folder, "displayName") or ""
            unread_raw = folder.get("unreadItemCount")
            total_raw = folder.get("totalItemCount")
            folders.append(
                Folder(
                    id=require_str(folder, "id"),
                    name=display_name,
                    folder_type=_decode_folder_type(display_name),
                    unread_count=unread_raw if isinstance(unread_raw, int) else 0,
                    total_count=total_raw if isinstance(total_raw, int) else 0,
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
        path = f"/me/messages/{email_id}/attachments/{attachment_id}"
        data = self._get(path)

        content_bytes = optional_str(data, "contentBytes")
        size_raw = data.get("size")

        return Attachment(
            id=require_str(data, "id"),
            name=optional_str(data, "name") or "",
            content_type=optional_str(data, "contentType") or "application/octet-stream",
            size=size_raw if isinstance(size_raw, int) else 0,
            content_bytes=content_bytes,
        )


__all__ = [
    "_OutlookEmailClient",
]
