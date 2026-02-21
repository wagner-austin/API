"""Google Gmail email client using Gmail API.

Implements EmailClientProtocol for Gmail API.
"""

from __future__ import annotations

import base64
import email.utils
import urllib.parse
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

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
from platform_email.testing import HTTPErrorProtocol, hooks
from platform_email.types import (
    Attachment,
    BodyType,
    Draft,
    Email,
    EmailAddress,
    EmailImportance,
    EmailListResult,
    Folder,
    FolderType,
)


def _parse_email_address(header: str) -> EmailAddress:
    """Parse an email address from a header string.

    Args:
        header: Email header like "Name <email@example.com>" or just "email@example.com".

    Returns:
        EmailAddress with address and name.
    """
    name, addr = email.utils.parseaddr(header)
    return EmailAddress(address=addr, name=name)


def _decode_folder_type(label_name: str) -> FolderType:
    """Map Gmail label name to FolderType.

    Args:
        label_name: Gmail label name.

    Returns:
        FolderType literal.
    """
    upper_name = label_name.upper()
    if upper_name == "INBOX":
        return "inbox"
    if upper_name == "SENT":
        return "sent"
    if upper_name == "DRAFT":
        return "drafts"
    if upper_name == "TRASH":
        return "trash"
    if upper_name == "SPAM":
        return "spam"
    if upper_name in ("ARCHIVE", "ALL"):
        return "archive"
    return "custom"


def _get_header_value(headers: list[JSONValue], name: str) -> str:
    """Get a header value from Gmail message headers.

    Args:
        headers: List of header objects with name and value.
        name: Header name to find (case-insensitive).

    Returns:
        Header value or empty string if not found.
    """
    name_lower = name.lower()
    for header in headers:
        if not isinstance(header, dict):
            continue
        header_name = optional_str(header, "name") or ""
        if header_name.lower() == name_lower:
            return optional_str(header, "value") or ""
    return ""


def _try_decode_base64(data: str) -> str | None:
    """Try to decode base64url-encoded data.

    Args:
        data: Base64url-encoded string.

    Returns:
        Decoded string or None if decoding fails.
    """
    try:
        return base64.urlsafe_b64decode(data + "==").decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def _decode_simple_body(payload: JSONObject) -> tuple[str, BodyType] | None:
    """Try to decode a simple (non-multipart) body.

    Args:
        payload: Gmail message payload.

    Returns:
        Tuple of (body_content, body_type) or None if no simple body.
    """
    body_data = payload.get("body")
    if not isinstance(body_data, dict):
        return None
    data = optional_str(body_data, "data")
    if not data:
        return None
    decoded = _try_decode_base64(data)
    if decoded is None:
        return None
    mime_type = optional_str(payload, "mimeType") or "text/plain"
    body_type: BodyType = "html" if "html" in mime_type.lower() else "text"
    return decoded, body_type


def _decode_multipart_body(parts: list[JSONValue]) -> tuple[str, BodyType] | None:
    """Decode body from multipart message parts.

    Args:
        parts: List of message parts.

    Returns:
        Tuple of (body_content, body_type) or None if no body found.
    """
    html_body = ""
    text_body = ""
    for part in parts:
        if not isinstance(part, dict):
            continue
        mime_type = optional_str(part, "mimeType") or ""
        part_body = part.get("body")
        if not isinstance(part_body, dict):
            continue
        data = optional_str(part_body, "data")
        if not data:
            continue
        decoded = _try_decode_base64(data)
        if decoded is None:
            continue
        if mime_type == "text/html":
            html_body = decoded
        elif mime_type == "text/plain":
            text_body = decoded
    if html_body:
        return html_body, "html"
    if text_body:
        return text_body, "text"
    return None


def _decode_body_content(payload: JSONObject) -> tuple[str, BodyType]:
    """Decode body content from Gmail message payload.

    Args:
        payload: Gmail message payload.

    Returns:
        Tuple of (body_content, body_type).
    """
    # Try simple body first
    simple_result = _decode_simple_body(payload)
    if simple_result is not None:
        return simple_result

    # Try multipart
    parts = payload.get("parts")
    if isinstance(parts, list):
        multipart_result = _decode_multipart_body(parts)
        if multipart_result is not None:
            return multipart_result

    return "", "text"


def _extract_labels(data: JSONObject) -> list[str]:
    """Extract labels from Gmail message data.

    Args:
        data: Gmail message data.

    Returns:
        List of label strings.
    """
    labels_raw = data.get("labelIds")
    labels: list[str] = []
    if isinstance(labels_raw, list):
        for label in labels_raw:
            if isinstance(label, str):
                labels.append(label)
    return labels


def _extract_headers(data: JSONObject) -> list[JSONValue]:
    """Extract headers from Gmail message data.

    Args:
        data: Gmail message data.

    Returns:
        List of header objects.
    """
    payload = data.get("payload")
    if isinstance(payload, dict):
        headers_raw = payload.get("headers")
        if isinstance(headers_raw, list):
            return headers_raw
    return []


def _parse_address_list(header_value: str) -> tuple[EmailAddress, ...]:
    """Parse a comma-separated list of email addresses.

    Args:
        header_value: Comma-separated email addresses.

    Returns:
        Tuple of EmailAddress objects.
    """
    if not header_value:
        return ()
    addrs: list[EmailAddress] = []
    for addr_str in header_value.split(","):
        addrs.append(_parse_email_address(addr_str.strip()))
    return tuple(addrs)


def _check_has_attachments(payload: JSONObject | None) -> bool:
    """Check if a Gmail message has attachments.

    Args:
        payload: Gmail message payload.

    Returns:
        True if message has attachments.
    """
    if not isinstance(payload, dict):
        return False
    parts = payload.get("parts")
    if not isinstance(parts, list):
        return False
    for part in parts:
        if isinstance(part, dict):
            filename = optional_str(part, "filename")
            if filename:
                return True
    return False


def _parse_importance(headers: list[JSONValue]) -> EmailImportance:
    """Parse importance from email headers.

    Args:
        headers: List of header objects.

    Returns:
        EmailImportance literal.
    """
    importance_header = _get_header_value(headers, "Importance")
    if importance_header.lower() == "high":
        return "high"
    if importance_header.lower() == "low":
        return "low"
    return "normal"


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

    def _decode_message(self, data: JSONObject, include_body: bool = True) -> Email:
        """Decode a message from Gmail API format.

        Args:
            data: JSON object representing a message.
            include_body: Whether to decode body content.

        Returns:
            Email.
        """
        message_id = require_str(data, "id")
        thread_id = optional_str(data, "threadId") or ""
        labels = _extract_labels(data)
        headers = _extract_headers(data)
        payload = data.get("payload")

        # Get body content
        body_content = ""
        body_type: BodyType = "text"
        if include_body and isinstance(payload, dict):
            body_content, body_type = _decode_body_content(payload)

        date_header = _get_header_value(headers, "Date")

        return Email(
            id=message_id,
            thread_id=thread_id,
            folder_id=labels[0] if labels else "",
            subject=_get_header_value(headers, "Subject"),
            body=body_content,
            body_type=body_type,
            from_address=_parse_email_address(_get_header_value(headers, "From")),
            to=_parse_address_list(_get_header_value(headers, "To")),
            cc=_parse_address_list(_get_header_value(headers, "Cc")),
            bcc=_parse_address_list(_get_header_value(headers, "Bcc")),
            sent_at=date_header,
            received_at=date_header,
            is_read="UNREAD" not in labels,
            is_draft="DRAFT" in labels,
            has_attachments=_check_has_attachments(payload if isinstance(payload, dict) else None),
            importance=_parse_importance(headers),
        )

    def _create_mime_message(
        self,
        *,
        to: tuple[str, ...],
        subject: str,
        body: str,
        body_type: BodyType,
        cc: tuple[str, ...],
        bcc: tuple[str, ...],
        attachments: tuple[Attachment, ...],
    ) -> str:
        """Create a MIME message and encode it.

        Args:
            to: Recipient addresses.
            subject: Email subject.
            body: Email body.
            body_type: Body content type.
            cc: CC addresses.
            bcc: BCC addresses.
            attachments: Attachments to include.

        Returns:
            Base64-encoded MIME message.
        """
        if attachments:
            msg: MIMEBase = MIMEMultipart()
            mime_type = "html" if body_type == "html" else "plain"
            msg.attach(MIMEText(body, mime_type))

            for att in attachments:
                if att["content_bytes"]:
                    content = base64.b64decode(att["content_bytes"])
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(content)
                    part.add_header("Content-Disposition", f'attachment; filename="{att["name"]}"')
                    msg.attach(part)
        else:
            mime_type = "html" if body_type == "html" else "plain"
            msg = MIMEText(body, mime_type)

        msg["To"] = ", ".join(to)
        msg["Subject"] = subject
        if cc:
            msg["Cc"] = ", ".join(cc)
        if bcc:
            msg["Bcc"] = ", ".join(bcc)

        return base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")

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
        raw_message = self._create_mime_message(
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
        raw_message = self._create_mime_message(
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
        raw_message = self._create_mime_message(
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

        return self._decode_message(data, include_body=False)

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
