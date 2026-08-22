"""Gmail wire decoding and MIME construction."""

from __future__ import annotations

import base64
import email.utils
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    optional_str,
    require_str,
)
from platform_core.logging import get_logger

from platform_email.types import (
    Attachment,
    BodyType,
    Email,
    EmailAddress,
    EmailImportance,
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
    except (ValueError, UnicodeDecodeError) as exc:
        get_logger(__name__).debug("base64url decode failed: %s", exc)
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


def _decode_message(data: JSONObject, include_body: bool = True) -> Email:
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
