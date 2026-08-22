"""Outlook (Graph) wire decoding."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    optional_str,
    require_str,
)

from platform_email.types import (
    BodyType,
    Email,
    EmailAddress,
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


def _decode_message(data: JSONObject) -> Email:
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
