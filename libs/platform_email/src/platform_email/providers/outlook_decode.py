"""Outlook (Graph) wire decoding."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    optional_str,
    require_str,
)
from platform_core.members import as_member

from platform_email.types import (
    BodyType,
    Email,
    EmailAddress,
    EmailImportance,
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


# Outlook's well-known folder display names, lower-cased, and the folder each one is.
_OUTLOOK_SYSTEM_FOLDERS: Final[Mapping[str, FolderType]] = {
    "inbox": FolderType.INBOX,
    "sent items": FolderType.SENT,
    "sent": FolderType.SENT,
    "drafts": FolderType.DRAFTS,
    "deleted items": FolderType.TRASH,
    "trash": FolderType.TRASH,
    "junk email": FolderType.SPAM,
    "spam": FolderType.SPAM,
    "junk": FolderType.SPAM,
    "archive": FolderType.ARCHIVE,
}


def _decode_folder_type(display_name: str) -> FolderType:
    """Map Outlook folder display name to FolderType.

    Args:
        display_name: Folder display name from Graph API.

    Returns:
        The system folder the name is, or CUSTOM for any folder a user made.
    """
    return _OUTLOOK_SYSTEM_FOLDERS.get(display_name.lower(), FolderType.CUSTOM)


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
    body_type = BodyType.TEXT
    if isinstance(body_raw, dict):
        body_content = optional_str(body_raw, "content") or ""
        content_type = optional_str(body_raw, "contentType")
        if content_type is not None:
            # Graph's bodyType is the enumeration text, html: another word is refused.
            body_type = as_member(content_type.lower(), "body.contentType", BodyType)

    # Graph's importance is the enumeration low, normal, high: another word is refused.
    importance_word = optional_str(data, "importance")
    final_importance = (
        EmailImportance.NORMAL
        if importance_word is None
        else as_member(importance_word.lower(), "importance", EmailImportance)
    )

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
