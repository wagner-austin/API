"""Email-related TypedDict definitions.

Provides Email, EmailAddress, and EmailListResult types with encode/decode functions.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    optional_str,
    require_bool,
    require_list,
    require_str,
)

# =============================================================================
# Literal Types
# =============================================================================

BodyType = Literal["text", "html"]
EmailImportance = Literal["low", "normal", "high"]


# =============================================================================
# Validation Helpers
# =============================================================================


def _require_body_type(obj: JSONObject, key: str) -> BodyType:
    """Extract and validate BodyType from JSON object.

    Args:
        obj: JSON object to extract from.
        key: Key to extract.

    Returns:
        Validated BodyType literal.

    Raises:
        JSONTypeError: If value is not a valid BodyType.
    """
    value = require_str(obj, key)
    if value == "text":
        return "text"
    if value == "html":
        return "html"
    raise JSONTypeError(f"Field '{key}' must be text/html, got '{value}'")


def _require_email_importance(obj: JSONObject, key: str) -> EmailImportance:
    """Extract and validate EmailImportance from JSON object.

    Args:
        obj: JSON object to extract from.
        key: Key to extract.

    Returns:
        Validated EmailImportance literal.

    Raises:
        JSONTypeError: If value is not a valid EmailImportance.
    """
    value = require_str(obj, key)
    if value == "low":
        return "low"
    if value == "normal":
        return "normal"
    if value == "high":
        return "high"
    raise JSONTypeError(f"Field '{key}' must be low/normal/high, got '{value}'")


def _require_dict_value(value: JSONValue, context: str) -> JSONObject:
    """Require value to be a dict.

    Args:
        value: JSON value to check.
        context: Context string for error message.

    Returns:
        Value as JSONObject.

    Raises:
        JSONTypeError: If value is not a dict.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"{context} must be an object, got {type(value).__name__}")
    return value


# =============================================================================
# EmailAddress
# =============================================================================


class EmailAddress(TypedDict):
    """Email address with display name.

    Attributes:
        address: Email address (e.g., "user@example.com").
        name: Display name (e.g., "John Doe").
    """

    address: str
    name: str


def encode_email_address(addr: EmailAddress) -> JSONObject:
    """Encode EmailAddress to JSON-serializable dict.

    Args:
        addr: EmailAddress to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {
        "address": addr["address"],
        "name": addr["name"],
    }
    return result


def decode_email_address(data: JSONObject) -> EmailAddress:
    """Decode EmailAddress from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated EmailAddress.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return EmailAddress(
        address=require_str(data, "address"),
        name=require_str(data, "name"),
    )


def _require_email_address_tuple(obj: JSONObject, key: str) -> tuple[EmailAddress, ...]:
    """Extract and validate a tuple of EmailAddress from JSON object.

    Args:
        obj: JSON object to extract from.
        key: Key to extract.

    Returns:
        Tuple of validated EmailAddress objects.

    Raises:
        JSONTypeError: If value is not a list of email address objects.
    """
    raw_list = require_list(obj, key)
    result: list[EmailAddress] = []
    for i, item in enumerate(raw_list):
        item_dict = _require_dict_value(item, f"{key}[{i}]")
        result.append(decode_email_address(item_dict))
    return tuple(result)


# =============================================================================
# Attachment (forward reference for Email)
# =============================================================================

# Note: Attachment is defined in attachment.py and imported where needed
# to avoid circular imports. Email uses tuple[Attachment, ...] but we
# encode/decode them separately.


# =============================================================================
# Email
# =============================================================================


class Email(TypedDict):
    """Email message.

    Attributes:
        id: Unique email identifier.
        thread_id: Thread/conversation identifier.
        folder_id: Folder containing this email.
        subject: Email subject line.
        body: Email body content.
        body_type: Body content type (text or html).
        from_address: Sender email address.
        to: Recipient email addresses.
        cc: CC recipient email addresses.
        bcc: BCC recipient email addresses.
        sent_at: ISO 8601 datetime when sent.
        received_at: ISO 8601 datetime when received.
        is_read: Whether email has been read.
        is_draft: Whether this is a draft.
        has_attachments: Whether email has attachments.
        importance: Email importance level.
    """

    id: str
    thread_id: str
    folder_id: str
    subject: str
    body: str
    body_type: BodyType
    from_address: EmailAddress
    to: tuple[EmailAddress, ...]
    cc: tuple[EmailAddress, ...]
    bcc: tuple[EmailAddress, ...]
    sent_at: str
    received_at: str
    is_read: bool
    is_draft: bool
    has_attachments: bool
    importance: EmailImportance


def encode_email(e: Email) -> JSONObject:
    """Encode Email to JSON-serializable dict.

    Args:
        e: Email to encode.

    Returns:
        JSON-serializable dict representation.
    """
    to_list: list[JSONValue] = []
    for addr in e["to"]:
        to_list.append(encode_email_address(addr))
    cc_list: list[JSONValue] = []
    for addr in e["cc"]:
        cc_list.append(encode_email_address(addr))
    bcc_list: list[JSONValue] = []
    for addr in e["bcc"]:
        bcc_list.append(encode_email_address(addr))
    result: JSONObject = {
        "id": e["id"],
        "thread_id": e["thread_id"],
        "folder_id": e["folder_id"],
        "subject": e["subject"],
        "body": e["body"],
        "body_type": e["body_type"],
        "from_address": encode_email_address(e["from_address"]),
        "to": to_list,
        "cc": cc_list,
        "bcc": bcc_list,
        "sent_at": e["sent_at"],
        "received_at": e["received_at"],
        "is_read": e["is_read"],
        "is_draft": e["is_draft"],
        "has_attachments": e["has_attachments"],
        "importance": e["importance"],
    }
    return result


def decode_email(data: JSONObject) -> Email:
    """Decode Email from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated Email.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    from_raw = data.get("from_address")
    from_dict = _require_dict_value(from_raw, "from_address")
    return Email(
        id=require_str(data, "id"),
        thread_id=require_str(data, "thread_id"),
        folder_id=require_str(data, "folder_id"),
        subject=require_str(data, "subject"),
        body=require_str(data, "body"),
        body_type=_require_body_type(data, "body_type"),
        from_address=decode_email_address(from_dict),
        to=_require_email_address_tuple(data, "to"),
        cc=_require_email_address_tuple(data, "cc"),
        bcc=_require_email_address_tuple(data, "bcc"),
        sent_at=require_str(data, "sent_at"),
        received_at=require_str(data, "received_at"),
        is_read=require_bool(data, "is_read"),
        is_draft=require_bool(data, "is_draft"),
        has_attachments=require_bool(data, "has_attachments"),
        importance=_require_email_importance(data, "importance"),
    )


# =============================================================================
# EmailListResult
# =============================================================================


class EmailListResult(TypedDict):
    """Result of listing emails with pagination.

    Attributes:
        emails: Tuple of emails in this page.
        next_page_token: Token for fetching next page, None if no more pages.
    """

    emails: tuple[Email, ...]
    next_page_token: str | None


def encode_email_list_result(r: EmailListResult) -> JSONObject:
    """Encode EmailListResult to JSON-serializable dict.

    Args:
        r: EmailListResult to encode.

    Returns:
        JSON-serializable dict representation.
    """
    emails_list: list[JSONValue] = []
    for email in r["emails"]:
        emails_list.append(encode_email(email))
    result: JSONObject = {
        "emails": emails_list,
        "next_page_token": r["next_page_token"],
    }
    return result


def decode_email_list_result(data: JSONObject) -> EmailListResult:
    """Decode EmailListResult from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated EmailListResult.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    emails_raw = require_list(data, "emails")
    emails: list[Email] = []
    for i, item in enumerate(emails_raw):
        item_dict = _require_dict_value(item, f"emails[{i}]")
        emails.append(decode_email(item_dict))
    return EmailListResult(
        emails=tuple(emails),
        next_page_token=optional_str(data, "next_page_token"),
    )


__all__ = [
    "BodyType",
    "Email",
    "EmailAddress",
    "EmailImportance",
    "EmailListResult",
    "decode_email",
    "decode_email_address",
    "decode_email_list_result",
    "encode_email",
    "encode_email_address",
    "encode_email_list_result",
]
