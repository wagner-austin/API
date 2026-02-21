"""Draft-related TypedDict definitions.

Provides Draft type with encode/decode functions.
Drafts are emails in an unsent state.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    require_str,
)

from platform_email.types.email import (
    BodyType,
    EmailAddress,
    _require_body_type,
    _require_email_address_tuple,
    encode_email_address,
)

# =============================================================================
# Draft
# =============================================================================


class Draft(TypedDict):
    """Email draft.

    Attributes:
        id: Unique draft identifier.
        subject: Draft subject line.
        body: Draft body content.
        body_type: Body content type (text or html).
        to: Recipient email addresses.
        cc: CC recipient email addresses.
        bcc: BCC recipient email addresses.
    """

    id: str
    subject: str
    body: str
    body_type: BodyType
    to: tuple[EmailAddress, ...]
    cc: tuple[EmailAddress, ...]
    bcc: tuple[EmailAddress, ...]


def encode_draft(d: Draft) -> JSONObject:
    """Encode Draft to JSON-serializable dict.

    Args:
        d: Draft to encode.

    Returns:
        JSON-serializable dict representation.
    """
    to_list: list[JSONValue] = []
    for addr in d["to"]:
        to_list.append(encode_email_address(addr))
    cc_list: list[JSONValue] = []
    for addr in d["cc"]:
        cc_list.append(encode_email_address(addr))
    bcc_list: list[JSONValue] = []
    for addr in d["bcc"]:
        bcc_list.append(encode_email_address(addr))
    result: JSONObject = {
        "id": d["id"],
        "subject": d["subject"],
        "body": d["body"],
        "body_type": d["body_type"],
        "to": to_list,
        "cc": cc_list,
        "bcc": bcc_list,
    }
    return result


def decode_draft(data: JSONObject) -> Draft:
    """Decode Draft from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated Draft.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return Draft(
        id=require_str(data, "id"),
        subject=require_str(data, "subject"),
        body=require_str(data, "body"),
        body_type=_require_body_type(data, "body_type"),
        to=_require_email_address_tuple(data, "to"),
        cc=_require_email_address_tuple(data, "cc"),
        bcc=_require_email_address_tuple(data, "bcc"),
    )


__all__ = [
    "Draft",
    "decode_draft",
    "encode_draft",
]
