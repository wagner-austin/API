"""Attachment-related TypedDict definitions.

Provides Attachment type with encode/decode functions.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    optional_str,
    require_int,
    require_str,
)

# =============================================================================
# Attachment
# =============================================================================


class Attachment(TypedDict):
    """Email attachment.

    Attributes:
        id: Unique attachment identifier.
        name: Filename of the attachment.
        content_type: MIME content type (e.g., "application/pdf").
        size: Size in bytes.
        content_bytes: Base64-encoded content, None if not fetched.
    """

    id: str
    name: str
    content_type: str
    size: int
    content_bytes: str | None


def encode_attachment(a: Attachment) -> JSONObject:
    """Encode Attachment to JSON-serializable dict.

    Args:
        a: Attachment to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {
        "id": a["id"],
        "name": a["name"],
        "content_type": a["content_type"],
        "size": a["size"],
        "content_bytes": a["content_bytes"],
    }
    return result


def decode_attachment(data: JSONObject) -> Attachment:
    """Decode Attachment from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated Attachment.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return Attachment(
        id=require_str(data, "id"),
        name=require_str(data, "name"),
        content_type=require_str(data, "content_type"),
        size=require_int(data, "size"),
        content_bytes=optional_str(data, "content_bytes"),
    )


__all__ = [
    "Attachment",
    "decode_attachment",
    "encode_attachment",
]
