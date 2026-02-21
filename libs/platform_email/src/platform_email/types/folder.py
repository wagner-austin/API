"""Folder-related TypedDict definitions.

Provides Folder and FolderType types with encode/decode functions.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_int,
    require_str,
)

# =============================================================================
# Literal Types
# =============================================================================

FolderType = Literal["inbox", "sent", "drafts", "trash", "spam", "archive", "custom"]


# =============================================================================
# Validation Helpers
# =============================================================================


def _require_folder_type(obj: JSONObject, key: str) -> FolderType:
    """Extract and validate FolderType from JSON object.

    Args:
        obj: JSON object to extract from.
        key: Key to extract.

    Returns:
        Validated FolderType literal.

    Raises:
        JSONTypeError: If value is not a valid FolderType.
    """
    value = require_str(obj, key)
    if value == "inbox":
        return "inbox"
    if value == "sent":
        return "sent"
    if value == "drafts":
        return "drafts"
    if value == "trash":
        return "trash"
    if value == "spam":
        return "spam"
    if value == "archive":
        return "archive"
    if value == "custom":
        return "custom"
    raise JSONTypeError(
        f"Field '{key}' must be inbox/sent/drafts/trash/spam/archive/custom, got '{value}'"
    )


# =============================================================================
# Folder
# =============================================================================


class Folder(TypedDict):
    """Email folder.

    Attributes:
        id: Unique folder identifier.
        name: Display name of the folder.
        folder_type: Type of folder (inbox, sent, etc.).
        unread_count: Number of unread emails in folder.
        total_count: Total number of emails in folder.
    """

    id: str
    name: str
    folder_type: FolderType
    unread_count: int
    total_count: int


def encode_folder(f: Folder) -> JSONObject:
    """Encode Folder to JSON-serializable dict.

    Args:
        f: Folder to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {
        "id": f["id"],
        "name": f["name"],
        "folder_type": f["folder_type"],
        "unread_count": f["unread_count"],
        "total_count": f["total_count"],
    }
    return result


def decode_folder(data: JSONObject) -> Folder:
    """Decode Folder from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated Folder.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return Folder(
        id=require_str(data, "id"),
        name=require_str(data, "name"),
        folder_type=_require_folder_type(data, "folder_type"),
        unread_count=require_int(data, "unread_count"),
        total_count=require_int(data, "total_count"),
    )


__all__ = [
    "Folder",
    "FolderType",
    "decode_folder",
    "encode_folder",
]
