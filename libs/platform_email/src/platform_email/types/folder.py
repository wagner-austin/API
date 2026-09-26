"""Folder-related TypedDict definitions.

Provides Folder and FolderType types with encode/decode functions.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    require_int,
    require_str,
)
from platform_core.members import require_member

# =============================================================================
# Vocabularies
# =============================================================================


class FolderType(StrEnum):
    """Which system folder a provider's folder or label is; CUSTOM for a user-made one."""

    INBOX = "inbox"
    SENT = "sent"
    DRAFTS = "drafts"
    TRASH = "trash"
    SPAM = "spam"
    ARCHIVE = "archive"
    CUSTOM = "custom"


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
        folder_type=require_member(data, "folder_type", FolderType),
        unread_count=require_int(data, "unread_count"),
        total_count=require_int(data, "total_count"),
    )


__all__ = [
    "Folder",
    "FolderType",
    "decode_folder",
    "encode_folder",
]
