"""Tests for platform_email.types.folder module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_email.types.folder import (
    Folder,
    FolderType,
    decode_folder,
    encode_folder,
)

# =============================================================================
# Folder tests
# =============================================================================


class TestFolder:
    """Tests for Folder encode/decode functions."""

    def test_encode_folder(self) -> None:
        """Test encoding a Folder to JSON."""
        folder = Folder(
            id="folder-123",
            name="My Inbox",
            folder_type=FolderType.INBOX,
            unread_count=5,
            total_count=100,
        )
        result = encode_folder(folder)

        assert result["id"] == "folder-123"
        assert result["name"] == "My Inbox"
        assert result["folder_type"] == "inbox"
        assert result["unread_count"] == 5
        assert result["total_count"] == 100

    def test_decode_folder(self) -> None:
        """Test decoding a Folder from JSON."""
        data: JSONObject = {
            "id": "folder-456",
            "name": "Sent Items",
            "folder_type": "sent",
            "unread_count": 0,
            "total_count": 50,
        }
        result = decode_folder(data)

        assert result["id"] == "folder-456"
        assert result["name"] == "Sent Items"
        assert result["folder_type"] is FolderType.SENT
        assert result["unread_count"] == 0
        assert result["total_count"] == 50

    def test_decode_folder_reads_every_folder_type(self) -> None:
        """Every FolderType's wire word decodes to that member."""
        for folder_type in FolderType:
            data: JSONObject = {
                "id": "f",
                "name": "F",
                "folder_type": folder_type.value,
                "unread_count": 0,
                "total_count": 0,
            }
            assert decode_folder(data)["folder_type"] is folder_type

    def test_decode_folder_raises_for_missing_id(self) -> None:
        """Test that missing id raises JSONTypeError."""
        data: JSONObject = {
            "name": "Test",
            "folder_type": "custom",
            "unread_count": 0,
            "total_count": 0,
        }
        with pytest.raises(JSONTypeError):
            decode_folder(data)

    def test_decode_folder_raises_for_invalid_folder_type(self) -> None:
        """An unknown folder_type is refused naming the word and every admitted one."""
        data: JSONObject = {
            "id": "folder-err",
            "name": "Bad Folder",
            "folder_type": "deleted",
            "unread_count": 0,
            "total_count": 0,
        }
        with pytest.raises(
            JSONTypeError,
            match=r"^Invalid folder_type 'deleted': must be one of 'inbox', 'sent', 'drafts', "
            r"'trash', 'spam', 'archive', 'custom'$",
        ):
            decode_folder(data)

    def test_roundtrip(self) -> None:
        """Test encode then decode preserves data."""
        original = Folder(
            id="roundtrip-folder",
            name="Archive",
            folder_type=FolderType.ARCHIVE,
            unread_count=10,
            total_count=200,
        )
        encoded = encode_folder(original)
        decoded = decode_folder(encoded)
        assert decoded == original
