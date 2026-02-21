"""Tests for platform_email.types.folder module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_email.types.folder import (
    Folder,
    _require_folder_type,
    decode_folder,
    encode_folder,
)

# =============================================================================
# _require_folder_type tests
# =============================================================================


class TestRequireFolderType:
    """Tests for _require_folder_type function."""

    def test_returns_inbox(self) -> None:
        """Test that 'inbox' value returns 'inbox' literal."""
        result = _require_folder_type({"type": "inbox"}, "type")
        assert result == "inbox"

    def test_returns_sent(self) -> None:
        """Test that 'sent' value returns 'sent' literal."""
        result = _require_folder_type({"type": "sent"}, "type")
        assert result == "sent"

    def test_returns_drafts(self) -> None:
        """Test that 'drafts' value returns 'drafts' literal."""
        result = _require_folder_type({"type": "drafts"}, "type")
        assert result == "drafts"

    def test_returns_trash(self) -> None:
        """Test that 'trash' value returns 'trash' literal."""
        result = _require_folder_type({"type": "trash"}, "type")
        assert result == "trash"

    def test_returns_spam(self) -> None:
        """Test that 'spam' value returns 'spam' literal."""
        result = _require_folder_type({"type": "spam"}, "type")
        assert result == "spam"

    def test_returns_archive(self) -> None:
        """Test that 'archive' value returns 'archive' literal."""
        result = _require_folder_type({"type": "archive"}, "type")
        assert result == "archive"

    def test_returns_custom(self) -> None:
        """Test that 'custom' value returns 'custom' literal."""
        result = _require_folder_type({"type": "custom"}, "type")
        assert result == "custom"

    def test_raises_for_invalid_value(self) -> None:
        """Test that invalid values raise JSONTypeError."""
        with pytest.raises(JSONTypeError) as exc_info:
            _require_folder_type({"type": "deleted"}, "type")
        assert "must be inbox/sent/drafts/trash/spam/archive/custom" in str(exc_info.value)
        assert "deleted" in str(exc_info.value)


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
            folder_type="inbox",
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
        assert result["folder_type"] == "sent"
        assert result["unread_count"] == 0
        assert result["total_count"] == 50

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
        """Test that invalid folder_type raises JSONTypeError."""
        data: JSONObject = {
            "id": "folder-err",
            "name": "Bad Folder",
            "folder_type": "unknown",
            "unread_count": 0,
            "total_count": 0,
        }
        with pytest.raises(JSONTypeError) as exc_info:
            decode_folder(data)
        assert "unknown" in str(exc_info.value)

    def test_roundtrip(self) -> None:
        """Test encode then decode preserves data."""
        original = Folder(
            id="roundtrip-folder",
            name="Archive",
            folder_type="archive",
            unread_count=10,
            total_count=200,
        )
        encoded = encode_folder(original)
        decoded = decode_folder(encoded)
        assert decoded == original

    def test_all_folder_types_roundtrip(self) -> None:
        """Test that all folder types can be roundtripped."""
        # Test each folder type individually
        folders = [
            Folder(id="f-inbox", name="I", folder_type="inbox", unread_count=0, total_count=0),
            Folder(id="f-sent", name="S", folder_type="sent", unread_count=0, total_count=0),
            Folder(id="f-drafts", name="D", folder_type="drafts", unread_count=0, total_count=0),
            Folder(id="f-trash", name="T", folder_type="trash", unread_count=0, total_count=0),
            Folder(id="f-spam", name="Sp", folder_type="spam", unread_count=0, total_count=0),
            Folder(id="f-archive", name="A", folder_type="archive", unread_count=0, total_count=0),
            Folder(id="f-custom", name="C", folder_type="custom", unread_count=0, total_count=0),
        ]
        for folder in folders:
            encoded = encode_folder(folder)
            decoded = decode_folder(encoded)
            assert decoded["folder_type"] == folder["folder_type"]
