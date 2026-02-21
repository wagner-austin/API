"""Tests for platform_email.types.attachment module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_email.types.attachment import (
    Attachment,
    decode_attachment,
    encode_attachment,
)


class TestAttachment:
    """Tests for Attachment encode/decode functions."""

    def test_encode_attachment_with_content(self) -> None:
        """Test encoding an Attachment with content_bytes."""
        attachment = Attachment(
            id="att-123",
            name="document.pdf",
            content_type="application/pdf",
            size=1024,
            content_bytes="base64encodedcontent==",
        )
        result = encode_attachment(attachment)

        assert result["id"] == "att-123"
        assert result["name"] == "document.pdf"
        assert result["content_type"] == "application/pdf"
        assert result["size"] == 1024
        assert result["content_bytes"] == "base64encodedcontent=="

    def test_encode_attachment_without_content(self) -> None:
        """Test encoding an Attachment with None content_bytes."""
        attachment = Attachment(
            id="att-456",
            name="image.png",
            content_type="image/png",
            size=2048,
            content_bytes=None,
        )
        result = encode_attachment(attachment)

        assert result["content_bytes"] is None

    def test_decode_attachment_with_content(self) -> None:
        """Test decoding an Attachment with content_bytes."""
        data: JSONObject = {
            "id": "att-789",
            "name": "spreadsheet.xlsx",
            "content_type": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            "size": 4096,
            "content_bytes": "xlsxcontent==",
        }
        result = decode_attachment(data)

        assert result["id"] == "att-789"
        assert result["name"] == "spreadsheet.xlsx"
        assert result["size"] == 4096
        assert result["content_bytes"] == "xlsxcontent=="

    def test_decode_attachment_without_content(self) -> None:
        """Test decoding an Attachment without content_bytes."""
        data: JSONObject = {
            "id": "att-no-content",
            "name": "video.mp4",
            "content_type": "video/mp4",
            "size": 1048576,
        }
        result = decode_attachment(data)

        assert result["content_bytes"] is None

    def test_decode_attachment_raises_for_missing_id(self) -> None:
        """Test that missing id raises JSONTypeError."""
        data: JSONObject = {
            "name": "test.txt",
            "content_type": "text/plain",
            "size": 100,
        }
        with pytest.raises(JSONTypeError):
            decode_attachment(data)

    def test_decode_attachment_raises_for_missing_name(self) -> None:
        """Test that missing name raises JSONTypeError."""
        data: JSONObject = {
            "id": "att-err",
            "content_type": "text/plain",
            "size": 100,
        }
        with pytest.raises(JSONTypeError):
            decode_attachment(data)

    def test_decode_attachment_raises_for_missing_size(self) -> None:
        """Test that missing size raises JSONTypeError."""
        data: JSONObject = {
            "id": "att-err",
            "name": "test.txt",
            "content_type": "text/plain",
        }
        with pytest.raises(JSONTypeError):
            decode_attachment(data)

    def test_roundtrip_with_content(self) -> None:
        """Test encode then decode preserves data with content."""
        original = Attachment(
            id="roundtrip-1",
            name="file.zip",
            content_type="application/zip",
            size=512,
            content_bytes="zipcontent==",
        )
        encoded = encode_attachment(original)
        decoded = decode_attachment(encoded)
        assert decoded == original

    def test_roundtrip_without_content(self) -> None:
        """Test encode then decode preserves data without content."""
        original = Attachment(
            id="roundtrip-2",
            name="large.bin",
            content_type="application/octet-stream",
            size=10000000,
            content_bytes=None,
        )
        encoded = encode_attachment(original)
        decoded = decode_attachment(encoded)
        assert decoded == original
