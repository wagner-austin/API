"""Tests for text message decoders.

Tests for decode_join_confirm, decode_world_info, and decode_text_message.
"""

from __future__ import annotations

import pytest

from tankpit_bot.protocol import (
    DecodeError,
    decode_join_confirm,
    decode_text_message,
    decode_world_info,
)


class TestDecodeJoinConfirm:
    """Tests for decode_join_confirm function."""

    def test_decodes_valid_join_confirm(self) -> None:
        """Decodes valid join confirmation message."""
        data = b"=2|2024-01-15|PlayerName|4|1|0|1|0"
        result = decode_join_confirm(data)
        assert result["msg_type"] == 0x3D
        assert result["team"] == 2
        assert result["join_date"] == "2024-01-15"
        assert result["name"] == "PlayerName"
        assert result["rank"] == 4
        assert result["equipment"] == [1, 0, 1, 0]

    def test_decodes_with_missing_equipment(self) -> None:
        """Decodes join confirmation with missing equipment fields."""
        data = b"=1|2024-01-15|Tank|3"
        result = decode_join_confirm(data)
        assert result["equipment"] == []

    def test_raises_on_wrong_prefix(self) -> None:
        """Raises DecodeError when prefix is wrong."""
        with pytest.raises(DecodeError):
            decode_join_confirm(b"+wrong|prefix")

    def test_raises_on_too_few_parts(self) -> None:
        """Raises DecodeError when too few parts."""
        with pytest.raises(DecodeError):
            decode_join_confirm(b"=1|2|3")


class TestDecodeWorldInfo:
    """Tests for decode_world_info function."""

    def test_decodes_valid_world_info(self) -> None:
        """Decodes valid world info message."""
        data = b"+123|WorldName|456|1,2,3|2|mode|image.png|2024"
        result = decode_world_info(data)
        assert result["msg_type"] == 0x2B
        assert result["world_id"] == 123
        assert result["name"] == "WorldName"
        assert result["field_id"] == 456
        assert result["flags"] == [1, 2, 3]
        assert result["team"] == 2
        assert result["mode"] == "mode"
        assert result["image"] == "image.png"
        assert result["year"] == 2024

    def test_handles_non_numeric_year(self) -> None:
        """Handles non-numeric year field."""
        data = b"+1|Name|2|1|0|mode|img|invalid"
        result = decode_world_info(data)
        assert result["year"] == 0

    def test_raises_on_wrong_prefix(self) -> None:
        """Raises DecodeError when prefix is wrong."""
        with pytest.raises(DecodeError):
            decode_world_info(b"=wrong|prefix")

    def test_raises_on_too_few_parts(self) -> None:
        """Raises DecodeError when too few parts."""
        with pytest.raises(DecodeError):
            decode_world_info(b"+1|2|3|4|5")


class TestDecodeTextMessage:
    """Tests for decode_text_message function."""

    def test_dispatches_join_confirm(self) -> None:
        """Dispatches to join confirm decoder."""
        data = b"=2|2024-01-15|Player|4|1|0|1|0"
        result = decode_text_message(data)
        assert result["msg_type"] == 0x3D

    def test_dispatches_world_info(self) -> None:
        """Dispatches to world info decoder."""
        data = b"+123|Name|456|1|2|mode|img|2024"
        result = decode_text_message(data)
        assert result["msg_type"] == 0x2B

    def test_raises_on_empty_body(self) -> None:
        """Raises DecodeError on empty body."""
        with pytest.raises(DecodeError) as exc:
            decode_text_message(b"")
        assert "empty body" in str(exc.value)

    def test_raises_on_unknown_type(self) -> None:
        """Raises DecodeError on unknown message type."""
        with pytest.raises(DecodeError) as exc:
            decode_text_message(b"X unknown")
        assert "unknown type" in str(exc.value)
