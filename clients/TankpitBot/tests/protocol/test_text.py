"""Tests for text message decoders.

Tests for decode_join_confirm, decode_world_info, and decode_text_message.
"""

from __future__ import annotations

import pytest

from tankpit_bot.protocol import (
    decode_join_confirm,
    decode_text_message,
    decode_world_info,
    try_decode_plaintext_ack,
)
from tankpit_bot.wire.helpers import DecodeError


class TestDecodeJoinConfirm:
    """Tests for decode_join_confirm function."""

    def test_decodes_valid_join_confirm(self) -> None:
        """Decodes valid join confirmation message.

        The four trailing counts are the room's ACTIVE FORCES per
        color (orange, purple, blue, red), confirmed live 2026-08-28:
        a world empty of humans reads 9,9,9,9 because it always
        carries 9 bots per color.
        """
        data = b"=2|2024-01-15|PlayerName|4|1|0|1|0"
        result = decode_join_confirm(data)
        assert result["msg_type"] == 0x3D
        assert result["team"] == 2
        assert result["game_start"] == "2024-01-15"
        assert result["name"] == "PlayerName"
        assert result["rank"] == 4
        assert result["active_forces"] == [1, 0, 1, 0]

    def test_decodes_with_missing_active_forces(self) -> None:
        """Decodes join confirmation with the trailing counts absent."""
        data = b"=1|2024-01-15|Tank|3"
        result = decode_join_confirm(data)
        assert result["active_forces"] == []

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


class TestTryDecodePlaintextAck:
    """Tests for try_decode_plaintext_ack function."""

    def test_decodes_autoscroll_ack(self) -> None:
        """Raw "A0"/"A1" decode to the autoscroll ack (key probe 2026-07-24)."""
        assert try_decode_plaintext_ack(b"A0") == {
            "msg_type": "autoscroll_ack",
            "enabled": False,
        }
        assert try_decode_plaintext_ack(b"A1") == {
            "msg_type": "autoscroll_ack",
            "enabled": True,
        }

    def test_decodes_chat_ack(self) -> None:
        """Raw "C0"/"C1" decode to the chat ack (key probe 2026-07-24)."""
        assert try_decode_plaintext_ack(b"C0") == {"msg_type": "chat_ack", "enabled": False}
        assert try_decode_plaintext_ack(b"C1") == {"msg_type": "chat_ack", "enabled": True}

    def test_returns_none_on_wrong_length(self) -> None:
        """Bodies that are not exactly two bytes are not acks."""
        assert try_decode_plaintext_ack(b"A") is None
        assert try_decode_plaintext_ack(b"A10") is None

    def test_returns_none_on_non_ascii_flag(self) -> None:
        """A two-byte 0x41 body with a binary flag byte is not an ack."""
        assert try_decode_plaintext_ack(bytes([0x41, 0x01])) is None

    def test_returns_none_on_other_letter(self) -> None:
        """Two-byte bodies of non-toggle letters are not acks."""
        assert try_decode_plaintext_ack(b"V1") is None
