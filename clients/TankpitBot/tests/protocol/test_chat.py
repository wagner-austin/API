"""Tests for the chat protocol module (preset table + outbound command).

Wire truth: sniff-20260729-214411 — 44 decoded chat sends, all in the
6-byte Hb form ``[6,'m',id,x,y,0]``; 8 inbound 0x4D echoes.
"""

from __future__ import annotations

from tankpit_bot.protocol.chat import (
    CHAT_HELLO,
    CHAT_MESSAGES,
    CMD_CHAT,
    TYPE_CHAT,
    build_chat_command,
    chat_message_text,
)


class TestChatTable:
    """The preset message table mirrors the JS E[] object."""

    def test_exactly_65_messages(self) -> None:
        """The game defines E[0] through E[64] — nothing more or less."""
        assert len(CHAT_MESSAGES) == 65
        assert sorted(CHAT_MESSAGES) == list(range(65))

    def test_hello_is_id_41(self) -> None:
        """The greeting the bot sends on a human lock."""
        assert CHAT_HELLO == 41
        assert CHAT_MESSAGES[CHAT_HELLO] == "HELLO"

    def test_known_texts_pin_the_table_corners(self) -> None:
        """First, last, and the wire-observed echo IDs resolve correctly."""
        assert CHAT_MESSAGES[0] == "Attack the red"
        assert CHAT_MESSAGES[64] == "I'm playing TankPit, dad."
        # The 8 IDs the server echoed in sniff-20260729-214411.
        assert CHAT_MESSAGES[12] == "Base is here"
        assert CHAT_MESSAGES[18] == "Buzz off!"
        assert CHAT_MESSAGES[42] == "BYE"

    def test_chat_message_text_known(self) -> None:
        """Lookup returns the preset text for a known ID."""
        assert chat_message_text(41) == "HELLO"
        assert chat_message_text(49) == "My dog plays better than you!"

    def test_chat_message_text_unknown(self) -> None:
        """IDs outside the E[] table get an explicit unknown marker."""
        assert chat_message_text(65) == "unknown_65"
        assert chat_message_text(-1) == "unknown_-1"


class TestBuildChatCommand:
    """Outbound frame layout for the plaintext send path."""

    def test_command_constants(self) -> None:
        """0x6D 'm' with the 6-byte Hb frame's type byte."""
        assert CMD_CHAT == 0x6D
        assert TYPE_CHAT == 6

    def test_hello_frame_bytes(self) -> None:
        """The exact plaintext image of the live HELLO send.

        sniff-20260729-214411 at t+125.9s: XOR-decoded send was
        ``06 6d 29 8d ec 00`` — HELLO (41) from tile (141, 236).
        The bot's plaintext frame is the same bytes behind the
        ``!`` prefix and 2-byte LE length header.
        """
        frame = build_chat_command(41, 141, 236)
        assert frame == bytes([7, 0, 0x21, 0x06, 0x6D, 41, 141, 236, 0])

    def test_coordinates_masked_to_byte_range(self) -> None:
        """Out-of-range inputs clamp into the single-byte wire fields."""
        frame = build_chat_command(300, 256, -1)
        assert frame[5] == 300 & 0xFF
        assert frame[6] == 0
        assert frame[7] == 0xFF

    def test_flag_byte_always_zero(self) -> None:
        """All 44 observed live sends carried flag=0 — so does the bot."""
        assert build_chat_command(12, 97, 212)[-1] == 0
