"""Tests for tankpit_bot.sniffer XOR encoding/decoding."""

from __future__ import annotations

from tankpit_bot.sniffer.xor import reset_xor_state, xor_decode


class TestXorDecode:
    """Tests for xor_decode function."""

    def test_decode_with_no_global_table(self) -> None:
        """Test xor_decode returns body[1:] when no global table."""
        # Reset to no table
        reset_xor_state()

        result = xor_decode(b"\x2e\x01\x02\x03")
        assert result == b"\x01\x02\x03"

    def test_decode_short_body(self) -> None:
        """Test xor_decode handles short body."""
        # Reset to no table
        reset_xor_state()

        result = xor_decode(b"\x2e")
        assert result == b""
