"""Tests for decode_map_data (0x4C 'L' Ig handler).

Trace-verified against tpclient.js Ig.h. The MapData body has two
sections:
  1. Fuel-dot RLE: ``X(a[0],a[1])`` bytes that the decoder validates
     for length and then skips. The bot stopped consulting the
     fuel-dot atlas 2026-06-22, so the coordinates inside the RLE
     region are not surfaced on the result anymore.
  2. Tank entries: 5 bytes each. ``[x, y, tid_lo, tid_hi, packed]``
     where ``packed`` = ``rank<<4 | damage<<2 | team``.
"""

from __future__ import annotations

import pytest

from tankpit_bot.protocol import DecodeError, decode_map_data


def _pack(rank: int, damage: int, team: int) -> int:
    """Build the per-entry packed byte the way JS Ig.h reads it."""
    return ((rank & 0xF) << 4) | ((damage & 0x3) << 2) | (team & 0x3)


class TestDecodeMapDataTankEntries:
    """Tank-entry section tests."""

    def test_single_tank_entry(self) -> None:
        """Decodes the 5-byte packed entry per JS Ig.h."""
        # No RLE; one tank: x=10, y=20, tid=0x0102, rank=5, damage=1, team=2.
        body = bytes([0, 0, 10, 20, 0x02, 0x01, _pack(5, 1, 2)])
        result = decode_map_data(body)
        assert len(result["tanks"]) == 1
        tank = result["tanks"][0]
        assert tank["x"] == 10
        assert tank["y"] == 20
        assert tank["tank_id"] == 0x0102
        assert tank["rank"] == 5
        assert tank["damage"] == 1
        assert tank["team"] == 2

    def test_multiple_tank_entries(self) -> None:
        """Multiple tanks decoded in wire order."""
        body = bytes(
            [
                0,
                0,
                # tank 1
                1,
                2,
                0x05,
                0x00,
                _pack(0, 0, 0),
                # tank 2
                3,
                4,
                0x07,
                0x00,
                _pack(7, 3, 1),
            ]
        )
        result = decode_map_data(body)
        assert len(result["tanks"]) == 2
        assert result["tanks"][0]["tank_id"] == 5
        assert result["tanks"][1]["tank_id"] == 7
        assert result["tanks"][1]["rank"] == 7

    def test_rle_region_is_skipped_before_tank_section(self) -> None:
        """The decoder advances past the RLE region and finds the tank section.

        Pre-2026-06-22 the RLE region was decoded into ``fuel_dots``
        on the result; now the decoder uses the byte count only to
        skip past those bytes. Feeding a non-empty RLE region must
        not corrupt the tank section that follows it.
        """
        # rle_count=1, one RLE cell (ignored), then one tank entry.
        body = bytes([1, 0, 3, 50, 60, 0x09, 0x00, _pack(4, 2, 3)])
        result = decode_map_data(body)
        assert len(result["tanks"]) == 1
        assert result["tanks"][0]["tank_id"] == 9
        assert result["tanks"][0]["team"] == 3
        assert result["tanks"][0]["damage"] == 2


class TestDecodeMapDataErrors:
    """Failure-mode tests."""

    def test_raises_on_header_truncated(self) -> None:
        """Header is u16 LE; <2 bytes is a hard error."""
        with pytest.raises(DecodeError):
            decode_map_data(bytes([0]))

    def test_raises_when_rle_overflows_body(self) -> None:
        """``rle_count`` larger than body raises rather than truncates."""
        # Claim 100 RLE bytes but supply none.
        with pytest.raises(DecodeError):
            decode_map_data(bytes([100, 0]))

    def test_raises_when_tank_tail_not_multiple_of_five(self) -> None:
        """Tank section must be a clean multiple of 5; never rounds down."""
        # 0 RLE bytes + 4 trailing bytes (one byte short of an entry).
        body = bytes([0, 0, 10, 20, 0x02, 0x01])
        with pytest.raises(DecodeError, match="not a multiple"):
            decode_map_data(body)


class TestDecodeMapDataIdentity:
    """The msg_type discriminant lands at 0x4C."""

    def test_msg_type_is_0x4c(self) -> None:
        result = decode_map_data(bytes([0, 0]))
        assert result["msg_type"] == 0x4C
