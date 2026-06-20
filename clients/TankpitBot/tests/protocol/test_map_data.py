"""Tests for decode_map_data (0x4C 'L' Ig handler).

Trace-verified against tpclient.js Ig.h. The MapData body has two
sections:
  1. Fuel-dot RLE: ``X(a[0],a[1])`` bytes that drive a (x,y) cursor
     from (1,1). Each byte advances ``x`` by its value (wrapping to
     ``y += 1; x %= 256``). ``255`` is continuation-only; every other
     value emits the current cursor as a dot.
  2. Tank entries: 5 bytes each. ``[x, y, tid_lo, tid_hi, packed]``
     where ``packed`` = ``rank<<4 | damage<<2 | team``.
"""

from __future__ import annotations

import pytest

from tankpit_bot.protocol import DecodeError, decode_map_data


def _pack(rank: int, damage: int, team: int) -> int:
    """Build the per-entry packed byte the way JS Ig.h reads it."""
    return ((rank & 0xF) << 4) | ((damage & 0x3) << 2) | (team & 0x3)


class TestDecodeMapDataFuelDots:
    """Fuel-dot RLE decoding tests."""

    def test_single_dot_emitted_at_start_offset(self) -> None:
        """A single byte ``h`` emits cursor ``(1+h, 1)``."""
        # rle_count=1, one cell value=5 -> dot at (6, 1)
        body = bytes([1, 0, 5])
        result = decode_map_data(body)
        assert result["fuel_dots"] == [(6, 1)]
        assert result["tanks"] == []

    def test_skip_byte_advances_without_emit(self) -> None:
        """``255`` advances the cursor but emits no dot."""
        # rle_count=2: cell 255 then cell 1.
        # 255: x = 1 + 255 = 256 -> wraps to y=2, x=0. No emit.
        # 1: x = 0 + 1 = 1. Emit (1, 2).
        body = bytes([2, 0, 255, 1])
        result = decode_map_data(body)
        assert result["fuel_dots"] == [(1, 2)]

    def test_wrap_emits_post_wrap_coords(self) -> None:
        """When ``x`` crosses 255 the emit uses post-wrap ``(x, y+1)``."""
        # rle_count=1: cell 254. x = 1 + 254 = 255 -> not > 255, emit (255, 1).
        body = bytes([1, 0, 254])
        assert decode_map_data(body)["fuel_dots"] == [(255, 1)]

    def test_explicit_wrap_when_x_exceeds_255(self) -> None:
        """``x > 255`` triggers wrap and the emit uses ``x %= 256``."""
        # cell 250 then cell 10:
        #   1+250 = 251 -> emit (251, 1)
        #   251+10 = 261 -> wrap -> y=2, x=5 -> emit (5, 2)
        body = bytes([2, 0, 250, 10])
        assert decode_map_data(body)["fuel_dots"] == [(251, 1), (5, 2)]


class TestDecodeMapDataTankEntries:
    """Tank-entry section tests."""

    def test_single_tank_entry(self) -> None:
        """Decodes the 5-byte packed entry per JS Ig.h."""
        # No RLE; one tank: x=10, y=20, tid=0x0102, rank=5, damage=1, team=2.
        body = bytes([0, 0, 10, 20, 0x02, 0x01, _pack(5, 1, 2)])
        result = decode_map_data(body)
        assert result["fuel_dots"] == []
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

    def test_combined_fuel_dots_and_tanks(self) -> None:
        """Fuel-dot section parses first; tank section starts at offset 2+count."""
        # rle_count=1, cell=3 -> dot at (4, 1); then one tank entry.
        body = bytes([1, 0, 3, 50, 60, 0x09, 0x00, _pack(4, 2, 3)])
        result = decode_map_data(body)
        assert result["fuel_dots"] == [(4, 1)]
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
