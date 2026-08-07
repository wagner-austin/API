"""Tests for world-message decoders.

``test_world.py`` was 707 lines; the viewport and terrain decoders are
now a sibling.
"""

from __future__ import annotations

import pytest

from tankpit_bot.protocol import (
    decode_active_forces,
    decode_build_pickup,
    decode_cache_update,
    decode_chat_message,
    decode_decoration,
    decode_overlay_update,
    decode_promotion,
    decode_statistics,
    decode_sync,
)
from tankpit_bot.wire.helpers import DecodeError


class TestDecodeSync:
    """Tests for decode_sync function."""

    def test_decodes_sync(self) -> None:
        """Decodes sync message (always succeeds)."""
        result = decode_sync(b"")
        assert result["msg_type"] == 0x3F

    def test_ignores_extra_data(self) -> None:
        """Ignores any extra data in sync message."""
        result = decode_sync(bytes([1, 2, 3, 4, 5]))
        assert result["msg_type"] == 0x3F


class TestDecodeCacheUpdate:
    """Tests for decode_cache_update function."""

    def test_decodes_cache_updates(self) -> None:
        """Decodes cache update entries."""
        data = bytes([10, 20, 0x04, 0x03, 30, 40, 0xFF, 0xFF])
        result = decode_cache_update(data)
        assert result["msg_type"] == 0x43
        assert result["updates"] == [(10, 20, 0x0304), (30, 40, -1)]

    def test_raises_on_invalid_length(self) -> None:
        """Raises DecodeError on invalid entry length."""
        with pytest.raises(DecodeError):
            decode_cache_update(bytes([1, 2, 3]))


class TestDecodeOverlayUpdate:
    """Tests for decode_overlay_update function."""

    def test_decodes_overlay_updates(self) -> None:
        """Decodes overlay update entries."""
        data = bytes([10, 20, 7, 30, 40, 255])
        result = decode_overlay_update(data)
        assert result["msg_type"] == 0x40
        assert result["updates"] == [(10, 20, 7), (30, 40, 255)]

    def test_raises_on_invalid_length(self) -> None:
        """Raises DecodeError on invalid entry length."""
        with pytest.raises(DecodeError):
            decode_overlay_update(bytes([1, 2]))


class TestDecodeChatMessage:
    """Tests for decode_chat_message function."""

    def test_decodes_chat_with_coords(self) -> None:
        """Decodes chat message with coordinates."""
        # sender_id=0x0102, message_type=1, x=50, y=60
        data = bytes([0x02, 0x01, 1, 50, 60])
        result = decode_chat_message(data)
        assert result["msg_type"] == 0x4D
        assert result["sender_id"] == 0x0102
        assert result["message_type"] == 1
        assert result["x"] == 50
        assert result["y"] == 60

    def test_decodes_chat_without_coords(self) -> None:
        """Decodes chat message without coordinates."""
        data = bytes([0x02, 0x01, 1])
        result = decode_chat_message(data)
        assert result["x"] is None
        assert result["y"] is None

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_chat_message(bytes([1, 2]))


class TestDecodeStatistics:
    """Tests for decode_statistics function."""

    def test_decodes_statistics_long_format(self) -> None:
        """Decodes long-format statistics (>12 bytes)."""
        data = (
            bytes([0x10, 0x00, 30, 45])  # hours=16, mins=30, secs=45
            + (100).to_bytes(4, "big")  # destroyed (32-bit BE)
            + bytes([50, 0x00])  # deactivated (LE u16 = 50)
            + (5000).to_bytes(4, "big")  # promo_points (32-bit BE)
        )
        result = decode_statistics(data)
        assert result["msg_type"] == 0x56
        assert result["playtime_hours"] == 16
        assert result["playtime_minutes"] == 30
        assert result["playtime_seconds"] == 45
        assert result["destroyed"] == 100
        assert result["deactivated"] == 50
        assert result["score"] == 5000

    def test_decodes_statistics_short_format(self) -> None:
        """Decodes short-format statistics (<=12 bytes)."""
        data = (
            bytes([0x10, 0x00, 30, 45])  # hours=16, mins=30, secs=45
            + bytes([100, 0x00])  # destroyed (LE u16 = 100)
            + bytes([50, 0x00])  # deactivated (LE u16 = 50)
            + (5000).to_bytes(4, "big")  # promo_points (32-bit BE)
        )
        result = decode_statistics(data)
        assert result["destroyed"] == 100
        assert result["deactivated"] == 50
        assert result["score"] == 5000

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_statistics(bytes([1, 2, 3, 4]))


class TestDecodeActiveForces:
    """Tests for decode_active_forces function."""

    def test_decodes_active_forces(self) -> None:
        """Decodes active forces message."""
        data = bytes([10, 15, 8, 12])  # Team counts
        result = decode_active_forces(data)
        assert result["msg_type"] == 0x2A
        assert result["team_counts"] == [10, 15, 8, 12]

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_active_forces(bytes([1, 2]))


class TestDecodeActivePlayers:
    """Tests for ``decode_active_players``."""

    def test_decodes_two_player_roster(self) -> None:
        """Each 3-byte record decodes to ``(tank_id_LE, rank)``."""
        from tankpit_bot.protocol.decoders.session_events import decode_active_players

        # Two records: (501, 5) + (1027, 2). 501 = 0x01F5 LE -> [0xF5, 0x01].
        data = bytes([0xF5, 0x01, 0x05, 0x03, 0x04, 0x02])
        result = decode_active_players(data)
        assert result["msg_type"] == 0x2F
        assert result["players"] == [
            {"tank_id": 501, "rank": 5},
            {"tank_id": 1027, "rank": 2},
        ]

    def test_rejects_non_multiple_of_three(self) -> None:
        """Body whose length isn't divisible by 3 is rejected."""
        from tankpit_bot.protocol.decoders.session_events import decode_active_players

        with pytest.raises(DecodeError):
            decode_active_players(bytes([0xF5, 0x01]))


class TestDecodeTop10:
    """Tests for ``decode_top10``."""

    def test_decodes_header_and_one_row(self) -> None:
        """Header parses + one row decodes to the expected entry."""
        from tankpit_bot.protocol.decoders.session_events import decode_top10

        # team_filter=255 (all), viewer_score=24bit BE = 0x010203 = 66051,
        # viewer_position=7, then ONE row:
        #   position=1, score=24bit BE = 0x102030 = 1056816, team=2,
        #   rank=8, name_len=4, name=b"Yupr"
        data = bytes(
            [
                0xFF,
                0x01,
                0x02,
                0x03,
                0x07,
                0x01,
                0x10,
                0x20,
                0x30,
                0x02,
                0x08,
                0x04,
                ord("Y"),
                ord("u"),
                ord("p"),
                ord("r"),
            ]
        )
        result = decode_top10(data)
        assert result["msg_type"] == 0x31
        assert result["team_filter"] == 255
        assert result["viewer_score"] == 0x010203
        assert result["viewer_position"] == 7
        assert len(result["entries"]) == 1
        row = result["entries"][0]
        assert row["position"] == 1
        assert row["score"] == 0x102030
        assert row["team"] == 2
        assert row["rank"] == 8
        assert row["name"] == "Yupr"
        assert row["tank_id"] == -1

    def test_rejects_short_header(self) -> None:
        """Bodies shorter than the 5-byte header are rejected."""
        from tankpit_bot.protocol.decoders.session_events import decode_top10

        with pytest.raises(DecodeError):
            decode_top10(bytes([0xFF, 0x00, 0x00, 0x00]))

    def test_rejects_truncated_row_header(self) -> None:
        """A row whose header runs past the body end is rejected."""
        from tankpit_bot.protocol.decoders.session_events import decode_top10

        # Header (5 bytes) + only 6 row bytes -- short of the 7-byte row header.
        data = bytes([0xFF, 0, 0, 0, 0]) + bytes([1, 0, 0, 0, 2, 8])
        with pytest.raises(DecodeError):
            decode_top10(data)

    def test_rejects_row_name_overflow(self) -> None:
        """A row whose name extends past the body end is rejected."""
        from tankpit_bot.protocol.decoders.session_events import decode_top10

        # Row header says name_len=10 but only 4 name bytes follow.
        header = bytes([0xFF, 0, 0, 0, 0])
        row = bytes([1, 0, 0, 0, 2, 8, 10]) + b"ABCD"
        with pytest.raises(DecodeError):
            decode_top10(header + row)


class TestDecodePingResponse:
    """Tests for the bare 0x60 PingResponse decoder."""

    def test_returns_bare_typed_message(self) -> None:
        """Body is discarded; only the msg_type tag survives."""
        from tankpit_bot.protocol.decoders.session_events import decode_ping_response

        result = decode_ping_response(b"")
        assert result == {"msg_type": 0x60}

    def test_ignores_unexpected_body_bytes(self) -> None:
        """Decoder is body-agnostic; non-empty bodies still decode."""
        from tankpit_bot.protocol.decoders.session_events import decode_ping_response

        result = decode_ping_response(bytes([0xDE, 0xAD]))
        assert result == {"msg_type": 0x60}


class TestDecodeConnectionLost:
    """Tests for the bare 0x7E ConnectionLost decoder."""

    def test_returns_bare_typed_message(self) -> None:
        """Body is discarded; only the msg_type tag survives."""
        from tankpit_bot.protocol.decoders.session_events import decode_connection_lost

        result = decode_connection_lost(b"")
        assert result == {"msg_type": 0x7E}


class TestDecodePromotion:
    """Tests for decode_promotion function.

    0x2B '+' Rf binary form: ``Rf(a[0], 1===a[1])``. a[0] is the new
    rank index; a[1]==1 sets the "You have been promoted!" banner.
    """

    def test_decodes_promotion_with_banner(self) -> None:
        """Banner-on promotion: rank=4, was_promoted=True."""
        data = bytes([4, 1])
        result = decode_promotion(data)
        assert result["msg_type"] == 0x2B
        assert result["new_rank"] == 4
        assert result["was_promoted"] is True

    def test_decodes_silent_rank_set(self) -> None:
        """Silent rank set (e.g. join-time): a[1]=0 -> was_promoted=False."""
        data = bytes([2, 0])
        result = decode_promotion(data)
        assert result["new_rank"] == 2
        assert result["was_promoted"] is False

    def test_banner_flag_only_true_on_exact_one(self) -> None:
        """JS uses ``1===a[1]``: byte 2 is False (not truthy)."""
        data = bytes([1, 2])
        result = decode_promotion(data)
        assert result["was_promoted"] is False

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data (require 2 bytes)."""
        with pytest.raises(DecodeError):
            decode_promotion(bytes([1]))


class TestDecodeDecoration:
    """Tests for decode_decoration function.

    0x4E 'N' Sf decoder: ``Sf(X(a[0],a[1]), a[2], a[3])`` -- tank_id,
    decoration slot, level. The renderer prints a banner only when
    ``level`` exceeds the tank's current level in that slot.
    """

    def test_decodes_decoration(self) -> None:
        """Decodes tank_id (LE), slot, level per JS Sf.h."""
        data = bytes([0x05, 0x01, 2, 3])
        result = decode_decoration(data)
        assert result["msg_type"] == 0x4E
        assert result["tank_id"] == 0x0105
        assert result["slot"] == 2
        assert result["level"] == 3

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data (require 4 bytes)."""
        with pytest.raises(DecodeError):
            decode_decoration(bytes([1, 2, 3]))


class TestDecodeBuildPickup:
    """Tests for decode_build_pickup function.

    0x42 'B' Jg decoder: ``Jg(X(a[0],a[1]), a[2], a[3], a[4], a[5],
    a[6], a[7], a[8])`` -- tank_id, source x/y, drop x/y, direction,
    obstacle_type, flag. JS ``Jg.prototype.h`` stamps the drop tile's
    ``j`` field with ``a[7]`` and only treats ``a[7] === 1`` as a
    bridge module; other non-zero values are obstacle subtypes.
    """

    def test_decodes_bridge_build(self) -> None:
        """obstacle_type=1 is the bridge variant."""
        data = bytes([0x10, 0x00, 5, 6, 7, 8, 12, 1, 0])
        result = decode_build_pickup(data)
        assert result["msg_type"] == 0x42
        assert result["tank_id"] == 0x0010
        assert result["source_x"] == 5
        assert result["source_y"] == 6
        assert result["drop_x"] == 7
        assert result["drop_y"] == 8
        assert result["direction"] == 12
        assert result["obstacle_type"] == 1
        assert result["flag"] == 0

    def test_decodes_obstacle_pickup(self) -> None:
        """Cleared obstacle (obstacle_type=0) with a non-zero flag."""
        data = bytes([0x05, 0x00, 1, 1, 2, 2, 4, 0, 3])
        result = decode_build_pickup(data)
        assert result["obstacle_type"] == 0
        assert result["flag"] == 3

    def test_decodes_real_obstacle_type_2(self) -> None:
        """Production captures (2026-06-19) show obstacle_type=2 for a
        regular obstacle drop -- byte 7 must be carried through as a
        plain int, not coerced to bool."""
        # Real capture: 2 of 2 production 0x42 samples have a[7]=0x02.
        # Sample sourced from runs/bot/*.capture_session.json via
        # analysis_scripts/crack_tank_update.py
        # inner_hex=1505c776c676770200 (tid=1301 src=(199,118)).
        data = bytes.fromhex("1505c776c676770200")
        result = decode_build_pickup(data)
        assert result["tank_id"] == 1301
        assert result["source_x"] == 199
        assert result["source_y"] == 118
        assert result["drop_x"] == 198
        assert result["drop_y"] == 118
        assert result["obstacle_type"] == 2

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data (require 9 bytes)."""
        with pytest.raises(DecodeError):
            decode_build_pickup(bytes([1, 2, 3, 4, 5, 6, 7, 8]))
