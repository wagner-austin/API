"""Tests for combat message decoders.

Tests for shoot, hit confirmation, deactivation, and mine decoders.
"""

from __future__ import annotations

import pytest

from tankpit_bot.protocol import (
    decode_deactivation,
    decode_shoot_event,
)
from tankpit_bot.wire.helpers import DecodeError


class TestDecodeShootEvent:
    """Tests for decode_shoot_event function.

    Real wire bytes from runs/bot/bot-20260619-050303 capture, validated
    three ways: enemy src tracking, homing target tile, wire damage
    transitions. Field layout per tpclient.js Gg.h (V.S):
      [team][shooter_id:2 LE][src_x][src_y][tgt_x][tgt_y][aim_x][aim_y][weapon]
    """

    def test_decodes_own_dual_shot(self) -> None:
        """Own dual shot at orange-8 -- real bytes from t+35.48s.

        Straight shot: aim == target (both at (155,155)).
        """
        # Body after 0x53 opcode stripped:
        # 02 15 05 9b 9a 9b 9b 9b 9b 01
        data = bytes.fromhex("0215059b9a9b9b9b9b01")
        result = decode_shoot_event(data)
        assert result["msg_type"] == 0x53
        assert result["team"] == 2  # blue
        assert result["shooter_id"] == 1301  # Artax
        assert result["source_x"] == 155
        assert result["source_y"] == 154
        assert result["target_x"] == 155
        assert result["target_y"] == 155
        assert result["aim_x"] == 155
        assert result["aim_y"] == 155
        assert result["weapon"] == 1  # dual

    def test_decodes_enemy_single_shot(self) -> None:
        """Enemy single shot at us -- orange-8 firing back from (155,155).

        Straight shot: aim == target (both at (155,154)).
        """
        data = bytes.fromhex("0316029b9b9b9a9b9a00")
        result = decode_shoot_event(data)
        assert result["msg_type"] == 0x53
        assert result["team"] == 3  # orange
        assert result["shooter_id"] == 534  # orange-8
        assert result["source_x"] == 155  # orange-8's tile
        assert result["source_y"] == 155
        assert result["target_x"] == 155  # our tile
        assert result["target_y"] == 154
        assert result["aim_x"] == 155
        assert result["aim_y"] == 154
        assert result["weapon"] == 0  # single

    def test_decodes_homing_shot_landing_off_command(self) -> None:
        """Homing seeker landed at (170,174) when bot fired toward (155,155).

        Homing weapon: aim is the initial barrel direction (where the
        bot pointed when firing); target is the homing impact tile.
        """
        data = bytes.fromhex("0215059b9aaaaeaaae03")
        result = decode_shoot_event(data)
        assert result["source_x"] == 155
        assert result["source_y"] == 154
        assert result["target_x"] == 170  # homing seeker's actual impact
        assert result["target_y"] == 174
        assert result["aim_x"] == 170  # initial barrel aim (same here)
        assert result["aim_y"] == 174
        assert result["weapon"] == 3  # homing

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_shoot_event(bytes([1, 2, 3, 4, 5]))


# decode_hit_confirmation was deleted 2026-06-19: it was a stranded
# alternate decoder for 0x48 that never matched any JS handler. The
# canonical 0x48 decoder is decode_enemy_detection (Tg.h / V.H, x/y/
# team/rank/tank_id), wired via MSG_ENEMY_DETECT in routing.py.


class TestDecodeDeactivation:
    """Tests for decode_deactivation function."""

    def test_decodes_deactivation(self) -> None:
        """Decodes deactivation with JS-verified layout."""
        # [status:1] [victim_id:2 LE] [promo_eligible:1] [killer_id:2 LE]
        data = bytes([0x05, 0x02, 0x01, 0x01, 0x04, 0x03])
        result = decode_deactivation(data)
        assert result["msg_type"] == 0x41
        assert result["status"] == 5
        assert result["victim_id"] == 0x0102
        assert result["promo_eligible"] is True
        assert result["killer_id"] == 0x0304
        assert result["is_mine_kill"] is False

    def test_decodes_mine_kill(self) -> None:
        """Decodes mine kill (killer_id >= 65530)."""
        # killer_id = 65530 + 2 = 65532 → mine from team 2 (blue)
        hi = (65532 >> 8) & 0xFF
        lo = 65532 & 0xFF
        data = bytes([0x00, 0x02, 0x01, 0x00, lo, hi])
        result = decode_deactivation(data)
        assert result["msg_type"] == 0x41
        assert result["is_mine_kill"] is True
        assert result["killer_id"] == 2

    def test_decodes_with_extra_bytes(self) -> None:
        """Decodes deactivation ignoring trailing bytes."""
        data = bytes([0x00, 0x02, 0x01, 0x00, 0x04, 0x03, 0xFF, 0xFF])
        result = decode_deactivation(data)
        assert result["msg_type"] == 0x41
        assert result["victim_id"] == 0x0102
        assert result["killer_id"] == 0x0304

    def test_decodes_real_wire_kill(self) -> None:
        """Real wire bytes from runs/bot/latest msg #89: Artax killed purple-8.

        The wire 7-byte container body has opcode 0x41 stripped by the
        protocol routing layer; this test decodes the remaining 6 bytes.
        Confirms wire 0x41 fires for own kills (prior live runs missed
        this because the routing min_len gate was too strict).
        """
        data = bytes.fromhex("01040201 1505".replace(" ", ""))
        result = decode_deactivation(data)
        assert result["msg_type"] == 0x41
        assert result["status"] == 1
        assert result["victim_id"] == 516  # purple-8
        assert result["promo_eligible"] is True
        assert result["killer_id"] == 1301  # Artax
        assert result["is_mine_kill"] is False

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data (need 6 bytes)."""
        with pytest.raises(DecodeError):
            decode_deactivation(bytes([1, 2, 3, 4, 5]))


# Protocol-layer mine_placement / mine_detonation decoders were deleted
# 2026-06-19. Both wire formats arrive only as container subtypes; their
# canonical decoders live in tankpit_bot.container.decoders.combat.
# Coverage for those lives in tests/container/test_tank_decoders.py.
