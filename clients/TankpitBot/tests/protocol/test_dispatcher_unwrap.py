"""Tests for the 0x2E tunnel unwrap.

Every subtype the tunneled dispatcher routes, and the container
fallthrough for the rest.
"""

from __future__ import annotations

from tankpit_bot.protocol import (
    MSG_ACTION_DONE,
    MSG_DEACTIVATE,
    MSG_DECORATION,
    MSG_EQUIP_GAIN,
    MSG_EQUIP_TOGGLE,
    MSG_FUEL_DEPOSIT,
    MSG_FUEL_GAIN,
    MSG_INVENTORY,
    MSG_MOVE_RESPONSE,
    MSG_MOVEMENT,
    MSG_PROMOTION,
    MSG_RADAR_RESULT,
    MSG_SHOOT,
    MSG_SYNC,
    MSG_TANK_ENTRY,
    MSG_TANK_EXIT,
    MSG_TANK_STATS,
    MSG_TANK_STATUS_FULL,
    MSG_VIEWPORT,
    decode_message,
)


class TestUnwrap0x2e:
    """Tests for tunneled protocol messages inside 0x2E envelopes."""

    # --- Resource messages ---

    def test_unwraps_inventory_from_0x2e(self) -> None:
        """Decodes tunneled 0x49 inventory from inside 0x2E."""
        data = bytes([MSG_INVENTORY, 1, 40, 40, 40, 40, 40])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == 0x49
        assert result["counts"] == [40, 40, 40, 40, 40]

    def test_unwraps_equipment_gain_from_0x2e(self) -> None:
        """Decodes tunneled 0x67 equipment gain from inside 0x2E."""
        data = bytes([MSG_EQUIP_GAIN, 1, 1, 2, 0, 1, 0])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == 0x67
        assert result["gained"] == [1, 2, 0, 1, 0]

    def test_unwraps_equipment_toggle_from_0x2e(self) -> None:
        """Decodes tunneled 0x74 equipment toggle from inside 0x2E."""
        data = bytes([MSG_EQUIP_TOGGLE, 0, 0, 1, 1, 0])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == 0x74
        assert result["enabled"] == [False, False, True, True, False]

    def test_unwraps_fuel_gain_from_0x2e(self) -> None:
        """Decodes tunneled 0x44 fuel gain from inside 0x2E."""
        data = bytes([MSG_FUEL_GAIN, 0x34, 0x12, 1])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == 0x44

    def test_unwraps_fuel_deposit_from_0x2e(self) -> None:
        """Decodes tunneled 0x64 fuel deposit from inside 0x2E."""
        data = bytes([MSG_FUEL_DEPOSIT, 0x64, 0x00])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == 0x64

    # --- Combat messages ---

    def test_unwraps_shoot_event_from_0x2e(self) -> None:
        """Decodes tunneled 0x53 shoot event from inside 0x2E."""
        data = bytes([MSG_SHOOT, 0x02, 0x01, 10, 20, 15, 25, 0x03, 0x04, 0x05, 1, 5, 0])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == 0x53

    def test_unwraps_deactivation_from_0x2e(self) -> None:
        """Decodes tunneled 0x41 deactivation from inside 0x2E."""
        data = bytes([MSG_DEACTIVATE, 0x02, 0x01, 0x04, 0x03, 5, 0x07, 0x06])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == 0x41

    # --- Radar messages ---

    def test_unwraps_radar_result_from_0x2e(self) -> None:
        """Decodes tunneled 0x46 radar result from inside 0x2E."""
        data = bytes([MSG_RADAR_RESULT, 3, 1])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == 0x46

    # --- Tank messages ---

    def test_unwraps_tank_entry_from_0x2e(self) -> None:
        """Decodes tunneled 0x28 tank entry from inside 0x2E."""
        data = bytes([MSG_TANK_ENTRY, 5, 0x02, 0x01, 60, 0, 0, 0, 0, 0, 0])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == 0x28

    def test_unwraps_tank_exit_from_0x2e(self) -> None:
        """Decodes tunneled 0x29 tank exit from inside 0x2E.

        Fifteen archived bodies were landing in ``unknown_container``
        for want of this arm ([[session-state-deglobalisation]]).
        """
        data = bytes([MSG_TANK_EXIT, 0x02, 0x95, 0x03, 0x00, 0x00])
        result = decode_message(MSG_TANK_STATS, data)
        assert result == {
            "msg_type": 0x29,
            "team": 2,
            "tank_id": 917,
            "was_silent": False,
            "was_eliminated": False,
        }

    def test_tank_exit_arm_is_exact_length(self) -> None:
        """A 0x29 body of any other length keeps its container route."""
        data = bytes([MSG_TANK_EXIT, 0x02, 0x95, 0x03, 0x00, 0x00, 0x00])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == "unknown_container"

    def test_unwraps_promotion_from_0x2e(self) -> None:
        """Decodes tunneled binary 0x2B promotion from inside 0x2E."""
        data = bytes([MSG_PROMOTION, 0x01, 0x01])
        result = decode_message(MSG_TANK_STATS, data)
        assert result == {"msg_type": 0x2B, "new_rank": 1, "was_promoted": True}

    def test_promotion_arm_is_exact_length(self) -> None:
        """0x2B is also the TEXT room-list byte, so its arm stays tight."""
        data = bytes([MSG_PROMOTION, 0x01, 0x01, 0x01])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == "unknown_container"

    def test_unwraps_decoration_from_0x2e(self) -> None:
        """Decodes tunneled 0x4E decoration/award from inside 0x2E."""
        data = bytes([MSG_DECORATION, 0x15, 0x05, 0x01, 0x03])
        result = decode_message(MSG_TANK_STATS, data)
        assert result == {"msg_type": 0x4E, "tank_id": 1301, "slot": 1, "level": 3}

    def test_decoration_arm_is_exact_length(self) -> None:
        """A 0x4E body of any other length keeps its container route."""
        data = bytes([MSG_DECORATION, 0x15, 0x05, 0x01])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == "unknown_container"

    def test_unwraps_tank_status_full_from_0x2e(self) -> None:
        """Decodes tunneled 0x3E tank status from inside 0x2E."""
        header = bytes([MSG_TANK_STATUS_FULL, 0x42, 0x02, 0x01, 0xDE, 0xAD, 0xBE, 0xEF])
        lb = bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06])
        result = decode_message(MSG_TANK_STATS, header + lb)
        assert result["msg_type"] == 0x3E

    # --- Movement messages ---

    def test_unwraps_movement_from_0x2e(self) -> None:
        """Decodes tunneled 0x47 movement from inside 0x2E."""
        data = bytes([MSG_MOVEMENT, 0x02, 0x01, 50, 60, 3, 1, 0x03, 0x04, 0x05, 0, 0, 0])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == 0x47

    def test_unwraps_move_response_from_0x2e(self) -> None:
        """Decodes tunneled 0x3D move response from inside 0x2E (12 inner bytes)."""
        data = bytes([MSG_MOVE_RESPONSE, 1, 0x02, 0x01, 50, 60, 3, 0x00, 4, 0x05, 0x06, 0x07, 0])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == 0x3D

    # --- World messages ---

    def test_unwraps_sync_from_0x2e(self) -> None:
        """Decodes tunneled 0x3F sync from inside 0x2E."""
        data = bytes([MSG_SYNC, 0x01])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == 0x3F

    def test_unwraps_viewport_from_0x2e(self) -> None:
        """Decodes tunneled 0x5A viewport from inside 0x2E."""
        data = bytes([MSG_VIEWPORT, 0, 0])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == 0x5A

    # --- Misc messages ---

    def test_unwraps_action_done_from_0x2e(self) -> None:
        """Decodes tunneled 0x54 action done from inside 0x2E."""
        data = bytes([MSG_ACTION_DONE, 0x00])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == 0x54

    # --- Fallback to container decoder ---

    def test_nested_0x2e_decodes_as_tank_status_sync(self) -> None:
        """Nested 0x2E decodes as TankStatusSync with fuel from long format."""
        # Long format: subtype=3, tank_id=1227, damage=2, rank=4, score, lb, flag, fuel
        inner = bytes([0x03, 0xCB, 0x04, 0x02, 0x04, 0x00, 0x22, 0x84, 0x08, 0x00, 0x78, 0x05])
        data = bytes([MSG_TANK_STATS]) + inner
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == 0x2E
        assert result["tank_id"] == 1227
        assert result["fuel"] == 1400

    def test_nested_0x2e_short_falls_through_to_container(self) -> None:
        """Nested 0x2E with < 9 bytes falls through to container decoder.

        Container's TankStatusSync (2-3 byte catch-all) was deleted
        2026-06-19; short bodies now resolve to unknown_container.
        """
        data = bytes([MSG_TANK_STATS, 0x01])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == "unknown_container"

    def test_short_data_falls_through_to_container(self) -> None:
        """Single byte data is too short for unwrap, goes to container."""
        data = bytes([0x49])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == "teleport_landed"

    def test_decode_error_falls_through_to_container(self) -> None:
        """DecodeError in protocol decoder falls through to container."""
        # 0x49 subtype but only 2 bytes total — too short for inventory
        data = bytes([MSG_INVENTORY, 0x01])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == "unknown_container"

    def test_unknown_subtype_falls_through_to_container(self) -> None:
        """Unknown subtype falls through to container structure matching."""
        data = bytes([0xFF, 0x01])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == "unknown_container"

    def test_tunneled_0x4f_with_bad_structure_falls_through(self) -> None:
        """Tunneled 0x4F with invalid radar scan structure falls through."""
        # 0x4F subtype but inner data has container_count=1 and only 3 bytes
        # (needs at least 2 + 4 = 6). Structural check fails → falls through
        # to container identification. Result is not the 0x4F radar type.
        data = bytes([0x4F, 0x01, 0x00, 0xAA])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] != 0x4F

    def test_tunneled_0x4f_with_valid_radar_structure_decodes(self) -> None:
        """Tunneled 0x4F with valid radar scan structure decodes as radar."""
        # 0x4F subtype, container_count=0, flags=0, no mines → valid empty radar
        data = bytes([0x4F, 0x00, 0x00])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == 0x4F

    def test_tunneled_0x4f_with_bad_remaining_bytes_falls_through(self) -> None:
        """Tunneled 0x4F with remaining bytes not divisible by 3 falls through."""
        # container_count=0, flags=0, then 2 extra bytes (not divisible by 3)
        data = bytes([0x4F, 0x00, 0x00, 0xAA, 0xBB])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] != 0x4F

    def test_tunneled_0x4f_too_short_falls_through(self) -> None:
        """Tunneled 0x4F with insufficient data falls through to container."""
        # 0x4F subtype but only 1 inner byte — too short for structural check.
        data = bytes([0x4F, 0x00])
        result = decode_message(MSG_TANK_STATS, data)
        assert result["msg_type"] == "unknown_container"
