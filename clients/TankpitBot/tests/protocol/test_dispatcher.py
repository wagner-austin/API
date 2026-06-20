"""Tests for message type detection and main dispatcher.

Tests for is_text_message, decode_message, try_decode_message, try_decode_binary_message,
and message type constants.
"""

from __future__ import annotations

import pytest

from tankpit_bot.protocol import (
    MSG_ACTION_DONE,
    MSG_ACTIVE_FORCES,
    MSG_BUILD_PICKUP,
    MSG_CACHE_OVERLAY_UPDATE,
    MSG_CACHE_UPDATE,
    MSG_CHAT,
    MSG_DEACTIVATE,
    MSG_DECORATION,
    MSG_ENEMY_DETECT,
    MSG_EQUIP_GAIN,
    MSG_EQUIP_TOGGLE,
    MSG_FUEL_DEPOSIT,
    MSG_FUEL_GAIN,
    MSG_INVENTORY,
    MSG_MAP_DATA,
    MSG_MOVE_RESPONSE,
    MSG_MOVEMENT,
    MSG_OVERLAY_UPDATE,
    MSG_PROMOTION,
    MSG_RADAR_RESULT,
    MSG_SHOOT,
    MSG_STATISTICS,
    MSG_SUPERVISOR,
    MSG_SYNC,
    MSG_TANK_ENTRY,
    MSG_TANK_EXIT,
    MSG_TANK_INFO,
    MSG_TANK_POS,
    MSG_TANK_REMOVE,
    MSG_TANK_STATS,
    MSG_TANK_STATUS_FULL,
    MSG_TERRAIN_UPDATE,
    MSG_VIEWPORT,
    SUPERVISOR_ERROR_CANT_GO,
    SUPERVISOR_ERROR_INSUFFICIENT_FUEL,
    TEXT_MSG_TYPES,
    DecodeError,
    decode_message,
    is_text_message,
    try_decode_binary_message,
    try_decode_message,
)


class TestIsTextMessage:
    """Tests for is_text_message function."""

    def test_text_message_types(self) -> None:
        """Returns True for text message types."""
        assert is_text_message(MSG_TANK_POS) is True  # '='
        assert is_text_message(MSG_PROMOTION) is True  # '+'

    def test_binary_message_types(self) -> None:
        """Returns False for binary message types."""
        assert is_text_message(MSG_SHOOT) is False
        assert is_text_message(MSG_MOVEMENT) is False
        assert is_text_message(MSG_CACHE_UPDATE) is False

    def test_text_msg_types_constant(self) -> None:
        """TEXT_MSG_TYPES contains expected values."""
        assert MSG_TANK_POS in TEXT_MSG_TYPES
        assert MSG_PROMOTION in TEXT_MSG_TYPES


class TestDecodeMessage:
    """Tests for decode_message dispatcher function."""

    def test_dispatches_combat_messages(self) -> None:
        """Dispatches to combat message decoders."""
        # Shoot
        shoot_data = bytes([0x02, 0x01, 10, 20, 15, 25, 0x03, 0x04, 0x05, 1, 5, 0])
        result = decode_message(MSG_SHOOT, shoot_data)
        assert result["msg_type"] == 0x53

        # Deactivation
        deactivation_data = bytes([0x02, 0x01, 0x04, 0x03, 5, 0x07, 0x06])
        result = decode_message(MSG_DEACTIVATE, deactivation_data)
        assert result["msg_type"] == 0x41

        # 0x4B MinePlacement / 0x45 MineDetonation are container-only
        # subtypes -- protocol-layer routing for them was deleted
        # 2026-06-19. They're tested at the container layer.

    def test_dispatches_resource_messages(self) -> None:
        """Dispatches to resource message decoders."""
        # Fuel gain
        fuel_gain_data = bytes([0x34, 0x12, 1])
        result = decode_message(MSG_FUEL_GAIN, fuel_gain_data)
        assert result["msg_type"] == 0x44

        # Fuel deposit
        fuel_deposit_data = bytes([0x64, 0x00])
        result = decode_message(MSG_FUEL_DEPOSIT, fuel_deposit_data)
        assert result["msg_type"] == 0x64

        # Inventory
        inventory_data = bytes([1, 5, 10, 3, 7, 0])
        result = decode_message(MSG_INVENTORY, inventory_data)
        assert result["msg_type"] == 0x49

        # Equipment gain
        equip_gain_data = bytes([1, 1, 2, 0, 1, 0])
        result = decode_message(MSG_EQUIP_GAIN, equip_gain_data)
        assert result["msg_type"] == 0x67

        # Equipment toggle
        equip_toggle_data = bytes([1, 0, 1, 1, 0])
        result = decode_message(MSG_EQUIP_TOGGLE, equip_toggle_data)
        assert result["msg_type"] == 0x74

    def test_dispatches_radar_messages(self) -> None:
        """Dispatches to radar message decoders."""
        # Radar result
        radar_data = bytes([3, 1])
        result = decode_message(MSG_RADAR_RESULT, radar_data)
        assert result["msg_type"] == 0x46

        # Enemy detection
        enemy_data = bytes([0x02, 0x01, 50, 60, 4, 2])
        result = decode_message(MSG_ENEMY_DETECT, enemy_data)
        assert result["msg_type"] == 0x48

        # Combined tile update
        tile_data = bytes([1, 0, 10, 20, 0x34, 0x12, 30, 40, 7])
        result = decode_message(MSG_CACHE_OVERLAY_UPDATE, tile_data)
        assert result == {
            "msg_type": 0x4F,
            "cache_updates": [(10, 20, 0x1234)],
            "overlay_updates": [(30, 40, 7)],
        }

    def test_dispatches_tank_messages(self) -> None:
        """Dispatches to tank message decoders."""
        # Tank entry
        entry_data = bytes([5, 0x02, 0x01, 60, 0, 0, 0, 0, 0, 0])
        result = decode_message(MSG_TANK_ENTRY, entry_data)
        assert result["msg_type"] == 0x28

        # Tank exit (0x29 ')' - announcement, 5 bytes per JS Vf)
        exit_data = bytes([1, 0x02, 0x01, 0, 1])
        result = decode_message(MSG_TANK_EXIT, exit_data)
        assert result["msg_type"] == 0x29
        assert result["tank_id"] == 0x0102
        assert result["was_eliminated"] is True

        # Tank remove (0x58 'X' - server-driven physical removal, 2 bytes)
        remove_data = bytes([0x02, 0x01])
        result = decode_message(MSG_TANK_REMOVE, remove_data)
        assert result["msg_type"] == 0x58
        assert result["tank_id"] == 0x0102

        # Tank stats (0x2E) - uses container decoder. 0x53 ShootEvent
        # routes via the protocol tunnel path now; use teleport_landed
        # (1-byte) as a stable container-path witness.
        landed_data = bytes([0x54])
        result = decode_message(MSG_TANK_STATS, landed_data)
        assert result["msg_type"] == "teleport_landed"

        # Tunneled terrain/structure tile update inside 0x2E
        terrain_tunnel = bytes([MSG_TERRAIN_UPDATE, 8, 166, 2])
        result = decode_message(MSG_TANK_STATS, terrain_tunnel)
        assert result == {"msg_type": MSG_TERRAIN_UPDATE, "updates": [(8, 166, 2)]}

        # Tank status full
        status_header = bytes([0x42, 0x02, 0x01, 0xDE, 0xAD, 0xBE, 0xEF])
        status_lb = bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06])
        result = decode_message(MSG_TANK_STATUS_FULL, status_header + status_lb)
        assert result["msg_type"] == 0x3E

        # Tank info
        info_data = bytes([2, 0x02, 0x01, 0xDE, 0xAD, 0xBE, 0xEF, 0x03, 0x04, 0x05])
        result = decode_message(MSG_TANK_INFO, info_data)
        assert result["msg_type"] == 0x21

    def test_dispatches_movement_messages(self) -> None:
        """Dispatches to movement message decoders."""
        # Movement (12 bytes: tid(2)+pos(2)+dir+flag+lb(3)+rank+dmg+carry)
        movement_data = bytes([0x02, 0x01, 50, 60, 3, 1, 0x03, 0x04, 0x05, 0, 0, 0])
        result = decode_message(MSG_MOVEMENT, movement_data)
        assert result["msg_type"] == 0x47

        # Move response (12 bytes: team+tid(2)+pos(2)+dir+dmg+rank+lb(3)+carry)
        response_data = bytes([1, 0x02, 0x01, 50, 60, 3, 0x00, 4, 0x05, 0x06, 0x07, 0])
        result = decode_message(MSG_MOVE_RESPONSE, response_data)
        assert result["msg_type"] == 0x3D

    def test_dispatches_world_messages(self) -> None:
        """Dispatches to world message decoders."""
        # Viewport
        viewport_data = bytes([0, 0])
        result = decode_message(MSG_VIEWPORT, viewport_data)
        assert result["msg_type"] == 0x5A

        # Terrain update
        terrain_data = bytes([10, 20, 5])
        result = decode_message(MSG_TERRAIN_UPDATE, terrain_data)
        assert result["msg_type"] == 0x4A

        # Sync
        sync_data = b""
        result = decode_message(MSG_SYNC, sync_data)
        assert result["msg_type"] == 0x3F

        # Cache update
        container_data = bytes([10, 20, 0x04, 0x03])
        result = decode_message(MSG_CACHE_UPDATE, container_data)
        assert result["msg_type"] == 0x43
        assert result["updates"] == [(10, 20, 0x0304)]

        # Overlay update
        overlay_data = bytes([11, 21, 7, 12, 22, 255])
        result = decode_message(MSG_OVERLAY_UPDATE, overlay_data)
        assert result["msg_type"] == 0x40
        assert result["updates"] == [(11, 21, 7), (12, 22, 255)]

        # MapData (0x4C) -- empty body (just the u16 RLE header)
        result = decode_message(MSG_MAP_DATA, bytes([0, 0]))
        assert result["msg_type"] == 0x4C
        assert result["fuel_dots"] == []
        assert result["tanks"] == []

    def test_dispatches_misc_messages(self) -> None:
        """Dispatches to misc message decoders."""
        # Chat
        chat_data = bytes([0x02, 0x01, 1])
        result = decode_message(MSG_CHAT, chat_data)
        assert result["msg_type"] == 0x4D

        # Statistics
        stats_data = bytes([0x10, 0x00, 30, 45]) + bytes(12)
        result = decode_message(MSG_STATISTICS, stats_data)
        assert result["msg_type"] == 0x56

        # Active forces
        forces_data = bytes([10, 15, 8, 12])
        result = decode_message(MSG_ACTIVE_FORCES, forces_data)
        assert result["msg_type"] == 0x2A

        # Supervisor
        supervisor_data = bytes([1, 0, 3])
        result = decode_message(MSG_SUPERVISOR, supervisor_data)
        assert result["msg_type"] == 0x52

        # Action done
        action_data = b""
        result = decode_message(MSG_ACTION_DONE, action_data)
        assert result["msg_type"] == 0x54

        # SupervisorText (0x3C 'wg', free-form server message)
        result = decode_message(0x3C, b"Hello!")
        assert result["msg_type"] == 0x3C
        assert result["message"] == "Hello!"

        # Promotion (binary Rf form, disambiguated by routing)
        promo_data = bytes([3, 1])
        result = decode_message(MSG_PROMOTION, promo_data)
        assert result["msg_type"] == 0x2B
        assert result["new_rank"] == 3
        assert result["was_promoted"] is True

        # Decoration (Sf)
        deco_data = bytes([0x05, 0x00, 1, 2])
        result = decode_message(MSG_DECORATION, deco_data)
        assert result["msg_type"] == 0x4E
        assert result["tank_id"] == 5
        assert result["slot"] == 1
        assert result["level"] == 2

        # BuildPickup (Jg)
        build_data = bytes([0x10, 0x00, 1, 2, 3, 4, 7, 1, 0])
        result = decode_message(MSG_BUILD_PICKUP, build_data)
        assert result["msg_type"] == 0x42
        assert result["tank_id"] == 0x0010
        assert result["obstacle_type"] == 1

    def test_raises_on_unknown_type(self) -> None:
        """Raises DecodeError on unknown message type."""
        with pytest.raises(DecodeError) as exc:
            decode_message(0xFF, b"data")
        assert "unknown type 0xFF" in str(exc.value)


class TestTryDecodeMessage:
    """Tests for try_decode_message function."""

    def test_returns_decoded_for_known_types(self) -> None:
        """Returns decoded message for known types."""
        sync_data = b""
        result = try_decode_message(MSG_SYNC, sync_data)
        # Use decode_message which never returns None for known types
        # to verify the dispatch works correctly
        expected = decode_message(MSG_SYNC, sync_data)
        assert result == expected

    def test_returns_none_for_unknown_types(self) -> None:
        """Returns None for unknown message types."""
        result = try_decode_message(0xFF, b"data")
        assert result is None


class TestTryDecodeBinaryMessage:
    """Tests for try_decode_binary_message function."""

    def test_returns_decoded_for_known_types(self) -> None:
        """Returns decoded message for known types."""
        sync_data = b""
        result = try_decode_binary_message(MSG_SYNC, sync_data)
        # Use decode_message which never returns None for known types
        # to verify the dispatch works correctly
        expected = decode_message(MSG_SYNC, sync_data)
        assert result == expected

    def test_returns_none_for_unknown_types(self) -> None:
        """Returns None for unknown message types."""
        result = try_decode_binary_message(0xFF, b"data")
        assert result is None

    def test_returns_combat_message(self) -> None:
        """Returns decoded combat message (MSG_SHOOT)."""
        shoot_data = bytes([0x02, 0x01, 10, 20, 15, 25, 0x03, 0x04, 0x05, 1, 5, 0])
        result = try_decode_binary_message(MSG_SHOOT, shoot_data)
        expected = decode_message(MSG_SHOOT, shoot_data)
        assert result == expected

    def test_returns_resource_message(self) -> None:
        """Returns decoded resource message (MSG_FUEL_GAIN)."""
        fuel_data = bytes([0x34, 0x12, 1])
        result = try_decode_binary_message(MSG_FUEL_GAIN, fuel_data)
        expected = decode_message(MSG_FUEL_GAIN, fuel_data)
        assert result == expected

    def test_returns_radar_message(self) -> None:
        """Returns decoded radar message (MSG_RADAR_RESULT)."""
        radar_data = bytes([3, 1])
        result = try_decode_binary_message(MSG_RADAR_RESULT, radar_data)
        expected = decode_message(MSG_RADAR_RESULT, radar_data)
        assert result == expected

    def test_returns_tank_message(self) -> None:
        """Returns decoded tank message (MSG_TANK_ENTRY)."""
        entry_data = bytes([5, 0x02, 0x01, 60, 0, 0, 0, 0, 0, 0])
        result = try_decode_binary_message(MSG_TANK_ENTRY, entry_data)
        expected = decode_message(MSG_TANK_ENTRY, entry_data)
        assert result == expected

    def test_returns_movement_message(self) -> None:
        """Returns decoded movement message (MSG_MOVEMENT)."""
        movement_data = bytes([0x02, 0x01, 50, 60, 3, 1, 0x03, 0x04, 0x05, 0, 0, 0])
        result = try_decode_binary_message(MSG_MOVEMENT, movement_data)
        expected = decode_message(MSG_MOVEMENT, movement_data)
        assert result == expected

    def test_returns_misc_message(self) -> None:
        """Returns decoded misc message (MSG_CHAT)."""
        chat_data = bytes([0x02, 0x01, 1])
        result = try_decode_binary_message(MSG_CHAT, chat_data)
        expected = decode_message(MSG_CHAT, chat_data)
        assert result == expected


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


class TestMessageConstants:
    """Tests for message type constants."""

    def test_msg_constants_are_ascii_ord(self) -> None:
        """Message constants match ASCII ordinal values."""
        assert ord("S") == MSG_SHOOT
        assert ord("G") == MSG_MOVEMENT
        assert ord("A") == MSG_DEACTIVATE
        assert ord("?") == MSG_SYNC
        assert ord("!") == MSG_TANK_INFO
        assert ord("C") == MSG_CACHE_UPDATE
        assert ord("Z") == MSG_VIEWPORT

    def test_supervisor_error_constants(self) -> None:
        """Supervisor error code constants have expected values."""
        assert SUPERVISOR_ERROR_CANT_GO == 1
        assert SUPERVISOR_ERROR_INSUFFICIENT_FUEL == 8
