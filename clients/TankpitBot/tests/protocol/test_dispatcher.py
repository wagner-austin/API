"""Tests for message-type detection and the decode dispatchers.

``test_dispatcher.py`` was 642 lines; the 0x2E unwrap suite is now a
sibling.
"""

from __future__ import annotations

import pytest

from tankpit_bot.protocol import (
    MSG_ACTION_DONE,
    MSG_ACTIVE_FORCES,
    MSG_BUILD_PICKUP,
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
    MSG_RADAR_SCAN,
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
    decode_message,
    is_text_message,
    try_decode_binary_message,
)
from tankpit_bot.wire.helpers import DecodeError


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

        # Radar scan result (0x4F -- single wire personality per JS ch)
        tile_data = bytes([1, 0, 10, 20, 0x34, 0x12, 30, 40, 7])
        result = decode_message(MSG_RADAR_SCAN, tile_data)
        assert result == {
            "msg_type": 0x4F,
            "containers": [{"x": 10, "y": 20, "volume": 0x1234}],
            "mines": [{"x": 30, "y": 40, "team": 3}],
            "mine_clears": [],
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

        # Session-broadcast messages route via _decode_session_broadcast.
        from tankpit_bot.protocol import (
            MSG_ACTIVE_PLAYERS,
            MSG_DISCONNECT,
            MSG_PING,
            MSG_TOP10,
        )

        # 0x2F ActivePlayers
        active_players_data = bytes([0xF5, 0x01, 0x05])  # tank_id=501, rank=5
        result = decode_message(MSG_ACTIVE_PLAYERS, active_players_data)
        assert result["msg_type"] == 0x2F

        # 0x31 Top10 (header-only)
        top10_data = bytes([0xFF, 0x00, 0x00, 0x00, 0x00])
        result = decode_message(MSG_TOP10, top10_data)
        assert result["msg_type"] == 0x31

        # 0x60 PingResponse (bare)
        result = decode_message(MSG_PING, b"")
        assert result["msg_type"] == 0x60

        # 0x7E ConnectionLost (bare)
        result = decode_message(MSG_DISCONNECT, b"")
        assert result["msg_type"] == 0x7E

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
