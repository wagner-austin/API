"""Tests for tankpit_bot.sniffer world state integration functions."""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.sniffer import (
    dispatch_world_state_update,
    render_world_state_ascii,
    reset_world_state,
    update_world_state_from_position,
    update_world_state_from_radar,
    world_state,
)
from tests.fakes import FakeTerrainMap


class TestWorldStateIntegration:
    """Tests for sniffer world state integration functions."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state and hooks after each test."""
        reset_world_state()
        _test_hooks.path_exists = _test_hooks._real_path_exists
        _test_hooks.load_terrain_map = _test_hooks._real_load_terrain_map

    def test_reset_world_state_clears_state(self) -> None:
        """Test reset_world_state clears world state and terrain map."""
        update_world_state_from_position(100, 100)
        reset_world_state()

        assert world_state._world_state["self_state"] is None
        assert world_state._terrain_map is None

    def test_load_terrain_map_returns_none_if_no_file(self) -> None:
        """Test returns None when no terrain file exists."""
        from tankpit_bot.sniffer.world_state import _load_terrain_map_if_needed

        _test_hooks.path_exists = lambda path: False

        result = _load_terrain_map_if_needed()
        assert result is None

    def test_load_terrain_map_caches_result(self) -> None:
        """Test terrain map is cached after first load."""
        from tankpit_bot.sniffer.world_state import _load_terrain_map_if_needed

        fake_terrain = FakeTerrainMap()

        _test_hooks.path_exists = lambda path: True
        _test_hooks.load_terrain_map = lambda path: fake_terrain

        result1 = _load_terrain_map_if_needed()
        assert result1 is fake_terrain
        assert world_state._terrain_map is fake_terrain

        result2 = _load_terrain_map_if_needed()
        assert result2 is fake_terrain

    def test_update_world_state_from_position(self) -> None:
        """Test updates self position in world state."""
        update_world_state_from_position(128, 64)

        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None after position update")
        assert self_state["x"] == 128
        assert self_state["y"] == 64

    def test_update_world_state_from_position_updates_existing(self) -> None:
        """Test updates existing self position in world state."""
        # First call creates self_state
        update_world_state_from_position(100, 100)
        # Second call updates existing self_state
        update_world_state_from_position(200, 150)

        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None after position update")
        assert self_state["x"] == 200
        assert self_state["y"] == 150

    def test_update_world_state_from_radar_containers(self) -> None:
        """Test updates containers from radar."""
        from tankpit_bot.container import RadarContainerDict, RadarMineDict

        containers: list[RadarContainerDict] = [
            RadarContainerDict(x=50, y=60, volume=100),  # fuel with 100 units
            RadarContainerDict(x=55, y=65, volume=-1),  # equipment (volume=-1)
        ]
        mines: list[RadarMineDict] = []

        update_world_state_from_radar(containers, mines)

        assert "50,60" in world_state._world_state["containers"]
        assert world_state._world_state["containers"]["50,60"]["is_fuel"] is True
        assert "55,65" in world_state._world_state["containers"]
        assert world_state._world_state["containers"]["55,65"]["is_fuel"] is False

    def test_update_world_state_from_radar_mines(self) -> None:
        """Test updates mines from radar."""
        from tankpit_bot.container import RadarContainerDict, RadarMineDict

        containers: list[RadarContainerDict] = []
        mines: list[RadarMineDict] = [
            RadarMineDict(x=70, y=80, team=1),
            RadarMineDict(x=75, y=85, team=2),
        ]

        update_world_state_from_radar(containers, mines)

        assert "70,80" in world_state._world_state["mines"]
        assert world_state._world_state["mines"]["70,80"]["team"] == 1
        assert "75,85" in world_state._world_state["mines"]

    def test_render_world_state_ascii_returns_none_without_terrain(self) -> None:
        """Test returns None when no terrain file exists."""
        _test_hooks.path_exists = lambda path: False

        result = render_world_state_ascii()
        assert result is None

    def test_render_world_state_ascii_with_terrain(self) -> None:
        """Test renders ASCII with terrain map."""
        fake_terrain = FakeTerrainMap()
        _test_hooks.path_exists = lambda path: True
        _test_hooks.load_terrain_map = lambda path: fake_terrain

        update_world_state_from_position(128, 128)

        result = render_world_state_ascii()
        if result is None:
            raise AssertionError("expected string, got None")
        assert "Viewport:" in result
        assert "@" in result

    def test_dispatch_radar_response(self) -> None:
        """Test dispatch handles radar_response message."""
        from tankpit_bot.container import (
            RadarContainerDict,
            RadarMineDict,
            RadarResponseDict,
        )

        _test_hooks.path_exists = lambda path: False

        msg = RadarResponseDict(
            msg_type="radar_response",
            container_count=1,
            containers=[RadarContainerDict(x=100, y=100, volume=50)],
            mines=[RadarMineDict(x=110, y=110, team=0)],
        )

        dispatch_world_state_update(msg)

        assert "100,100" in world_state._world_state["containers"]
        assert "110,110" in world_state._world_state["mines"]

    def test_dispatch_radar_response_renders_ascii(self) -> None:
        """Test dispatch renders ASCII after radar update when terrain available."""
        from tankpit_bot.container import RadarContainerDict, RadarResponseDict

        fake_terrain = FakeTerrainMap()
        _test_hooks.path_exists = lambda path: True
        _test_hooks.load_terrain_map = lambda path: fake_terrain

        update_world_state_from_position(128, 128)

        msg = RadarResponseDict(
            msg_type="radar_response",
            container_count=1,
            containers=[RadarContainerDict(x=128, y=128, volume=50)],
            mines=[],
        )

        dispatch_world_state_update(msg)

    def test_dispatch_movement_response(self) -> None:
        """Test dispatch handles MovementResponse (0x3D) message."""
        from tankpit_bot.protocol import MovementResponseDict

        msg = MovementResponseDict(
            msg_type=0x3D,
            team=0,
            tank_id=1,
            x=150,
            y=160,
            direction=0,
            rank=1,
            leaderboard_position=5,
        )

        dispatch_world_state_update(msg)

        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None after dispatch")
        assert self_state["x"] == 150
        assert self_state["y"] == 160

    def test_dispatch_position_update_absolute_coords_self(self) -> None:
        """Test dispatch handles position_update with absolute coords for self.

        When x or y >= 18 and flags indicate self (0x02 bit set),
        should update the self position.
        """
        from tankpit_bot.container import PositionUpdateDict

        msg = PositionUpdateDict(
            msg_type="position_update",
            flags=0x02,  # Self flag
            tank_id=638,
            x=202,
            y=149,
            extra_data=b"\x08\x03\x03\x00\x2e\x84\x00",
        )

        dispatch_world_state_update(msg)

        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None after dispatch")
        assert self_state["x"] == 202
        assert self_state["y"] == 149

    def test_dispatch_position_update_other_tank_ignored(self) -> None:
        """Test dispatch ignores position_update for other tanks.

        Position updates with flags=0x00 are for other tanks (bots, enemies)
        and should NOT update self position.
        """
        from tankpit_bot.container import PositionUpdateDict

        # First set a known position
        update_world_state_from_position(100, 100)

        msg = PositionUpdateDict(
            msg_type="position_update",
            flags=0x00,  # Other tank flag
            tank_id=539,
            x=193,
            y=150,
            extra_data=b"\x08\x03\x01\x00\x48\xe2\x00",
        )

        dispatch_world_state_update(msg)

        # Position should remain unchanged
        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["x"] == 100
        assert self_state["y"] == 100

    def test_dispatch_position_update_viewport_relative_ignored(self) -> None:
        """Test dispatch ignores position_update with viewport-relative coords.

        When both x < 18 and y < 18, they are viewport-relative coordinates
        and should NOT update the self position even with self flag.
        """
        from tankpit_bot.container import PositionUpdateDict

        # First set a known position
        update_world_state_from_position(100, 100)

        msg = PositionUpdateDict(
            msg_type="position_update",
            flags=0x02,  # Self flag
            tank_id=638,
            x=3,
            y=3,
            extra_data=b"\x00\x2e\x85\x0a\x00\x0c\x05",
        )

        dispatch_world_state_update(msg)

        # Position should remain unchanged
        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["x"] == 100
        assert self_state["y"] == 100

    def test_dispatch_position_update_viewport_relative_other_coords_ignored(self) -> None:
        """Test dispatch ignores position_update with any small viewport coords.

        Any coords where both x < 18 and y < 18 are viewport-relative,
        not just (3,3). For example, (2,3) at the left edge of viewport.
        """
        from tankpit_bot.container import PositionUpdateDict

        # First set a known position
        update_world_state_from_position(100, 100)

        msg = PositionUpdateDict(
            msg_type="position_update",
            flags=0x02,  # Self flag
            tank_id=638,
            x=2,
            y=3,
            extra_data=b"\x00" * 7,
        )

        dispatch_world_state_update(msg)

        # Position should remain unchanged
        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["x"] == 100
        assert self_state["y"] == 100

    def test_dispatch_movement_updates_self_position(self) -> None:
        """Test dispatch handles movement message for self.

        Movement messages have absolute start_x, start_y coordinates.
        Final position is calculated by applying waypoints to start position.
        Path: eeeessssssseeeeeeeeennnnnnn = 4e + 7s + 9e + 7n
        Final: (162 + 4 + 9, 111 + 7 - 7) = (175, 111)
        """
        from tankpit_bot.container import MovementDict

        msg = MovementDict(
            msg_type="movement",
            flags=0x7E,
            start_x=162,
            start_y=111,
            player_id=230446,
            tank_id=None,
            waypoints="eeeessssssseeeeeeeeennnnnnn",
            is_self=True,
        )

        dispatch_world_state_update(msg)

        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None after dispatch")
        # Final position after applying waypoints
        assert self_state["x"] == 175
        assert self_state["y"] == 111

    def test_dispatch_movement_ignores_enemy(self) -> None:
        """Test dispatch ignores movement message for enemies.

        Movement messages with is_self=False should not update self position.
        """
        from tankpit_bot.container import MovementDict

        # First set a known position
        update_world_state_from_position(100, 100)

        msg = MovementDict(
            msg_type="movement",
            flags=0x1E,
            start_x=200,
            start_y=200,
            player_id=12345,
            tank_id=None,
            waypoints="nnnn",
            is_self=False,
        )

        dispatch_world_state_update(msg)

        # Position should remain unchanged
        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["x"] == 100
        assert self_state["y"] == 100

    def test_dispatch_ignores_other_messages(self) -> None:
        """Test dispatch ignores non-radar/movement messages."""
        from tankpit_bot.protocol import SyncDict

        initial_self = world_state._world_state["self_state"]

        msg = SyncDict(msg_type=0x3F)

        dispatch_world_state_update(msg)

        assert world_state._world_state["self_state"] == initial_self

    def test_dispatch_handles_empty_radar_response(self) -> None:
        """Test dispatch handles radar_response with empty data."""
        from tankpit_bot.container import RadarResponseDict

        _test_hooks.path_exists = lambda path: False

        msg: RadarResponseDict = {
            "msg_type": "radar_response",
            "container_count": 0,
            "containers": [],
            "mines": [],
        }

        dispatch_world_state_update(msg)


# =============================================================================
# _apply_waypoints Tests
# =============================================================================


class TestApplyWaypoints:
    """Tests for _apply_waypoints helper function."""

    def test_apply_waypoints_empty(self) -> None:
        """Test empty waypoints returns start position."""
        x, y = world_state._apply_waypoints(100, 100, "")
        assert x == 100
        assert y == 100

    def test_apply_waypoints_north(self) -> None:
        """Test north direction decreases y."""
        x, y = world_state._apply_waypoints(100, 100, "nnn")
        assert x == 100
        assert y == 97

    def test_apply_waypoints_south(self) -> None:
        """Test south direction increases y."""
        x, y = world_state._apply_waypoints(100, 100, "sss")
        assert x == 100
        assert y == 103

    def test_apply_waypoints_east(self) -> None:
        """Test east direction increases x."""
        x, y = world_state._apply_waypoints(100, 100, "eee")
        assert x == 103
        assert y == 100

    def test_apply_waypoints_west(self) -> None:
        """Test west direction decreases x."""
        x, y = world_state._apply_waypoints(100, 100, "www")
        assert x == 97
        assert y == 100

    def test_apply_waypoints_mixed(self) -> None:
        """Test mixed waypoints."""
        # wsss = west, south, south, south
        x, y = world_state._apply_waypoints(100, 100, "wsss")
        assert x == 99
        assert y == 103

    def test_apply_waypoints_complex_path(self) -> None:
        """Test complex path from actual game data."""
        # eeeessssssseeeeeeeeennnnnnn = 4e + 7s + 9e + 7n
        # Final: (100 + 4 + 9, 100 + 7 - 7) = (113, 100)
        x, y = world_state._apply_waypoints(100, 100, "eeeessssssseeeeeeeeennnnnnn")
        assert x == 113
        assert y == 100

    def test_apply_waypoints_west_then_continue(self) -> None:
        """Test west followed by other directions (ensures loop continuation after w)."""
        # wne = west, north, east -> back to start
        x, y = world_state._apply_waypoints(100, 100, "wne")
        assert x == 100
        assert y == 99

    def test_apply_waypoints_ignores_unknown_characters(self) -> None:
        """Test unknown characters are ignored (covers else branch)."""
        # "nXs" = north, unknown 'X', south -> net y stays same
        x, y = world_state._apply_waypoints(100, 100, "nXs")
        assert x == 100
        assert y == 100


# =============================================================================
# Module-Level Function Tests
# =============================================================================


class TestModuleLevelFunctions:
    """Tests for new module-level functions in sniffer."""

    def test_reset_player_id_mapper(self) -> None:
        """Test reset_player_id_mapper clears state."""
        from tankpit_bot.sniffer import player_tracking, reset_player_id_mapper

        # Add some data
        player_tracking._player_id_mapper._player_to_tank[1] = 100
        player_tracking._tank_names[100] = "TestTank"

        reset_player_id_mapper()

        assert len(player_tracking._player_id_mapper._player_to_tank) == 0
        assert len(player_tracking._tank_names) == 0

    def test_resolve_movement_tank_no_mapping(self) -> None:
        """Test resolve_movement_tank returns pid when no mapping exists."""
        from tankpit_bot.sniffer import reset_player_id_mapper, resolve_movement_tank

        reset_player_id_mapper()
        result = resolve_movement_tank(42, 100, 100)
        assert result == "pid=42"

    def test_resolve_movement_tank_with_mapping(self) -> None:
        """Test resolve_movement_tank returns tank_id when mapped."""
        from tankpit_bot.sniffer import (
            player_tracking,
            reset_player_id_mapper,
            resolve_movement_tank,
        )

        reset_player_id_mapper()
        player_tracking._player_id_mapper._player_to_tank[42] = 100

        result = resolve_movement_tank(42, 100, 100)
        assert result == "tank=100"

    def test_resolve_movement_tank_with_name(self) -> None:
        """Test resolve_movement_tank returns name when available."""
        from tankpit_bot.sniffer import (
            player_tracking,
            reset_player_id_mapper,
            resolve_movement_tank,
        )

        reset_player_id_mapper()
        player_tracking._player_id_mapper._player_to_tank[42] = 100
        player_tracking._tank_names[100] = "CoolTank"

        result = resolve_movement_tank(42, 100, 100)
        assert result == '"CoolTank"'

    def test_resolve_movement_tank_position_correlation(self) -> None:
        """Test resolve_movement_tank uses position correlation."""
        from tankpit_bot.sniffer import (
            player_tracking,
            reset_player_id_mapper,
            resolve_movement_tank,
        )

        reset_player_id_mapper()
        # Set up position-to-tank mapping
        player_tracking._player_id_mapper._position_to_tank[(50, 60)] = 200

        result = resolve_movement_tank(42, 50, 60)
        assert result == "tank=200"
        # Verify mapping was cached
        assert player_tracking._player_id_mapper._player_to_tank[42] == 200

    def test_try_decode_received_text_invalid_base64_length(self) -> None:
        """Test try_decode_received_text returns None for invalid base64 length."""
        from tankpit_bot.sniffer import try_decode_received_text

        # Length not multiple of 4
        result = try_decode_received_text("abc")
        assert result is None

    def test_try_decode_received_text_invalid_chars(self) -> None:
        """Test try_decode_received_text returns None for invalid chars."""
        from tankpit_bot.sniffer import try_decode_received_text

        # Invalid character !
        result = try_decode_received_text("abc!")
        assert result is None

    def test_try_decode_received_text_too_short(self) -> None:
        """Test try_decode_received_text returns None for too short data."""
        import base64

        from tankpit_bot.sniffer import try_decode_received_text

        # Only 1 byte of data (less than 2)
        result = try_decode_received_text(base64.b64encode(b"x").decode())
        assert result is None

    def test_try_decode_received_text_empty_body(self) -> None:
        """Test try_decode_received_text returns None for empty body."""
        import base64

        from tankpit_bot.sniffer import try_decode_received_text

        # 2 bytes header, no body
        result = try_decode_received_text(base64.b64encode(b"\x00\x00").decode())
        assert result is None

    def test_try_decode_received_text_non_text_type(self) -> None:
        """Test try_decode_received_text returns None for non-text message types."""
        import base64

        from tankpit_bot.sniffer import try_decode_received_text

        # First byte 0x00 is not a text message type
        result = try_decode_received_text(base64.b64encode(b"\x03\x00\x00data").decode())
        assert result is None

    def test_try_decode_received_text_valid_join_confirm(self) -> None:
        """Test try_decode_received_text decodes valid join confirm."""
        import base64

        from tankpit_bot.sniffer import try_decode_received_text

        # '+' (0x2B) is a text message type for JOIN_CONFIRM
        body = b"+field=42\n"
        payload = base64.b64encode(len(body).to_bytes(2, "little") + body).decode()
        result = try_decode_received_text(payload)
        if result is None:
            raise AssertionError("expected non-None result")
        assert "JOIN_CONFIRM" in result or "field=42" in result

    def test_decode_received_text_message_logs_result(self) -> None:
        """Test decode_received_text_message logs when result is not None."""
        import base64

        from tankpit_bot.sniffer import decode_received_text_message

        # This tests the logging branch when result is not None
        body = b"+field=42\n"
        payload = base64.b64encode(len(body).to_bytes(2, "little") + body).decode()
        # Should not raise, just logs
        decode_received_text_message(payload)

    def test_decode_received_text_message_no_log_for_none(self) -> None:
        """Test decode_received_text_message does not log when result is None."""
        from tankpit_bot.sniffer import decode_received_text_message

        # Invalid payload, result is None, should not log
        decode_received_text_message("xxx")
