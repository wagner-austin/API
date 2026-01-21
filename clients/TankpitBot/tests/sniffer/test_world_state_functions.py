"""Tests for sniffer world state module-level functions."""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.sniffer import (
    reset_world_state,
    update_world_state_from_position,
    update_world_state_from_radar,
    world_state,
)


class TestPlayerIdMapper:
    """Tests for player ID mapper functions."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

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


class TestTextDecoding:
    """Tests for text decoding functions."""

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


class TestWorldStateGetter:
    """Tests for get_world_state function."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_get_world_state_returns_current_state(self) -> None:
        """Test get_world_state returns the current world state."""
        from tankpit_bot.sniffer.world_state import get_world_state

        # After reset, state should have None self_state
        state = get_world_state()
        assert state["self_state"] is None
        assert state["containers"] == {}
        assert state["mines"] == {}

        # Update position and verify state is updated
        update_world_state_from_position(50, 60)
        state = get_world_state()
        if state["self_state"] is None:
            raise AssertionError("self_state should not be None")
        assert state["self_state"]["x"] == 50
        assert state["self_state"]["y"] == 60


class TestContainerUpdate:
    """Tests for container update functions."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state and hooks after each test."""
        reset_world_state()
        _test_hooks.path_exists = _test_hooks._real_path_exists
        _test_hooks.load_terrain_map = _test_hooks._real_load_terrain_map

    def test_update_world_state_from_tank_registry_container_no_viewport(self) -> None:
        """Test container update when viewport_left is not known."""
        from tankpit_bot.sniffer import viewport
        from tankpit_bot.sniffer.world_state import (
            update_world_state_from_tank_registry_container,
        )

        # Ensure viewport_left is None
        viewport._viewport_left = None

        # Should log and return without updating
        update_world_state_from_tank_registry_container(100, 5)

        # No container should be added since viewport_left is unknown
        state = world_state._world_state
        assert state["containers"] == {}

    def test_update_world_state_from_tank_registry_container_with_viewport(self) -> None:
        """Test container update when viewport_left is known."""
        from tankpit_bot.sniffer import viewport
        from tankpit_bot.sniffer.world_state import (
            update_world_state_from_tank_registry_container,
        )

        # Set viewport_left
        viewport._viewport_left = 100

        # Update container - should calculate absolute x = 100 + 5 = 105
        update_world_state_from_tank_registry_container(50, 5)

        # Container should be added at (105, 50) with key "105,50"
        state = world_state._world_state
        assert "105,50" in state["containers"]

        # Reset viewport state
        viewport._viewport_left = None


class TestFuelUpdate:
    """Tests for fuel update functions."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_update_world_state_from_fuel_change(self) -> None:
        """Test fuel change updates self_state fuel."""
        from tankpit_bot.sniffer.world_state import update_world_state_from_fuel_change

        # First set up a position to create self_state
        update_world_state_from_position(100, 100)

        state = world_state._world_state
        if state["self_state"] is None:
            raise AssertionError("self_state should not be None")
        initial_fuel = state["self_state"]["fuel"]

        # Update fuel - adds to existing
        update_world_state_from_fuel_change(50)

        state = world_state._world_state
        if state["self_state"] is None:
            raise AssertionError("self_state should not be None")
        assert state["self_state"]["fuel"] == initial_fuel + 50

    def test_update_world_state_from_fuel_change_no_self_state(self) -> None:
        """Test fuel change does nothing when self_state is None."""
        from tankpit_bot.sniffer.world_state import update_world_state_from_fuel_change

        # Reset to ensure no self_state
        reset_world_state()

        # Verify self_state is None
        state = world_state._world_state
        assert state["self_state"] is None

        # Update fuel - should do nothing since no self_state
        update_world_state_from_fuel_change(50)

        state = world_state._world_state
        assert state["self_state"] is None


class TestContainerPickup:
    """Tests for container pickup functions."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_update_world_state_from_container_pickup(self) -> None:
        """Test container pickup removes container and adds fuel."""
        from tankpit_bot.container import RadarContainerDict, RadarMineDict
        from tankpit_bot.sniffer.world_state import (
            update_world_state_from_container_pickup,
        )

        # First set up a position to create self_state
        update_world_state_from_position(100, 100)

        # Add a container via radar
        containers: list[RadarContainerDict] = [RadarContainerDict(x=50, y=60, volume=100)]
        mines: list[RadarMineDict] = []
        update_world_state_from_radar(containers, mines)

        state = world_state._world_state
        assert "50,60" in state["containers"]

        # Pick up the container
        update_world_state_from_container_pickup(50, 60)

        state = world_state._world_state
        # Container should be removed
        assert "50,60" not in state["containers"]
