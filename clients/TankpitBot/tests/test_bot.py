"""Tests for tankpit_bot.bot module."""

from __future__ import annotations

import pytest

from tankpit_bot.bot import BotError, ProtocolNotDiscoveredError, main
from tests.conftest import FakeEnv


def test_main_prints_instructions(
    fake_env: FakeEnv,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() prints usage instructions."""
    main()

    captured = capsys.readouterr()
    lines = captured.out.split("\n")
    assert lines[0] == "TankpitBot - Automated Tankpit.com player"
    assert lines[4] == "  1. Run the sniffer to capture WebSocket traffic:"
    assert lines[5] == "     make sniff"


def test_main_uses_custom_capture_path(
    fake_env: FakeEnv,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() uses custom capture path from env."""
    fake_env.set("TANKPIT_CAPTURE", "custom_capture.json")

    main()

    captured = capsys.readouterr()
    output_lines = captured.out.strip().split("\n")
    # Find the line after "save the captured protocol to:"
    for i, line in enumerate(output_lines):
        if "save the captured protocol to:" in line:
            assert output_lines[i + 1].strip() == "custom_capture.json"
            return
    raise AssertionError("Expected 'save the captured protocol' line not found")


def test_main_default_capture_path(
    fake_env: FakeEnv,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() uses default capture path when env not set."""
    main()

    captured = capsys.readouterr()
    output_lines = captured.out.strip().split("\n")
    # Find the line after "save the captured protocol to:"
    for i, line in enumerate(output_lines):
        if "save the captured protocol to:" in line:
            assert output_lines[i + 1].strip() == "capture_session.json"
            return
    raise AssertionError("Expected 'save the captured protocol' line not found")


def test_bot_error_is_exception() -> None:
    """Test BotError is an Exception."""
    assert issubclass(BotError, Exception)
    err = BotError("test error")
    assert str(err) == "test error"


def test_protocol_not_discovered_error_is_bot_error() -> None:
    """Test ProtocolNotDiscoveredError is a BotError."""
    assert issubclass(ProtocolNotDiscoveredError, BotError)


# =============================================================================
# Bot Command Types Tests
# =============================================================================


class TestBotCommandTypes:
    """Tests for bot command factory functions."""

    def test_make_move_command(self) -> None:
        """Test make_move_command creates correct TypedDict."""
        from tankpit_bot.bot import make_move_command

        cmd = make_move_command(100, 150)
        assert cmd["cmd_type"] == "move"
        assert cmd["target_x"] == 100
        assert cmd["target_y"] == 150

    def test_make_shoot_command(self) -> None:
        """Test make_shoot_command creates correct TypedDict."""
        from tankpit_bot.bot import make_shoot_command

        cmd = make_shoot_command(50, 75)
        assert cmd["cmd_type"] == "shoot"
        assert cmd["target_x"] == 50
        assert cmd["target_y"] == 75

    def test_make_radar_command(self) -> None:
        """Test make_radar_command creates correct TypedDict."""
        from tankpit_bot.bot import make_radar_command

        cmd = make_radar_command()
        assert cmd["cmd_type"] == "radar"

    def test_make_pickup_move_command(self) -> None:
        """Test make_pickup_move_command creates correct TypedDict."""
        from tankpit_bot.bot import make_pickup_move_command

        cmd = make_pickup_move_command(200, 100)
        assert cmd["cmd_type"] == "pickup_move"
        assert cmd["target_x"] == 200
        assert cmd["target_y"] == 100

    def test_make_teleport_command(self) -> None:
        """Test make_teleport_command creates correct TypedDict."""
        from tankpit_bot.bot import make_teleport_command

        cmd = make_teleport_command(128, 64)
        assert cmd["cmd_type"] == "teleport"
        assert cmd["target_x"] == 128
        assert cmd["target_y"] == 64


# =============================================================================
# Bot Command Encoding Tests
# =============================================================================


class TestBotCommandEncoding:
    """Tests for bot command encoding functions."""

    def test_encode_move_command(self) -> None:
        """Test encode_move_command encodes command to expected bytes."""
        from tankpit_bot.bot import encode_move_command, make_move_command
        from tankpit_bot.protocol.commands import build_move_command

        cmd = make_move_command(100, 150)
        result = encode_move_command(cmd)
        expected = build_move_command(100, 150)
        assert result == expected

    def test_encode_shoot_command(self) -> None:
        """Test encode_shoot_command encodes command to expected bytes."""
        from tankpit_bot.bot import encode_shoot_command, make_shoot_command
        from tankpit_bot.protocol.commands import build_shoot_command

        cmd = make_shoot_command(50, 75)
        result = encode_shoot_command(cmd)
        expected = build_shoot_command(50, 75)
        assert result == expected

    def test_encode_radar_command(self) -> None:
        """Test encode_radar_command encodes command to expected bytes."""
        from tankpit_bot.bot import encode_radar_command, make_radar_command
        from tankpit_bot.protocol.commands import CMD_RADAR, build_query_command

        cmd = make_radar_command()
        result = encode_radar_command(cmd)
        expected = build_query_command(CMD_RADAR)
        assert result == expected

    def test_encode_pickup_move_command(self) -> None:
        """Test encode_pickup_move_command encodes command to expected bytes."""
        from tankpit_bot.bot import encode_pickup_move_command, make_pickup_move_command
        from tankpit_bot.protocol.commands import build_pickup_command

        cmd = make_pickup_move_command(200, 100)
        result = encode_pickup_move_command(cmd)
        expected = build_pickup_command(200, 100)
        assert result == expected

    def test_encode_teleport_command(self) -> None:
        """Test encode_teleport_command encodes command to expected bytes."""
        from tankpit_bot.bot import encode_teleport_command, make_teleport_command
        from tankpit_bot.protocol.commands import build_teleport_command

        cmd = make_teleport_command(128, 64)
        result = encode_teleport_command(cmd)
        expected = build_teleport_command(128, 64)
        assert result == expected


# =============================================================================
# Bot State Tests
# =============================================================================


class TestBotStates:
    """Tests for bot state machine functions."""

    def test_bot_state_enum_values(self) -> None:
        """Test BotState enum has expected states."""
        from tankpit_bot.bot import BotState

        # Verify key states have auto-generated int values
        assert BotState.INITIALIZING.value == 1
        assert BotState.WAITING_FOR_POSITION.value == 2
        assert BotState.IDLE.value == 3
        assert BotState.SCANNING.value == 4
        assert BotState.MOVING.value == 5

    def test_make_initial_state_data(self) -> None:
        """Test make_initial_state_data creates proper state dict."""
        from tankpit_bot.bot import make_initial_state_data

        state = make_initial_state_data()
        assert state["state"] == "INITIALIZING"
        assert state["target_x"] == 0
        assert state["target_y"] == 0
        assert state["fuel_threshold"] == 200

    def test_is_valid_transition_valid(self) -> None:
        """Test is_valid_transition returns True for valid transitions."""
        from tankpit_bot.bot import is_valid_transition

        # INITIALIZING -> WAITING_FOR_POSITION is valid
        assert is_valid_transition("INITIALIZING", "WAITING_FOR_POSITION")

    def test_is_valid_transition_invalid(self) -> None:
        """Test is_valid_transition returns False for invalid transitions."""
        from tankpit_bot.bot import is_valid_transition

        # IDLE -> INITIALIZING is not valid
        assert not is_valid_transition("IDLE", "INITIALIZING")

    def test_validate_transition_valid(self) -> None:
        """Test validate_transition does not raise for valid transitions."""
        from tankpit_bot.bot import validate_transition

        # Should not raise
        validate_transition("INITIALIZING", "WAITING_FOR_POSITION")

    def test_validate_transition_invalid(self) -> None:
        """Test validate_transition raises for invalid transitions."""
        from tankpit_bot.bot import validate_transition

        with pytest.raises(ValueError, match="Invalid transition"):
            validate_transition("IDLE", "INITIALIZING")

    def test_transition_to(self) -> None:
        """Test transition_to updates state."""
        from tankpit_bot.bot import make_initial_state_data, transition_to

        state = make_initial_state_data()
        new_state = transition_to(state, "WAITING_FOR_POSITION")
        assert new_state["state"] == "WAITING_FOR_POSITION"

    def test_set_target(self) -> None:
        """Test set_target updates target coordinates."""
        from tankpit_bot.bot import make_initial_state_data, set_target

        state = make_initial_state_data()
        new_state = set_target(state, 100, 150)
        assert new_state["target_x"] == 100
        assert new_state["target_y"] == 150

    def test_set_fuel_threshold(self) -> None:
        """Test set_fuel_threshold updates fuel threshold."""
        from tankpit_bot.bot import make_initial_state_data, set_fuel_threshold

        state = make_initial_state_data()
        new_state = set_fuel_threshold(state, 300)
        assert new_state["fuel_threshold"] == 300


# =============================================================================
# Bot Class Tests
# =============================================================================


class TestBotClass:
    """Tests for Bot class methods."""

    def test_bot_init(self, fake_env: FakeEnv) -> None:
        """Test Bot.__init__ sets up state correctly."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        assert bot.get_state() == "INITIALIZING"
        assert bot._cdp is None
        assert bot._page is None
        assert bot._equipment_enabled == [False, False, False, False, False]
        assert bot._map_is_open is False

    def test_bot_get_state(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_state returns current state name."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        state = bot.get_state()
        assert state == "INITIALIZING"

    def test_bot_get_state_data(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_state_data returns full state dict."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        state_data = bot.get_state_data()
        assert state_data["state"] == "INITIALIZING"
        assert state_data["target_x"] == 0
        assert state_data["target_y"] == 0
        assert state_data["fuel_threshold"] == 200

    def test_bot_get_world_state(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_world_state returns world state from module."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        world = bot.get_world_state()
        assert world["self_state"] is None
        assert world["containers"] == {}

    def test_bot_get_self_state_none(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_self_state returns None when not tracked."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        self_state = bot.get_self_state()
        assert self_state is None

    def test_bot_get_fuel_when_no_self_state(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_fuel returns 0 when self_state not tracked."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        fuel = bot.get_fuel()
        assert fuel == 0

    def test_bot_get_position_none(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_position returns None when not tracked."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        pos = bot.get_position()
        assert pos is None

    def test_bot_get_containers_empty(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_containers returns empty dict when none tracked."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        containers = bot.get_containers()
        assert containers == {}

    def test_bot_get_fuel_containers_empty(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_fuel_containers returns empty list when none tracked."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        fuel_containers = bot.get_fuel_containers()
        assert fuel_containers == []

    def test_bot_get_nearest_fuel_container_no_position(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_nearest_fuel_container returns None when no position."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        container = bot.get_nearest_fuel_container()
        assert container is None


class TestBotCommandsWithoutCDP:
    """Tests for Bot command methods when CDP session is not available."""

    def test_move_to_returns_false_without_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.move_to returns False when CDP not available."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.move_to(100, 100)
        assert result is False

    def test_pickup_move_to_returns_false_without_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.pickup_move_to returns False when CDP not available."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.pickup_move_to(100, 100)
        assert result is False

    def test_teleport_to_returns_false_without_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.teleport_to returns False when CDP not available."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.teleport_to(100, 100)
        assert result is False

    def test_shoot_at_returns_false_without_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.shoot_at returns False when CDP not available."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.shoot_at(100, 100)
        assert result is False

    def test_use_radar_returns_false_without_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.use_radar returns False when CDP not available."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.use_radar()
        assert result is False

    def test_toggle_equipment_returns_false_without_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.toggle_equipment returns False when CDP not available."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.toggle_equipment(1)
        assert result is False

    def test_toggle_equipment_invalid_slot(self, fake_env: FakeEnv) -> None:
        """Test Bot.toggle_equipment returns False for invalid slot."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.toggle_equipment(0)
        assert result is False
        result = bot.toggle_equipment(6)
        assert result is False

    def test_enable_equipment_invalid_slot(self, fake_env: FakeEnv) -> None:
        """Test Bot.enable_equipment returns False for invalid slot."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.enable_equipment(0)
        assert result is False
        result = bot.enable_equipment(6)
        assert result is False

    def test_enable_homing_returns_false_without_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.enable_homing returns False when CDP not available."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.enable_homing()
        assert result is False

    def test_enable_dual_returns_false_without_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.enable_dual returns False when CDP not available."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.enable_dual()
        assert result is False

    def test_enable_radar_equipment_returns_false_without_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.enable_radar_equipment returns False when CDP not available."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.enable_radar_equipment()
        assert result is False

    def test_open_map_returns_false_without_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.open_map returns False when CDP not available."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.open_map()
        assert result is False

    def test_close_map_returns_true_when_already_closed(self, fake_env: FakeEnv) -> None:
        """Test Bot.close_map returns True when map already closed."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.close_map()
        assert result is True

    def test_go_to_nearest_fuel_returns_false_when_no_fuel(self, fake_env: FakeEnv) -> None:
        """Test Bot.go_to_nearest_fuel returns False when no fuel containers."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.go_to_nearest_fuel()
        assert result is False

    def test_scan_and_collect_fuel_scans_when_no_containers(self, fake_env: FakeEnv) -> None:
        """Test Bot.scan_and_collect_fuel calls radar when no containers."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        # Should attempt radar scan (returns False because no CDP)
        result = bot.scan_and_collect_fuel()
        assert result is False


class TestBotEquipmentState:
    """Tests for Bot equipment state management."""

    def test_is_equipment_enabled_false_by_default(self, fake_env: FakeEnv) -> None:
        """Test Bot.is_equipment_enabled returns False by default."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        for slot in range(1, 6):
            assert bot.is_equipment_enabled(slot) is False

    def test_is_equipment_enabled_invalid_slot(self, fake_env: FakeEnv) -> None:
        """Test Bot.is_equipment_enabled returns False for invalid slot."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        assert bot.is_equipment_enabled(0) is False
        assert bot.is_equipment_enabled(6) is False

    def test_update_equipment_state(self, fake_env: FakeEnv) -> None:
        """Test Bot.update_equipment_state updates equipment list."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot.update_equipment_state([True, False, True, False, True])
        assert bot.is_equipment_enabled(1) is True
        assert bot.is_equipment_enabled(2) is False
        assert bot.is_equipment_enabled(3) is True
        assert bot.is_equipment_enabled(4) is False
        assert bot.is_equipment_enabled(5) is True

    def test_update_equipment_state_wrong_length(self, fake_env: FakeEnv) -> None:
        """Test Bot.update_equipment_state ignores wrong length list."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot.update_equipment_state([True, True, True])  # Wrong length
        # Should remain unchanged
        for slot in range(1, 6):
            assert bot.is_equipment_enabled(slot) is False

    def test_enable_equipment_already_enabled(self, fake_env: FakeEnv) -> None:
        """Test Bot.enable_equipment returns True if already enabled."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot.update_equipment_state([True, False, False, False, False])
        result = bot.enable_equipment(1)
        assert result is True


# =============================================================================
# Bot with Populated World State Tests
# =============================================================================


class TestBotWithWorldState:
    """Tests for Bot methods that work with populated world state."""

    def test_get_position_with_self_state(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_position returns position when tracked."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        update_world_state_from_position(100, 150)
        bot = Bot("https://test.tankpit.com/", headless=True)
        pos = bot.get_position()
        assert pos == (100, 150)

    def test_get_self_state_with_position(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_self_state returns state when tracked."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        update_world_state_from_position(50, 75)
        bot = Bot("https://test.tankpit.com/", headless=True)
        self_state = bot.get_self_state()
        # Type guard: fail test if None
        if self_state is None:
            raise AssertionError("Expected self_state to be populated")
        assert self_state["x"] == 50
        assert self_state["y"] == 75

    def test_get_fuel_with_self_state(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_fuel returns fuel when tracked."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_fuel_change,
            update_world_state_from_position,
        )

        reset_world_state()
        update_world_state_from_position(50, 75)
        update_world_state_from_fuel_change(500)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fuel = bot.get_fuel()
        # Initial fuel is 1000, plus 500
        assert fuel == 1500

    def test_get_fuel_containers_with_containers(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_fuel_containers returns fuel containers."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.container import RadarContainerDict
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_radar,
        )

        reset_world_state()
        containers: list[RadarContainerDict] = [
            RadarContainerDict(x=10, y=20, volume=100),
            RadarContainerDict(x=30, y=40, volume=200),
        ]
        update_world_state_from_radar(containers, [])
        bot = Bot("https://test.tankpit.com/", headless=True)
        fuel_containers = bot.get_fuel_containers()
        assert len(fuel_containers) == 2

    def test_get_nearest_fuel_container_with_containers(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_nearest_fuel_container returns nearest container."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.container import RadarContainerDict
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
            update_world_state_from_radar,
        )

        reset_world_state()
        update_world_state_from_position(50, 50)
        containers: list[RadarContainerDict] = [
            RadarContainerDict(x=10, y=10, volume=100),  # Distance: 80
            RadarContainerDict(x=60, y=60, volume=200),  # Distance: 20
            RadarContainerDict(x=100, y=100, volume=300),  # Distance: 100
        ]
        update_world_state_from_radar(containers, [])
        bot = Bot("https://test.tankpit.com/", headless=True)
        nearest = bot.get_nearest_fuel_container()
        # Type guard: fail test if None
        if nearest is None:
            raise AssertionError("Expected nearest container to be found")
        assert nearest["x"] == 60
        assert nearest["y"] == 60

    def test_get_nearest_fuel_container_no_fuel_containers(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_nearest_fuel_container returns None when no fuel containers."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True)
        nearest = bot.get_nearest_fuel_container()
        assert nearest is None

    def test_scan_and_collect_fuel_moves_when_containers_exist(self, fake_env: FakeEnv) -> None:
        """Test Bot.scan_and_collect_fuel moves when containers known."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.container import RadarContainerDict
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
            update_world_state_from_radar,
        )

        reset_world_state()
        update_world_state_from_position(50, 50)
        containers: list[RadarContainerDict] = [
            RadarContainerDict(x=60, y=60, volume=100),
        ]
        update_world_state_from_radar(containers, [])
        bot = Bot("https://test.tankpit.com/", headless=True)
        # Should attempt to move to container (returns False because no CDP)
        result = bot.scan_and_collect_fuel()
        assert result is False

    def test_go_to_nearest_fuel_with_containers(self, fake_env: FakeEnv) -> None:
        """Test Bot.go_to_nearest_fuel returns False (no CDP) but logs correctly."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.container import RadarContainerDict
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
            update_world_state_from_radar,
        )

        reset_world_state()
        update_world_state_from_position(50, 50)
        containers: list[RadarContainerDict] = [
            RadarContainerDict(x=60, y=60, volume=100),
        ]
        update_world_state_from_radar(containers, [])
        bot = Bot("https://test.tankpit.com/", headless=True)
        # Should attempt to move but return False (no CDP)
        result = bot.go_to_nearest_fuel()
        assert result is False


class TestBotMapState:
    """Tests for Bot map open/close state tracking."""

    def test_open_map_when_already_open(self, fake_env: FakeEnv) -> None:
        """Test Bot.open_map returns True when map already open."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._map_is_open = True
        result = bot.open_map()
        assert result is True

    def test_close_map_returns_false_without_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.close_map returns False when map open but no CDP."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._map_is_open = True
        result = bot.close_map()
        # close_map returns False when CDP unavailable and map is open
        assert result is False


# =============================================================================
# Bot with Mocked CDP Session Tests
# =============================================================================


class TestBotWithCDP:
    """Tests for Bot command methods with mocked CDP session."""

    def test_send_bytes_success(self, fake_env: FakeEnv) -> None:
        """Test Bot._send_bytes succeeds with CDP session."""
        from tankpit_bot.bot import Bot
        from tests.fakes import FakeCDPSession

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        result = bot._send_bytes(b"test", "test_cmd")
        assert result is True

    def test_move_to_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.move_to succeeds with CDP session."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        result = bot.move_to(100, 100)
        assert result is True
        assert bot.get_state() == "MOVING"

    def test_pickup_move_to_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.pickup_move_to succeeds with CDP session."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        result = bot.pickup_move_to(100, 100)
        assert result is True
        assert bot.get_state() == "COLLECTING"

    def test_teleport_to_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.teleport_to succeeds with CDP session."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession, FakePage

        reset_world_state()
        update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._page = FakePage(fake_cdp)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        result = bot.teleport_to(200, 200)
        assert result is True
        assert bot.get_state() == "MOVING"

    def test_shoot_at_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.shoot_at succeeds with CDP session."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        result = bot.shoot_at(100, 100)
        assert result is True
        assert bot.get_state() == "COMBAT"

    def test_shoot_at_already_combat(self, fake_env: FakeEnv) -> None:
        """Test Bot.shoot_at stays in COMBAT if already in COMBAT."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "COMBAT"
        result = bot.shoot_at(100, 100)
        assert result is True
        assert bot.get_state() == "COMBAT"

    def test_use_radar_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.use_radar succeeds with CDP session."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        result = bot.use_radar()
        assert result is True
        assert bot.get_state() == "SCANNING"

    def test_open_map_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.open_map succeeds with CDP session."""
        from tankpit_bot.bot import Bot
        from tests.fakes import FakeCDPSession

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        result = bot.open_map()
        assert result is True
        assert bot._map_is_open is True

    def test_close_map_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.close_map succeeds with CDP session."""
        from tankpit_bot.bot import Bot
        from tests.fakes import FakeCDPSession

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._map_is_open = True
        result = bot.close_map()
        assert result is True
        assert bot._map_is_open is False

    def test_toggle_equipment_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.toggle_equipment succeeds with CDP session."""
        from tankpit_bot.bot import Bot
        from tests.fakes import FakeCDPSession

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        result = bot.toggle_equipment(1)
        assert result is True

    def test_teleport_fails_if_send_fails(self, fake_env: FakeEnv) -> None:
        """Test teleport_to returns False if _send_bytes fails."""
        from tankpit_bot.bot import Bot
        from tests.fakes import FakeCDPSession, FakePage

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._page = FakePage(fake_cdp)
        bot._map_is_open = True  # Skip open_map
        # Remove CDP to make _send_bytes fail
        bot._cdp = None
        result = bot.teleport_to(100, 100)
        assert result is False


# =============================================================================
# Bot State Machine Update Tests
# =============================================================================


class TestBotStateUpdates:
    """Tests for Bot._update_state_from_world state transitions."""

    def test_update_state_initializing_to_waiting(self, fake_env: FakeEnv) -> None:
        """Test transition from INITIALIZING to WAITING_FOR_POSITION when magic set."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        assert bot.get_state() == "INITIALIZING"
        bot._magic = "test_magic_key"
        bot._update_state_from_world()
        assert bot.get_state() == "WAITING_FOR_POSITION"

    def test_update_state_waiting_to_idle(self, fake_env: FakeEnv) -> None:
        """Test transition from WAITING_FOR_POSITION to IDLE when position known."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()  # -> WAITING_FOR_POSITION
        assert bot.get_state() == "WAITING_FOR_POSITION"
        update_world_state_from_position(50, 50)
        bot._update_state_from_world()  # -> IDLE
        assert bot.get_state() == "IDLE"

    def test_update_state_low_fuel(self, fake_env: FakeEnv) -> None:
        """Test transition to LOW_FUEL when fuel below threshold."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()  # -> WAITING_FOR_POSITION
        update_world_state_from_position(50, 50)
        bot._update_state_from_world()  # -> IDLE
        assert bot.get_state() == "IDLE"
        # Set fuel threshold higher than current fuel (1000 default)
        bot._state_data = bot._state_data.copy()
        bot._state_data["fuel_threshold"] = 2000
        bot._update_state_from_world()  # -> LOW_FUEL
        assert bot.get_state() == "LOW_FUEL"

    def test_update_state_scanning_to_idle(self, fake_env: FakeEnv) -> None:
        """Test transition from SCANNING to IDLE when containers found."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.container import RadarContainerDict
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
            update_world_state_from_radar,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()  # -> WAITING_FOR_POSITION
        update_world_state_from_position(50, 50)
        bot._update_state_from_world()  # -> IDLE
        # Transition to SCANNING manually
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "SCANNING"
        bot._state_data["scan_pending"] = True
        # Add containers via radar
        containers: list[RadarContainerDict] = [
            RadarContainerDict(x=100, y=100, volume=50),
        ]
        update_world_state_from_radar(containers, [])
        bot._update_state_from_world()  # -> IDLE
        assert bot.get_state() == "IDLE"
        assert bot._state_data["scan_pending"] is False

    def test_update_state_moving_to_idle_at_target(self, fake_env: FakeEnv) -> None:
        """Test transition from MOVING to IDLE when reaching target."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()  # -> WAITING_FOR_POSITION
        update_world_state_from_position(50, 50)
        bot._update_state_from_world()  # -> IDLE
        # Transition to MOVING with target
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "MOVING"
        bot._state_data["target_x"] = 50
        bot._state_data["target_y"] = 50
        bot._update_state_from_world()  # -> IDLE (at target)
        assert bot.get_state() == "IDLE"

    def test_update_state_collecting_to_idle_at_target(self, fake_env: FakeEnv) -> None:
        """Test transition from COLLECTING to IDLE when reaching target."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()  # -> WAITING_FOR_POSITION
        update_world_state_from_position(100, 100)
        bot._update_state_from_world()  # -> IDLE
        # Transition to COLLECTING with target
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "COLLECTING"
        bot._state_data["target_x"] = 100
        bot._state_data["target_y"] = 100
        bot._update_state_from_world()  # -> IDLE (at target)
        assert bot.get_state() == "IDLE"


class TestBotOnMessageCaptured:
    """Tests for Bot._on_message_captured method."""

    def test_on_message_captured_updates_state(self, fake_env: FakeEnv) -> None:
        """Test _on_message_captured calls parent and updates state."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.types import CapturedMessage

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"  # Set magic so state can transition
        msg = CapturedMessage(
            direction="received",
            payload="test",
            timestamp_ms=1000,
            ws_url="wss://test.tankpit.com/ws",
        )
        bot._on_message_captured(msg)
        # Should have transitioned from INITIALIZING to WAITING_FOR_POSITION
        assert bot.get_state() == "WAITING_FOR_POSITION"


class TestBotStateUpdateBranches:
    """Tests for Bot._update_state_from_world branch coverage."""

    def test_update_state_moving_not_at_target(self, fake_env: FakeEnv) -> None:
        """Test MOVING state stays MOVING when not at target."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()  # -> WAITING_FOR_POSITION
        update_world_state_from_position(50, 50)
        bot._update_state_from_world()  # -> IDLE
        # Set MOVING with different target
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "MOVING"
        bot._state_data["target_x"] = 100  # Not at 50
        bot._state_data["target_y"] = 100  # Not at 50
        bot._update_state_from_world()  # stays MOVING (not at target)
        assert bot.get_state() == "MOVING"


class TestBotTeleportBranches:
    """Tests for Bot.teleport_to branch coverage."""

    def test_teleport_without_page(self, fake_env: FakeEnv) -> None:
        """Test teleport_to works when _page is None (skips waits)."""
        from tankpit_bot.bot import Bot
        from tests.fakes import FakeCDPSession

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._page = None  # No page - skips wait_for_timeout calls
        bot._map_is_open = False
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        result = bot.teleport_to(100, 100)
        assert result is True
        assert bot.get_state() == "MOVING"


class TestBotIdleAndLowFuelHandlers:
    """Tests for _handle_idle_state and _handle_low_fuel_state."""

    def test_handle_idle_state_no_containers(self, fake_env: FakeEnv) -> None:
        """Test _handle_idle_state scans when no fuel containers."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state
        from tests.fakes import FakeCDPSession

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        bot._handle_idle_state()
        # Should have called use_radar (transition to SCANNING)
        assert bot.get_state() == "SCANNING"

    def test_handle_idle_state_with_containers(self, fake_env: FakeEnv) -> None:
        """Test _handle_idle_state moves when fuel containers exist."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.container import RadarContainerDict
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
            update_world_state_from_radar,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(50, 50)
        containers: list[RadarContainerDict] = [
            RadarContainerDict(x=100, y=100, volume=50),
        ]
        update_world_state_from_radar(containers, [])
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        bot._handle_idle_state()
        # Should have called go_to_nearest_fuel (transition to COLLECTING)
        assert bot.get_state() == "COLLECTING"

    def test_handle_low_fuel_state_no_containers(self, fake_env: FakeEnv) -> None:
        """Test _handle_low_fuel_state scans when no fuel containers."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state
        from tests.fakes import FakeCDPSession

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "LOW_FUEL"
        bot._handle_low_fuel_state()
        # Should have called use_radar (transition to SCANNING)
        assert bot.get_state() == "SCANNING"

    def test_handle_low_fuel_state_with_containers(self, fake_env: FakeEnv) -> None:
        """Test _handle_low_fuel_state moves when fuel containers exist."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.container import RadarContainerDict
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
            update_world_state_from_radar,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(50, 50)
        containers: list[RadarContainerDict] = [
            RadarContainerDict(x=100, y=100, volume=50),
        ]
        update_world_state_from_radar(containers, [])
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "LOW_FUEL"
        bot._handle_low_fuel_state()
        # Should have called go_to_nearest_fuel (transition to COLLECTING)
        assert bot.get_state() == "COLLECTING"


# =============================================================================
# Bot Run and Main Tests
# =============================================================================


class TestBotGameLoop:
    """Tests for Bot._game_loop method."""

    def test_game_loop_exits_on_keyboard_interrupt(self, fake_env: FakeEnv) -> None:
        """Test _game_loop exits cleanly on KeyboardInterrupt."""
        from tankpit_bot.bot.base import Bot
        from tests.fakes import FakeCDPSession, FakePageInterrupting

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp

        # Use the FakePageInterrupting from fakes.py that raises after 1 wait
        interrupting_page = FakePageInterrupting(interrupt_after=1)

        # _game_loop will exit when KeyboardInterrupt is raised
        with pytest.raises(KeyboardInterrupt):
            bot._game_loop(interrupting_page)


class TestBotRunMethod:
    """Tests for Bot.run method."""

    def test_run_raises_without_playwright(self, fake_env: FakeEnv) -> None:
        """Test run() raises PlaywrightNotInstalledError when not available."""
        from tankpit_bot import _test_hooks
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.browser import PlaywrightNotInstalledError

        # Save original and set to None
        original = _test_hooks.sync_playwright
        _test_hooks.sync_playwright = None
        try:
            bot = Bot("https://test.tankpit.com/", headless=True)
            with pytest.raises(PlaywrightNotInstalledError):
                bot.run()
        finally:
            _test_hooks.sync_playwright = original

    def test_run_success_path(self, fake_env: FakeEnv) -> None:
        """Test run() goes through the success path and handles KeyboardInterrupt.

        This covers lines 603-614 in run(). The run() method catches
        KeyboardInterrupt internally and returns normally after cleanup.
        """
        from tankpit_bot import _test_hooks
        from tankpit_bot.bot.base import Bot
        from tests.fakes import fake_sync_playwright_bot

        # Set up the fake Playwright that will raise KeyboardInterrupt in game loop
        original = _test_hooks.sync_playwright
        _test_hooks.sync_playwright = fake_sync_playwright_bot

        try:
            bot = Bot("https://test.tankpit.com/", headless=True)
            # run() catches KeyboardInterrupt internally and returns normally
            bot.run()
            # After cleanup, _cdp and _page should be None
            assert bot._cdp is None
            assert bot._page is None
        finally:
            _test_hooks.sync_playwright = original


class TestBotBaseMain:
    """Tests for bot.base.main function."""

    def test_main_creates_and_runs_bot(self, fake_env: FakeEnv) -> None:
        """Test main() creates Bot and calls run()."""
        from tankpit_bot import _test_hooks
        from tankpit_bot._test_hooks import SyncPlaywrightContextManagerProtocol
        from tests.fakes import FakeSyncPlaywrightContextManagerBot

        # Track if sync_playwright factory was called
        factory_called = False

        def fake_sync_playwright_factory() -> SyncPlaywrightContextManagerProtocol:
            """Return fake sync_playwright that exits via KeyboardInterrupt."""
            nonlocal factory_called
            factory_called = True
            return FakeSyncPlaywrightContextManagerBot(interrupt_after=2)

        # Set up fakes
        original_pw = _test_hooks.sync_playwright
        _test_hooks.sync_playwright = fake_sync_playwright_factory

        try:
            from tankpit_bot.bot import base

            with pytest.raises(KeyboardInterrupt):
                base.main()
        finally:
            _test_hooks.sync_playwright = original_pw

        if not factory_called:
            raise AssertionError("Expected sync_playwright factory to be called")

    def test_main_sets_sync_playwright_when_none(self, fake_env: FakeEnv) -> None:
        """Test main() sets sync_playwright when it is None.

        This covers line 672 where sync_playwright is set from get_sync_playwright().
        """
        from tankpit_bot import _test_hooks
        from tankpit_bot._test_hooks import SyncPlaywrightFactoryProtocol
        from tests.fakes import FakeSyncPlaywrightContextManagerBot

        # Save originals
        original_pw = _test_hooks.sync_playwright
        original_get_pw = _test_hooks.get_sync_playwright

        # Set sync_playwright to None so main() will call get_sync_playwright()
        _test_hooks.sync_playwright = None

        # Track if get_sync_playwright was called
        get_called = False

        def fake_get_sync_playwright() -> SyncPlaywrightFactoryProtocol:
            """Fake get_sync_playwright that returns our test factory."""
            nonlocal get_called
            get_called = True

            def factory() -> FakeSyncPlaywrightContextManagerBot:
                return FakeSyncPlaywrightContextManagerBot(interrupt_after=2)

            return factory

        _test_hooks.get_sync_playwright = fake_get_sync_playwright

        try:
            from tankpit_bot.bot import base

            with pytest.raises(KeyboardInterrupt):
                base.main()
        finally:
            _test_hooks.sync_playwright = original_pw
            _test_hooks.get_sync_playwright = original_get_pw

        if not get_called:
            raise AssertionError("Expected get_sync_playwright to be called")


class TestBotGameLoopStates:
    """Tests for Bot._game_loop state handling."""

    def test_game_loop_handles_idle_state(self, fake_env: FakeEnv) -> None:
        """Test _game_loop calls _handle_idle_state when in IDLE state."""
        from tankpit_bot.bot.base import Bot
        from tests.fakes import FakeCDPSession, FakePageInterrupting

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"

        # Use page that interrupts after 3 waits to allow state handler to run
        interrupting_page = FakePageInterrupting(interrupt_after=3)

        with pytest.raises(KeyboardInterrupt):
            bot._game_loop(interrupting_page)

    def test_game_loop_handles_low_fuel_state(self, fake_env: FakeEnv) -> None:
        """Test _game_loop calls _handle_low_fuel_state when in LOW_FUEL state."""
        from tankpit_bot.bot.base import Bot
        from tests.fakes import FakeCDPSession, FakePageInterrupting

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "LOW_FUEL"

        # Use page that interrupts after 3 waits to allow state handler to run
        interrupting_page = FakePageInterrupting(interrupt_after=3)

        with pytest.raises(KeyboardInterrupt):
            bot._game_loop(interrupting_page)
