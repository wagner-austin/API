"""Tests for bot state machine functions and transitions."""

from __future__ import annotations

import pytest

from tests.conftest import FakeEnv


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
            update_world_state_from_fuel_total,
            update_world_state_from_position,
            update_world_state_from_radar,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()  # -> WAITING_FOR_POSITION
        update_world_state_from_position(50, 50)
        update_world_state_from_fuel_total(1400)
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
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()  # -> WAITING_FOR_POSITION
        update_world_state_from_position(50, 50)
        update_world_state_from_fuel_total(1400)
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
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()  # -> WAITING_FOR_POSITION
        update_world_state_from_position(100, 100)
        update_world_state_from_fuel_total(1400)
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
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()  # -> WAITING_FOR_POSITION
        update_world_state_from_position(50, 50)
        update_world_state_from_fuel_total(1400)
        bot._update_state_from_world()  # -> IDLE
        # Set MOVING with different target
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "MOVING"
        bot._state_data["target_x"] = 100  # Not at 50
        bot._state_data["target_y"] = 100  # Not at 50
        bot._update_state_from_world()  # stays MOVING (not at target)
        assert bot.get_state() == "MOVING"
