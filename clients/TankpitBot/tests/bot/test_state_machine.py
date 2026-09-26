"""Tests for the bot HFSM states and transitions.

``test_state_machine.py`` was 606 lines; the state-update detail is now
a sibling.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.bot.states import (
    ActionKind,
    ActionOutcome,
    BotState,
    make_in_flight_action,
    make_no_action,
)
from tankpit_bot.sniffer.world_state_containers import (
    update_world_state_from_fuel_total as _sm_update_fuel,
)
from tankpit_bot.sniffer.world_state_radar import (
    update_world_state_from_radar,
)
from tankpit_bot.types.literals import MessageDirection
from tests.bot._state_machine_fixtures import _set_bot_action
from tests.conftest import FakeEnv


class TestBotStateUpdates:
    """Tests for Bot._update_state_from_world state transitions."""

    def test_update_state_initializing_to_waiting(self, fake_env: FakeEnv) -> None:
        """Test transition from INITIALIZING to WAITING_FOR_POSITION."""
        from tankpit_bot.bot.base import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        assert bot.get_state() is BotState.INITIALIZING
        bot._magic = "test_magic_key"
        bot._update_state_from_world()
        assert bot.get_state() is BotState.WAITING_FOR_POSITION

    def test_update_state_waiting_to_idle(self, fake_env: FakeEnv) -> None:
        """Test transition from WAITING_FOR_POSITION to IDLE."""
        from tankpit_bot.bot.base import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot.world.update_world_state_from_position(50, 50)
        bot._update_state_from_world()
        assert bot.get_state() is BotState.IDLE

    def test_update_state_low_fuel(self, fake_env: FakeEnv) -> None:
        """Test transition to LOW_FUEL when fuel below threshold."""
        from tankpit_bot.bot.base import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot.world.update_world_state_from_position(50, 50)
        bot._update_state_from_world()
        assert bot.get_state() is BotState.IDLE
        bot._state_data = bot._state_data.copy()
        bot._state_data["fuel_threshold"] = 2000
        bot._update_state_from_world()
        assert bot.get_state() is BotState.LOW_FUEL

    def test_update_state_scanning_to_idle(self, fake_env: FakeEnv) -> None:
        """SCANNING completes when a radar response arrives."""
        from tankpit_bot.bot.base import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot.world.update_world_state_from_position(50, 50)
        _sm_update_fuel(bot.world, 1400)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(bot._state_data, BotState.SCANNING, ActionKind.SCAN, 0, 0)
        from tankpit_bot.protocol import RadarContainerDict

        update_world_state_from_radar(
            bot.world, [RadarContainerDict(x=100, y=100, volume=50)], [], []
        )
        bot._update_state_from_world()
        assert bot.get_state() is BotState.IDLE
        assert bot._state_data["in_flight_action"]["kind"] is ActionKind.NONE

    def test_update_state_scanning_to_idle_on_empty_radar(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """SCANNING completes even when the radar finds zero containers."""
        from tankpit_bot.bot.base import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot.world.update_world_state_from_position(50, 50)
        _sm_update_fuel(bot.world, 1400)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(bot._state_data, BotState.SCANNING, ActionKind.SCAN, 0, 0)
        update_world_state_from_radar(bot.world, [], [], [])
        bot._update_state_from_world()
        assert bot.get_state() is BotState.IDLE
        assert bot._state_data["in_flight_action"]["kind"] is ActionKind.NONE

    def test_update_state_moving_to_idle_at_target(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """MOVING completes when reaching target position."""
        from tankpit_bot.bot.base import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot.world.update_world_state_from_position(50, 50)
        _sm_update_fuel(bot.world, 1400)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(bot._state_data, BotState.MOVING, ActionKind.MOVE, 50, 50)
        bot._update_state_from_world()
        assert bot.get_state() is BotState.IDLE

    def test_update_state_collecting_to_idle_at_target(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """COLLECTING completes when reaching target position."""
        from tankpit_bot.bot.base import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot.world.update_world_state_from_position(100, 100)
        _sm_update_fuel(bot.world, 1400)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(
            bot._state_data, BotState.COLLECTING, ActionKind.COLLECT, 100, 100
        )
        bot._update_state_from_world()
        assert bot.get_state() is BotState.IDLE

    def test_update_state_collecting_to_idle_when_target_container_removed(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """COLLECTING completes when pickup removes the target container."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.protocol import RadarContainerDict, RadarMineDict
        from tankpit_bot.sniffer.world_state_containers import (
            update_world_state_from_container_pickup,
        )

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot.world.update_world_state_from_position(206, 83)
        _sm_update_fuel(bot.world, 1100)
        bot._update_state_from_world()
        containers: list[RadarContainerDict] = [
            RadarContainerDict(x=205, y=82, volume=-1),
        ]
        mines: list[RadarMineDict] = []
        update_world_state_from_radar(bot.world, containers, mines, [])
        bot._state_data = _set_bot_action(
            bot._state_data, BotState.COLLECTING, ActionKind.COLLECT, 205, 82
        )
        update_world_state_from_container_pickup(bot.world, 205, 82)
        bot._update_state_from_world()
        assert bot.get_state() is BotState.IDLE


class TestBotStates:
    """Tests for bot state machine functions."""

    def test_bot_state_members_carry_their_names(self) -> None:
        """Every BotState member's word is its own name."""
        assert [state.value for state in BotState] == [state.name for state in BotState]
        assert len(BotState) == 10

    def test_every_state_has_a_transition_row(self) -> None:
        """The transition table covers every state, so a lookup never needs a default."""
        from tankpit_bot.bot.states import VALID_TRANSITIONS

        assert set(VALID_TRANSITIONS) == set(BotState)

    def test_make_initial_state_data(self) -> None:
        """Test make_initial_state_data creates proper state dict."""
        from tankpit_bot.bot.states import make_initial_state_data

        state = make_initial_state_data()
        assert state["state"] is BotState.INITIALIZING
        assert state["fuel_threshold"] == 200
        action = state["in_flight_action"]
        assert action["kind"] is ActionKind.NONE
        assert action["outcome"] is ActionOutcome.CONFIRMED

    def test_is_valid_transition_valid(self) -> None:
        """Test is_valid_transition returns True for valid transitions."""
        from tankpit_bot.bot.states import is_valid_transition

        assert is_valid_transition(BotState.INITIALIZING, BotState.WAITING_FOR_POSITION)

    def test_low_fuel_to_combat_is_valid(self) -> None:
        """LOW_FUEL -> COMBAT is valid for low-fuel defense scenarios."""
        from tankpit_bot.bot.states import is_valid_transition

        assert is_valid_transition(BotState.LOW_FUEL, BotState.COMBAT)

    def test_is_valid_transition_invalid(self) -> None:
        """Test is_valid_transition returns False for invalid transitions."""
        from tankpit_bot.bot.states import is_valid_transition

        assert not is_valid_transition(BotState.IDLE, BotState.INITIALIZING)

    def test_validate_transition_valid(self) -> None:
        """Test validate_transition does not raise for valid transitions."""
        from tankpit_bot.bot.states import validate_transition

        validate_transition(BotState.INITIALIZING, BotState.WAITING_FOR_POSITION)

    def test_validate_transition_invalid(self) -> None:
        """validate_transition names both states and the allowed ones by word."""
        from tankpit_bot.bot.states import validate_transition

        with pytest.raises(
            ValueError,
            match=(
                r"Invalid transition from DISCONNECTED to IDLE\. "
                r"Allowed: \['INITIALIZING'\]"
            ),
        ):
            validate_transition(BotState.DISCONNECTED, BotState.IDLE)

    def test_transition_to(self) -> None:
        """Test transition_to updates state."""
        from tankpit_bot.bot.states import make_initial_state_data, transition_to

        state = make_initial_state_data()
        new_state = transition_to(state, BotState.WAITING_FOR_POSITION)
        assert new_state["state"] is BotState.WAITING_FOR_POSITION

    def test_transition_to_with_action(self) -> None:
        """transition_to replaces in_flight_action when provided."""
        from tankpit_bot.bot.states import make_initial_state_data, transition_to

        state = make_initial_state_data()
        action = make_in_flight_action(ActionKind.MOVE, 10, 20, 5000)
        new_state = transition_to(
            state,
            BotState.WAITING_FOR_POSITION,
            in_flight_action=action,
        )
        assert new_state["in_flight_action"]["kind"] is ActionKind.MOVE
        assert new_state["in_flight_action"]["target_x"] == 10
        assert new_state["in_flight_action"]["target_y"] == 20
        assert new_state["in_flight_action"]["started_ms"] == 5000
        assert new_state["in_flight_action"]["outcome"] is ActionOutcome.PENDING

    def test_transition_to_inherits_action_when_none(self) -> None:
        """transition_to inherits current action when no new one given."""
        from tankpit_bot.bot.states import make_initial_state_data, transition_to

        state = make_initial_state_data()
        action = make_in_flight_action(ActionKind.TELEPORT, 50, 60, 1000)
        state_with_action = transition_to(
            state,
            BotState.WAITING_FOR_POSITION,
            in_flight_action=action,
        )
        inherited = transition_to(state_with_action, BotState.IDLE)
        assert inherited["in_flight_action"]["kind"] is ActionKind.TELEPORT
        assert inherited["in_flight_action"]["target_x"] == 50

    def test_set_fuel_threshold(self) -> None:
        """Test set_fuel_threshold updates fuel threshold."""
        from tankpit_bot.bot.states import make_initial_state_data, set_fuel_threshold

        state = make_initial_state_data()
        new_state = set_fuel_threshold(state, 300)
        assert new_state["fuel_threshold"] == 300

    def test_make_no_action(self) -> None:
        """make_no_action creates a confirmed no-op action."""
        action = make_no_action()
        assert action["kind"] is ActionKind.NONE
        assert action["outcome"] is ActionOutcome.CONFIRMED
        assert action["target_x"] == 0
        assert action["target_y"] == 0
        assert action["started_ms"] == 0

    def test_make_in_flight_action(self) -> None:
        """make_in_flight_action creates a pending action with target."""
        action = make_in_flight_action(ActionKind.COLLECT, 42, 99, 5000)
        assert action["kind"] is ActionKind.COLLECT
        assert action["outcome"] is ActionOutcome.PENDING
        assert action["target_x"] == 42
        assert action["target_y"] == 99
        assert action["started_ms"] == 5000

    def test_encode_decode_in_flight_action_roundtrip(self) -> None:
        """encode then decode produces identical InFlightActionDict."""
        from tankpit_bot.bot.states import (
            decode_in_flight_action,
            encode_in_flight_action,
        )

        original = make_in_flight_action(ActionKind.TELEPORT, 128, 64, 9999)
        encoded = encode_in_flight_action(original)
        assert encoded["kind"] == "teleport"
        assert encoded["outcome"] == "pending"
        decoded = decode_in_flight_action(encoded)
        assert decoded == original
        assert decoded["kind"] is ActionKind.TELEPORT
        assert decoded["outcome"] is ActionOutcome.PENDING

    def test_decode_invalid_action_kind_raises(self) -> None:
        """Decode rejects invalid action kind."""
        from platform_core.json_utils import JSONObject

        from tankpit_bot.bot.states import decode_in_flight_action

        data: JSONObject = {
            "kind": "INVALID",
            "target_x": 0,
            "target_y": 0,
            "started_ms": 0,
            "outcome": "pending",
        }
        with pytest.raises(JSONTypeError, match="Invalid kind 'INVALID'"):
            decode_in_flight_action(data)

    def test_decode_invalid_action_outcome_raises(self) -> None:
        """Decode rejects invalid action outcome."""
        from platform_core.json_utils import JSONObject

        from tankpit_bot.bot.states import decode_in_flight_action

        data: JSONObject = {
            "kind": "move",
            "target_x": 0,
            "target_y": 0,
            "started_ms": 0,
            "outcome": "BOGUS",
        }
        with pytest.raises(JSONTypeError, match="Invalid outcome 'BOGUS'"):
            decode_in_flight_action(data)


class TestBotOnMessageCaptured:
    """Tests for Bot._on_message_captured method."""

    def test_on_message_captured_does_not_decode(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_on_message_captured only extracts magic, no state transition."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.types import CapturedMessage

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        msg = CapturedMessage(
            direction=MessageDirection.RECEIVED,
            payload="test",
            timestamp_ms=1000,
            ws_url="wss://test.tankpit.com/ws",
        )
        bot._on_message_captured(msg)
        assert bot.get_state() is BotState.INITIALIZING


class TestBotStateUpdateBranches:
    """Tests for Bot._update_state_from_world branch coverage."""

    def test_update_state_moving_not_at_target(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """MOVING stays MOVING when not at target."""
        from tankpit_bot.bot.base import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot.world.update_world_state_from_position(50, 50)
        _sm_update_fuel(bot.world, 1400)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(
            bot._state_data, BotState.MOVING, ActionKind.MOVE, 100, 100
        )
        bot._update_state_from_world()
        assert bot.get_state() is BotState.MOVING

    def test_update_state_teleporting_without_landing_stays_teleporting(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """TELEPORTING stays until landing is confirmed."""
        from tankpit_bot.bot.base import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot.world.update_world_state_from_position(196, 85)
        _sm_update_fuel(bot.world, 582)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(
            bot._state_data, BotState.TELEPORTING, ActionKind.TELEPORT, 196, 86
        )
        bot._update_state_from_world()
        assert bot.get_state() is BotState.TELEPORTING
