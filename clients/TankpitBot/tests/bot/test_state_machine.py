"""Tests for bot state machine functions and transitions."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.states import (
    ActionKind,
    BotStateDataDict,
    StateName,
    make_in_flight_action,
    make_no_action,
)
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.sniffer.world_state_combat import (
    mark_teleport_landed,
)
from tankpit_bot.sniffer.world_state_containers import (
    update_world_state_from_fuel_total as _sm_update_fuel,
)
from tankpit_bot.sniffer.world_state_radar import (
    update_world_state_from_radar,
)
from tests.conftest import FakeEnv


def _set_bot_action(
    state_data: BotStateDataDict,
    state: StateName,
    kind: ActionKind,
    tx: int,
    ty: int,
    started_ms: int = -1,
) -> BotStateDataDict:
    """Build new state data with state and in-flight action set."""
    from tankpit_bot.browser import get_current_time_ms

    ts = get_current_time_ms() if started_ms < 0 else started_ms
    return BotStateDataDict(
        state=state,
        fuel_threshold=state_data["fuel_threshold"],
        in_flight_action=make_in_flight_action(kind, tx, ty, ts),
    )


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
        assert BotState.TELEPORTING.value == 6

    def test_make_initial_state_data(self) -> None:
        """Test make_initial_state_data creates proper state dict."""
        from tankpit_bot.bot import make_initial_state_data

        state = make_initial_state_data()
        assert state["state"] == "INITIALIZING"
        assert state["fuel_threshold"] == 200
        action = state["in_flight_action"]
        assert action["kind"] == "none"
        assert action["outcome"] == "confirmed"

    def test_is_valid_transition_valid(self) -> None:
        """Test is_valid_transition returns True for valid transitions."""
        from tankpit_bot.bot import is_valid_transition

        assert is_valid_transition("INITIALIZING", "WAITING_FOR_POSITION")

    def test_low_fuel_to_combat_is_valid(self) -> None:
        """LOW_FUEL -> COMBAT is valid for low-fuel defense scenarios."""
        from tankpit_bot.bot import is_valid_transition

        assert is_valid_transition("LOW_FUEL", "COMBAT")

    def test_is_valid_transition_invalid(self) -> None:
        """Test is_valid_transition returns False for invalid transitions."""
        from tankpit_bot.bot import is_valid_transition

        assert not is_valid_transition("IDLE", "INITIALIZING")

    def test_validate_transition_valid(self) -> None:
        """Test validate_transition does not raise for valid transitions."""
        from tankpit_bot.bot import validate_transition

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

    def test_transition_to_with_action(self) -> None:
        """transition_to replaces in_flight_action when provided."""
        from tankpit_bot.bot import make_initial_state_data, transition_to

        state = make_initial_state_data()
        action = make_in_flight_action("move", 10, 20, 5000)
        new_state = transition_to(
            state,
            "WAITING_FOR_POSITION",
            in_flight_action=action,
        )
        assert new_state["in_flight_action"]["kind"] == "move"
        assert new_state["in_flight_action"]["target_x"] == 10
        assert new_state["in_flight_action"]["target_y"] == 20
        assert new_state["in_flight_action"]["started_ms"] == 5000
        assert new_state["in_flight_action"]["outcome"] == "pending"

    def test_transition_to_inherits_action_when_none(self) -> None:
        """transition_to inherits current action when no new one given."""
        from tankpit_bot.bot import make_initial_state_data, transition_to

        state = make_initial_state_data()
        action = make_in_flight_action("teleport", 50, 60, 1000)
        state_with_action = transition_to(
            state,
            "WAITING_FOR_POSITION",
            in_flight_action=action,
        )
        inherited = transition_to(state_with_action, "IDLE")
        assert inherited["in_flight_action"]["kind"] == "teleport"
        assert inherited["in_flight_action"]["target_x"] == 50

    def test_set_fuel_threshold(self) -> None:
        """Test set_fuel_threshold updates fuel threshold."""
        from tankpit_bot.bot import make_initial_state_data, set_fuel_threshold

        state = make_initial_state_data()
        new_state = set_fuel_threshold(state, 300)
        assert new_state["fuel_threshold"] == 300

    def test_make_no_action(self) -> None:
        """make_no_action creates a confirmed no-op action."""
        action = make_no_action()
        assert action["kind"] == "none"
        assert action["outcome"] == "confirmed"
        assert action["target_x"] == 0
        assert action["target_y"] == 0
        assert action["started_ms"] == 0

    def test_make_in_flight_action(self) -> None:
        """make_in_flight_action creates a pending action with target."""
        action = make_in_flight_action("collect", 42, 99, 5000)
        assert action["kind"] == "collect"
        assert action["outcome"] == "pending"
        assert action["target_x"] == 42
        assert action["target_y"] == 99
        assert action["started_ms"] == 5000

    def test_encode_decode_in_flight_action_roundtrip(self) -> None:
        """encode then decode produces identical InFlightActionDict."""
        from tankpit_bot.bot.states import (
            decode_in_flight_action,
            encode_in_flight_action,
        )

        original = make_in_flight_action("teleport", 128, 64, 9999)
        encoded = encode_in_flight_action(original)
        decoded = decode_in_flight_action(encoded)
        assert decoded == original

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
        with pytest.raises(ValueError, match="must be one of"):
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
        with pytest.raises(ValueError, match="must be one of"):
            decode_in_flight_action(data)


class TestBotStateUpdates:
    """Tests for Bot._update_state_from_world state transitions."""

    def test_update_state_initializing_to_waiting(self, fake_env: FakeEnv) -> None:
        """Test transition from INITIALIZING to WAITING_FOR_POSITION."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        assert bot.get_state() == "INITIALIZING"
        bot._magic = "test_magic_key"
        bot._update_state_from_world()
        assert bot.get_state() == "WAITING_FOR_POSITION"

    def test_update_state_waiting_to_idle(self, fake_env: FakeEnv) -> None:
        """Test transition from WAITING_FOR_POSITION to IDLE."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        update_world_state_from_position(50, 50)
        bot._update_state_from_world()
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
        bot._update_state_from_world()
        update_world_state_from_position(50, 50)
        bot._update_state_from_world()
        assert bot.get_state() == "IDLE"
        bot._state_data = bot._state_data.copy()
        bot._state_data["fuel_threshold"] = 2000
        bot._update_state_from_world()
        assert bot.get_state() == "LOW_FUEL"

    def test_update_state_scanning_to_idle(self, fake_env: FakeEnv) -> None:
        """SCANNING completes when a radar response arrives."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        update_world_state_from_position(50, 50)
        _sm_update_fuel(get_world_service(), 1400)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(bot._state_data, "SCANNING", "scan", 0, 0)
        from tankpit_bot.container import RadarContainerDict

        update_world_state_from_radar(
            get_world_service(),
            [RadarContainerDict(x=100, y=100, volume=50)],
            [],
        )
        bot._update_state_from_world()
        assert bot.get_state() == "IDLE"
        assert bot._state_data["in_flight_action"]["kind"] == "none"

    def test_update_state_scanning_to_idle_on_empty_radar(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """SCANNING completes even when the radar finds zero containers."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        update_world_state_from_position(50, 50)
        _sm_update_fuel(get_world_service(), 1400)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(bot._state_data, "SCANNING", "scan", 0, 0)
        update_world_state_from_radar(get_world_service(), [], [])
        bot._update_state_from_world()
        assert bot.get_state() == "IDLE"
        assert bot._state_data["in_flight_action"]["kind"] == "none"

    def test_update_state_moving_to_idle_at_target(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """MOVING completes when reaching target position."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        update_world_state_from_position(50, 50)
        _sm_update_fuel(get_world_service(), 1400)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(bot._state_data, "MOVING", "move", 50, 50)
        bot._update_state_from_world()
        assert bot.get_state() == "IDLE"

    def test_update_state_collecting_to_idle_at_target(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """COLLECTING completes when reaching target position."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        update_world_state_from_position(100, 100)
        _sm_update_fuel(get_world_service(), 1400)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(bot._state_data, "COLLECTING", "collect", 100, 100)
        bot._update_state_from_world()
        assert bot.get_state() == "IDLE"

    def test_update_state_collecting_to_idle_when_target_container_removed(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """COLLECTING completes when pickup removes the target container."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.container import RadarContainerDict, RadarMineDict
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tankpit_bot.sniffer.world_state_containers import (
            update_world_state_from_container_pickup,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        update_world_state_from_position(206, 83)
        _sm_update_fuel(get_world_service(), 1100)
        bot._update_state_from_world()
        containers: list[RadarContainerDict] = [
            RadarContainerDict(x=205, y=82, volume=-1),
        ]
        mines: list[RadarMineDict] = []
        update_world_state_from_radar(get_world_service(), containers, mines)
        bot._state_data = _set_bot_action(bot._state_data, "COLLECTING", "collect", 205, 82)
        update_world_state_from_container_pickup(get_world_service(), 205, 82)
        bot._update_state_from_world()
        assert bot.get_state() == "IDLE"

    def test_update_state_teleporting_to_idle_on_landed(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """TELEPORTING completes when the server confirms landing."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        update_world_state_from_position(196, 85)
        _sm_update_fuel(get_world_service(), 582)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(bot._state_data, "TELEPORTING", "teleport", 196, 86)
        mark_teleport_landed(get_world_service())
        bot._update_state_from_world()
        assert bot.get_state() == "IDLE"

    def test_update_state_teleport_landing_mismatch_marks_failed_target(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """Mismatched teleport landing blacklists the requested destination."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.browser import get_current_time_ms
        from tankpit_bot.sniffer.world_state import (
            is_move_target_failed,
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        update_world_state_from_position(196, 85)
        _sm_update_fuel(get_world_service(), 582)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(bot._state_data, "TELEPORTING", "teleport", 196, 86)

        mark_teleport_landed(get_world_service())
        bot._update_state_from_world()

        now = get_current_time_ms()
        assert bot.get_state() == "IDLE"
        assert is_move_target_failed(196, 86, now) is True

    def test_low_fuel_does_not_stomp_teleporting(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """LOW_FUEL does not override an in-flight TELEPORTING state."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        update_world_state_from_position(50, 50)
        _sm_update_fuel(get_world_service(), 100)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(bot._state_data, "TELEPORTING", "teleport", 60, 70)
        bot._state_data["fuel_threshold"] = 200
        bot._update_state_from_world()
        assert bot.get_state() == "TELEPORTING"

    def test_low_fuel_does_not_stomp_collecting(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """LOW_FUEL does not override an in-flight COLLECTING state."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.container import RadarContainerDict
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        update_world_state_from_position(50, 50)
        _sm_update_fuel(get_world_service(), 100)
        update_world_state_from_radar(
            get_world_service(),
            [RadarContainerDict(x=55, y=55, volume=500)],
            [],
        )
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(bot._state_data, "COLLECTING", "collect", 55, 55)
        bot._state_data["fuel_threshold"] = 200
        bot._update_state_from_world()
        assert bot.get_state() == "COLLECTING"

    def test_low_fuel_does_not_stomp_scanning(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """LOW_FUEL does not override an in-flight SCANNING state."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        update_world_state_from_position(50, 50)
        _sm_update_fuel(get_world_service(), 100)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(bot._state_data, "SCANNING", "scan", 0, 0)
        bot._state_data["fuel_threshold"] = 200
        bot._update_state_from_world()
        assert bot.get_state() == "SCANNING"

    def test_teleport_completes_before_low_fuel_checked(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """TELEPORTING completes to IDLE even when fuel is below threshold."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        update_world_state_from_position(50, 50)
        _sm_update_fuel(get_world_service(), 100)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(bot._state_data, "TELEPORTING", "teleport", 50, 50)
        bot._state_data["fuel_threshold"] = 200
        mark_teleport_landed(get_world_service())
        bot._update_state_from_world()
        assert bot.get_state() == "IDLE"


class TestBotOnMessageCaptured:
    """Tests for Bot._on_message_captured method."""

    def test_on_message_captured_does_not_decode(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_on_message_captured only extracts magic, no state transition."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.types import CapturedMessage

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        msg = CapturedMessage(
            direction="received",
            payload="test",
            timestamp_ms=1000,
            ws_url="wss://test.tankpit.com/ws",
        )
        bot._on_message_captured(msg)
        assert bot.get_state() == "INITIALIZING"


class TestBotStateUpdateBranches:
    """Tests for Bot._update_state_from_world branch coverage."""

    def test_update_state_moving_not_at_target(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """MOVING stays MOVING when not at target."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        update_world_state_from_position(50, 50)
        _sm_update_fuel(get_world_service(), 1400)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(bot._state_data, "MOVING", "move", 100, 100)
        bot._update_state_from_world()
        assert bot.get_state() == "MOVING"

    def test_update_state_teleporting_without_landing_stays_teleporting(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """TELEPORTING stays until landing is confirmed."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        update_world_state_from_position(196, 85)
        _sm_update_fuel(get_world_service(), 582)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(bot._state_data, "TELEPORTING", "teleport", 196, 86)
        bot._update_state_from_world()
        assert bot.get_state() == "TELEPORTING"
