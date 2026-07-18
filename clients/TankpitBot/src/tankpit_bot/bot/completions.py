"""State machine completions — detect when in-flight actions finish.

Each ``_maybe_complete_*`` method checks world state for an authoritative
server signal (teleport landing, position reached, radar response,
container consumed) and transitions the state machine to IDLE.

Split from bot/base.py for separation of concerns.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.bot.command_service import CommandService
from tankpit_bot.bot.states import (
    BotStateDataDict,
    InFlightActionDict,
    StateName,
    make_initial_state_data,
    make_no_action,
    transition_to,
    validate_transition,
)
from tankpit_bot.browser.cdp_service import CDPService
from tankpit_bot.browser.cdp_utils import get_current_time_ms
from tankpit_bot.browser.session_base import SessionBase
from tankpit_bot.ledger.outcome.collect import (
    emit_collect_container_consumed,
    emit_collect_position_reached,
)
from tankpit_bot.ledger.outcome.move import emit_move_position_reached
from tankpit_bot.ledger.outcome.scan import emit_scan_radar_complete
from tankpit_bot.ledger.outcome.teleport import emit_teleport_landed
from tankpit_bot.runtime_logging import emit_diagnostic, emit_state
from tankpit_bot.sniffer.world_state import (
    check_and_clear_radar_scan_complete,
    get_world_service,
    get_world_state,
    mark_move_target_failed,
)
from tankpit_bot.sniffer.world_state_combat import check_and_clear_teleport_landed
from tankpit_bot.state import SelfStateDict, WorldStateDict

log = get_logger(__name__)


class CompletionsMixin(SessionBase):
    """State machine transitions and completion detection.

    Inherits CDPService composition from SessionBase. Adds the HFSM
    state data, ``_transition`` method, and all ``_maybe_complete_*``
    checks that detect when in-flight actions finish.
    """

    def __init__(
        self,
        target_url: str,
        *,
        headless: bool = False,
        prefer_account: bool = False,
        cdp_service: CDPService | None = None,
        command_service: CommandService | None = None,
    ) -> None:
        """Initialize state machine data and delegate to SessionBase.

        Args:
            target_url: URL to navigate to.
            headless: Whether to run browser in headless mode.
            prefer_account: Whether to prefer account login.
            cdp_service: Injected CDPService. Created internally if None.
            command_service: Injected CommandService. Created internally if None.
        """
        super().__init__(
            target_url,
            headless=headless,
            prefer_account=prefer_account,
            cdp_service=cdp_service,
            command_service=command_service,
        )
        self._state_data: BotStateDataDict = make_initial_state_data()

    def get_state(self) -> StateName:
        """Get current bot state.

        Returns:
            Current state name.
        """
        return self._state_data["state"]

    def get_state_data(self) -> BotStateDataDict:
        """Get full state data.

        Returns:
            Current BotStateDataDict.
        """
        return self._state_data

    def _transition(
        self,
        new_state: StateName,
        *,
        in_flight_action: InFlightActionDict | None = None,
    ) -> None:
        """Transition to a new state with validation.

        Args:
            new_state: State to transition to.
            in_flight_action: New action record. Pass
                make_in_flight_action() for states with an active
                command, or make_no_action() for IDLE. If None,
                inherits the current action.

        Raises:
            ValueError: If transition is invalid.
        """
        current_state = self._state_data["state"]
        validate_transition(current_state, new_state)

        self._state_data = transition_to(
            self._state_data,
            new_state,
            in_flight_action=in_flight_action,
        )
        emit_state("%s -> %s", current_state, new_state)

    def _maybe_transition_from_initializing(self, self_state: SelfStateDict | None) -> bool:
        """Advance startup states when required bootstrap data is available.

        Args:
            self_state: Current self tank state, if known.

        Returns:
            True if a transition was applied.
        """
        current_state = self._state_data["state"]
        if current_state == "INITIALIZING" and self._magic is not None:
            self._transition("WAITING_FOR_POSITION")
            return True
        if current_state == "WAITING_FOR_POSITION" and self_state is not None:
            self._transition("IDLE")
            return True
        return False

    def _maybe_transition_to_low_fuel(self, self_state: SelfStateDict | None) -> bool:
        """Enter LOW_FUEL when the tracked fuel total is below threshold.

        Args:
            self_state: Current self tank state, if known.

        Returns:
            True if a transition was applied.
        """
        current_state = self._state_data["state"]
        excluded_states = (
            "LOW_FUEL",
            "INITIALIZING",
            "WAITING_FOR_POSITION",
            "DISCONNECTED",
            "TELEPORTING",
            "COLLECTING",
            "SCANNING",
        )
        is_low_fuel = (
            self_state is not None
            and current_state not in excluded_states
            and self_state["fuel"] < self._state_data["fuel_threshold"]
        )
        if is_low_fuel:
            self._transition("LOW_FUEL")
            return True
        return False

    def _maybe_complete_scan(self, world: WorldStateDict) -> bool:
        """Finish a pending scan when the server responds.

        Args:
            world: Current world state snapshot.

        Returns:
            True if a transition was applied.
        """
        action = self._state_data["in_flight_action"]
        if action["kind"] != "scan" or action["outcome"] != "pending":
            return False
        if not check_and_clear_radar_scan_complete():
            return False
        emit_scan_radar_complete(
            duration_ms=self._action_duration_ms(action),
            target_x=action["target_x"],
            target_y=action["target_y"],
        )
        self._transition("IDLE", in_flight_action=make_no_action())
        return True

    def _maybe_complete_walk(self, self_state: SelfStateDict | None) -> bool:
        """Finish MOVING once the tank reaches the exact walking target.

        Args:
            self_state: Current self tank state, if known.

        Returns:
            True if a transition was applied.
        """
        if self._state_data["state"] != "MOVING" or self_state is None:
            return False
        action = self._state_data["in_flight_action"]
        tx, ty = action["target_x"], action["target_y"]
        if self_state["x"] == tx and self_state["y"] == ty:
            emit_move_position_reached(
                duration_ms=self._action_duration_ms(action),
                target_x=tx,
                target_y=ty,
                landed_x=self_state["x"],
                landed_y=self_state["y"],
            )
            self._transition("IDLE", in_flight_action=make_no_action())
            return True
        return False

    def _maybe_complete_teleport(self, self_state: SelfStateDict | None) -> bool:
        """Finish TELEPORTING when the server confirms landing.

        Args:
            self_state: Current self tank state, if known.

        Returns:
            True if a transition was applied.
        """
        if (
            self._state_data["state"] == "TELEPORTING"
            and self_state is not None
            and check_and_clear_teleport_landed(get_world_service())
        ):
            action = self._state_data["in_flight_action"]
            tx, ty = action["target_x"], action["target_y"]
            if self_state["x"] != tx or self_state["y"] != ty:
                dist = abs(self_state["x"] - tx) + abs(self_state["y"] - ty)
                world = get_world_service().world_state
                enemy_on_tile = any(t["x"] == tx and t["y"] == ty for t in world["tanks"].values())
                if dist > 1 or not enemy_on_tile:
                    now = get_current_time_ms()
                    mark_move_target_failed(tx, ty, now)
                    emit_diagnostic(
                        diagnostic_kind="teleport_displacement",
                        target_x=tx,
                        target_y=ty,
                        landed_x=self_state["x"],
                        landed_y=self_state["y"],
                        dist=dist,
                    )
            emit_teleport_landed(
                duration_ms=self._action_duration_ms(action),
                target_x=tx,
                target_y=ty,
                landed_x=self_state["x"],
                landed_y=self_state["y"],
                messages=self._messages,
            )
            self._transition("IDLE", in_flight_action=make_no_action())
            return True
        return False

    def _maybe_complete_collection(
        self,
        world: WorldStateDict,
        self_state: SelfStateDict | None,
    ) -> bool:
        """Finish COLLECTING when the pickup target is reached or removed.

        Args:
            world: Current world state snapshot.
            self_state: Current self tank state, if known.

        Returns:
            True if a transition was applied.
        """
        if self._state_data["state"] != "COLLECTING" or self_state is None:
            return False
        action = self._state_data["in_flight_action"]
        tx, ty = action["target_x"], action["target_y"]
        target_key = f"{tx},{ty}"
        position_reached = self_state["x"] == tx and self_state["y"] == ty
        if position_reached or target_key not in world["containers"]:
            if position_reached:
                emit_collect_position_reached(
                    duration_ms=self._action_duration_ms(action),
                    target_x=tx,
                    target_y=ty,
                    landed_x=self_state["x"],
                    landed_y=self_state["y"],
                )
            else:
                emit_collect_container_consumed(
                    duration_ms=self._action_duration_ms(action),
                    target_x=tx,
                    target_y=ty,
                    landed_x=self_state["x"],
                    landed_y=self_state["y"],
                )
            self._transition("IDLE", in_flight_action=make_no_action())
            return True
        return False

    def _action_duration_ms(self, action: InFlightActionDict) -> int:
        """Return dispatch-to-now wall-clock ms for an in-flight action.

        Args:
            action: The in-flight action being resolved.

        Returns:
            Elapsed ms since dispatch, or ``-1`` when the gate fired
            with no recorded dispatch time.
        """
        started_ms = action["started_ms"]
        return get_current_time_ms() - started_ms if started_ms > 0 else -1

    def _update_state_from_world(self) -> None:
        """Update state machine based on current world state.

        Order matters: in-flight action completions (teleport, walk,
        collection, scan) are checked BEFORE low-fuel transitions.
        Otherwise LOW_FUEL would stomp TELEPORTING/COLLECTING states
        and cause repeated command spam.
        """
        world = get_world_state()
        self_state = world["self_state"]

        if self._maybe_transition_from_initializing(self_state):
            return
        if self._maybe_complete_teleport(self_state):
            return
        if self._maybe_complete_walk(self_state):
            return
        if self._maybe_complete_scan(world):
            return
        if self._maybe_complete_collection(world, self_state):
            return
        self._maybe_transition_to_low_fuel(self_state)


__all__ = [
    "CompletionsMixin",
]
