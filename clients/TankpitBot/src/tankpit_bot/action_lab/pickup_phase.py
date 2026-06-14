"""Shared move-and-pickup command-phase helpers for action-lab probes."""

from __future__ import annotations

from typing import Literal, Protocol

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.action_lab.teleport_phase import emit_command_dispatch_failure_diagnostic

_PICKUP_TIMEOUT_PER_TILE_MS = 500
_PICKUP_TIMEOUT_SETTLE_GRACE_MS = 1000
_PICKUP_POLL_INTERVAL_MS = 100.0


class PickupPhaseError(Exception):
    """Raised when a tracked pickup phase cannot proceed."""


class PickupPhaseProbeProtocol(action_session.BufferedWorldStateProviderProtocol, Protocol):
    """Minimal probe interface required to run one tracked pickup phase."""

    def move_to(self, x: int, y: int) -> bool:
        """Dispatch one movement command."""

    def _start_action_phase(
        self,
        phase: Literal["move", "pickup"],
        *,
        attempt_label: str,
    ) -> ActionPhaseCycleDict:
        """Start one action phase cycle."""

    def _end_action_phase(self, cycle: ActionPhaseCycleDict) -> None:
        """Close one active action phase."""

    def _reset_probe_state_to_idle(self) -> None:
        """Reset probe state to idle after the phase settles."""


def effective_pickup_timeout_ms(
    *,
    current_x: int,
    current_y: int,
    target_x: int,
    target_y: int,
    base_timeout_ms: int,
) -> int:
    """Return a pickup timeout sized for the current travel distance.

    Args:
        current_x: Current self X tile.
        current_y: Current self Y tile.
        target_x: Pickup target X tile.
        target_y: Pickup target Y tile.
        base_timeout_ms: Configured minimum pickup timeout.

    Returns:
        Timeout in milliseconds large enough for the move plus pickup settle.
    """
    travel_distance = abs(target_x - current_x) + abs(target_y - current_y)
    distance_budget_ms = (
        travel_distance * _PICKUP_TIMEOUT_PER_TILE_MS
    ) + _PICKUP_TIMEOUT_SETTLE_GRACE_MS
    if distance_budget_ms > base_timeout_ms:
        return distance_budget_ms
    return base_timeout_ms


def get_completed_pickup_outcome(
    probe: action_session.WorldStateProviderProtocol,
    *,
    target_x: int,
    target_y: int,
    fuel_before: int,
) -> tuple[Literal["picked_up_fuel"], int, int] | None:
    """Return a completed pickup outcome once the fuel credit is observed.

    Args:
        probe: Probe exposing the latest world state.
        target_x: Pickup target X tile.
        target_y: Pickup target Y tile.
        fuel_before: Fuel value before the pickup started.

    Returns:
        Completed pickup tuple when fuel increased, otherwise None.

    Raises:
        PickupPhaseError: If self state disappears while waiting.
    """
    world = probe.get_world_state()
    self_state = world["self_state"]
    if self_state is None:
        raise PickupPhaseError("self state disappeared while waiting for fuel pickup")
    _ = (target_x, target_y)
    if self_state["fuel"] > fuel_before:
        return ("picked_up_fuel", action_hooks.get_current_time_ms(), self_state["fuel"])
    return None


def wait_for_pickup_outcome(
    page: action_session.WaitPageProtocol,
    probe: action_session.BufferedWorldStateProviderProtocol,
    *,
    target_x: int,
    target_y: int,
    pickup_started_ms: int,
    fuel_before: int,
    timeout_ms: int,
) -> tuple[Literal["picked_up_fuel", "pickup_timeout"], int, int]:
    """Wait for a fuel pickup to complete or time out.

    Args:
        page: Page used for polling waits.
        probe: Probe exposing movement and world state.
        target_x: Pickup target X tile.
        target_y: Pickup target Y tile.
        pickup_started_ms: Timestamp when pickup tracking started.
        fuel_before: Fuel value before the pickup started.
        timeout_ms: Maximum wait time.

    Returns:
        Terminal pickup status, completion timestamp, and fuel after completion.

    Raises:
        PickupPhaseError: If self state disappears while waiting or after timeout.
    """
    while action_hooks.get_current_time_ms() - pickup_started_ms < timeout_ms:
        action_hooks.drain_buffered_messages(probe)
        pickup_outcome = get_completed_pickup_outcome(
            probe,
            target_x=target_x,
            target_y=target_y,
            fuel_before=fuel_before,
        )
        if pickup_outcome is not None:
            return pickup_outcome
        page.wait_for_timeout(_PICKUP_POLL_INTERVAL_MS)
    self_state = probe.get_world_state()["self_state"]
    if self_state is None:
        raise PickupPhaseError("self state disappeared after fuel pickup timeout")
    return ("pickup_timeout", action_hooks.get_current_time_ms(), self_state["fuel"])


class PickupImmediateOutcomeProtocol(Protocol):
    """Callable protocol for immediate pickup completion checks."""

    def __call__(
        self,
        probe: action_session.WorldStateProviderProtocol,
        *,
        target_x: int,
        target_y: int,
        fuel_before: int,
    ) -> tuple[Literal["picked_up_fuel"], int, int] | None:
        """Return an immediate pickup completion if available."""


class PickupOutcomeWaiterProtocol(Protocol):
    """Callable protocol for waiting on a terminal pickup outcome."""

    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        probe: action_session.BufferedWorldStateProviderProtocol,
        *,
        target_x: int,
        target_y: int,
        pickup_started_ms: int,
        fuel_before: int,
        timeout_ms: int,
    ) -> tuple[Literal["picked_up_fuel", "pickup_timeout"], int, int]:
        """Wait for one terminal pickup outcome."""


class PickupTimeoutSizerProtocol(Protocol):
    """Callable protocol for sizing pickup timeouts."""

    def __call__(
        self,
        *,
        current_x: int,
        current_y: int,
        target_x: int,
        target_y: int,
        base_timeout_ms: int,
    ) -> int:
        """Return the effective pickup timeout in milliseconds."""


def run_tracked_pickup_phase(
    page: action_session.WaitPageProtocol,
    probe: PickupPhaseProbeProtocol,
    *,
    attempt_label: str,
    target_x: int,
    target_y: int,
    current_x: int,
    current_y: int,
    fuel_before_pickup: int,
    pickup_timeout_ms: int,
    dispatch_failure_error: type[Exception],
    dispatch_failure_message: str = "move_to command dispatch failed during fuel collection",
    get_completed_outcome: PickupImmediateOutcomeProtocol = get_completed_pickup_outcome,
    wait_for_outcome: PickupOutcomeWaiterProtocol = wait_for_pickup_outcome,
    compute_timeout: PickupTimeoutSizerProtocol = effective_pickup_timeout_ms,
) -> tuple[
    ActionPhaseCycleDict,
    ActionPhaseCycleDict,
    int,
    Literal["picked_up_fuel", "pickup_timeout"],
    int,
    int,
]:
    """Run one tracked move-and-pickup phase.

    Args:
        page: Page used for polling waits.
        probe: Probe implementation dispatching movement.
        attempt_label: Attempt label attached to move and pickup cycles.
        target_x: Pickup target X tile.
        target_y: Pickup target Y tile.
        current_x: Current self X tile before pickup.
        current_y: Current self Y tile before pickup.
        fuel_before_pickup: Fuel value before pickup handling starts.
        pickup_timeout_ms: Base timeout for the pickup phase.
        dispatch_failure_error: Exception type raised on dispatch failure.
        dispatch_failure_message: Error text for dispatch failure.
        get_completed_outcome: Immediate pickup completion checker.
        wait_for_outcome: Terminal pickup outcome waiter.
        compute_timeout: Effective timeout calculator.

    Returns:
        Tuple of ``(move_cycle, pickup_cycle, pickup_started_ms, status,
        completion_timestamp_ms, fuel_after)``.

    Raises:
        PickupPhaseError: If self state disappears during pickup handling.
        Exception: Raised via ``dispatch_failure_error`` if movement dispatch
            fails.
    """
    pickup_started_ms = action_hooks.get_current_time_ms()
    move_cycle = probe._start_action_phase("move", attempt_label=attempt_label)
    pickup_cycle = probe._start_action_phase("pickup", attempt_label=attempt_label)
    timeout_ms = compute_timeout(
        current_x=current_x,
        current_y=current_y,
        target_x=target_x,
        target_y=target_y,
        base_timeout_ms=pickup_timeout_ms,
    )
    action_hooks.drain_buffered_messages(probe)
    immediate_pickup_outcome = get_completed_outcome(
        probe,
        target_x=target_x,
        target_y=target_y,
        fuel_before=fuel_before_pickup,
    )
    if immediate_pickup_outcome is None:
        if not probe.move_to(target_x, target_y):
            emit_command_dispatch_failure_diagnostic("move", dispatch_failure_message)
            probe._end_action_phase(move_cycle)
            probe._end_action_phase(pickup_cycle)
            raise dispatch_failure_error(dispatch_failure_message)
        status, completion_timestamp_ms, fuel_after = wait_for_outcome(
            page,
            probe,
            target_x=target_x,
            target_y=target_y,
            pickup_started_ms=pickup_started_ms,
            fuel_before=fuel_before_pickup,
            timeout_ms=timeout_ms,
        )
    else:
        status, completion_timestamp_ms, fuel_after = immediate_pickup_outcome
    probe._end_action_phase(move_cycle)
    probe._end_action_phase(pickup_cycle)
    probe._reset_probe_state_to_idle()
    return (
        move_cycle,
        pickup_cycle,
        pickup_started_ms,
        status,
        completion_timestamp_ms,
        fuel_after,
    )


__all__ = [
    "PickupImmediateOutcomeProtocol",
    "PickupOutcomeWaiterProtocol",
    "PickupPhaseError",
    "PickupPhaseProbeProtocol",
    "PickupTimeoutSizerProtocol",
    "effective_pickup_timeout_ms",
    "get_completed_pickup_outcome",
    "run_tracked_pickup_phase",
    "wait_for_pickup_outcome",
]
