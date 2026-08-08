"""Queue experiments: run one command pair and time the acks.

The three command-pair experiments, the waits that detect each command
landing, and the dispatcher over them. The probe session that drives
them is :mod:`tankpit_bot.action_lab.queue_probe`.
"""

from __future__ import annotations

from typing import Literal, Protocol

from tankpit_bot._test_hooks.bot import BufferedMessageSourceProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.probe_base import ProbeError
from tankpit_bot.action_lab.queue_probe_types import (
    QueueCommandTimingDict,
    QueueExperimentKind,
    QueueExperimentResultDict,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import SelfStateDict, WorldStateDict
from tankpit_bot.types import CapturedMessage

_POLL_INTERVAL_MS = 100

_SETTLE_MS = 500


class QueueWaitProbeProtocol(BufferedMessageSourceProtocol, Protocol):
    """Minimal probe protocol for queue experiment wait loops."""

    #: This session's world service; decoded frames land here.
    world: WorldService

    def _update_state_from_world(self) -> None:
        """Advance internal state from the current world snapshot."""

    def get_world_state(self) -> WorldStateDict:
        """Return the current world state."""

    def get_self_state(self) -> SelfStateDict | None:
        """Return the current self state."""

    def _require_page(self) -> action_session.WaitPageProtocol:
        """Return the page or raise."""


class QueueExperimentProbeProtocol(QueueWaitProbeProtocol, Protocol):
    """Probe protocol for queue experiments — adds command dispatch."""

    @property
    def messages(self) -> list[CapturedMessage]:
        """Return the captured message list."""

    def shoot(self, x: int, y: int, target_id: int = 0) -> bool:
        """Send shoot command."""

    def pickup_fuel(self, x: int, y: int) -> bool:
        """Send fuel pickup command."""

    def move_to(self, x: int, y: int) -> bool:
        """Send move command."""


class QueueProbeError(ProbeError):
    """Raised when a queue probe experiment cannot proceed."""


def _wait_for_position_change(
    probe: QueueWaitProbeProtocol,
    *,
    baseline_x: int,
    baseline_y: int,
    started_ms: int,
    timeout_ms: int,
) -> int | None:
    """Poll until the tank position changes from baseline.

    Args:
        probe: Probe instance with page and world state access.
        baseline_x: X position at the time the command was sent.
        baseline_y: Y position at the time the command was sent.
        started_ms: Timestamp when waiting started.
        timeout_ms: Maximum wait time.

    Returns:
        Timestamp when the change was detected, or None on timeout.
    """
    page = probe._require_page()
    while action_hooks.get_current_time_ms() - started_ms < timeout_ms:
        action_hooks.drain_buffered_messages(probe, probe.world)
        probe._update_state_from_world()
        self_state = probe.get_self_state()
        if self_state is not None and (
            self_state["x"] != baseline_x or self_state["y"] != baseline_y
        ):
            return action_hooks.get_current_time_ms()
        page.wait_for_timeout(float(_POLL_INTERVAL_MS))
    return None


def _wait_for_fuel_change(
    probe: QueueWaitProbeProtocol,
    *,
    baseline_fuel: int,
    started_ms: int,
    timeout_ms: int,
) -> int | None:
    """Poll until the fuel level changes from baseline.

    Args:
        probe: Probe instance with page and world state access.
        baseline_fuel: Fuel level at the time the command was sent.
        started_ms: Timestamp when waiting started.
        timeout_ms: Maximum wait time.

    Returns:
        Timestamp when the change was detected, or None on timeout.
    """
    page = probe._require_page()
    while action_hooks.get_current_time_ms() - started_ms < timeout_ms:
        action_hooks.drain_buffered_messages(probe, probe.world)
        probe._update_state_from_world()
        self_state = probe.get_self_state()
        if self_state is not None and self_state["fuel"] != baseline_fuel:
            return action_hooks.get_current_time_ms()
        page.wait_for_timeout(float(_POLL_INTERVAL_MS))
    return None


def _wait_for_world_timestamp_advance(
    probe: QueueWaitProbeProtocol,
    *,
    baseline_ms: int,
    started_ms: int,
    timeout_ms: int,
) -> int | None:
    """Poll until the world timestamp advances past baseline.

    Args:
        probe: Probe instance.
        baseline_ms: World timestamp before commands were sent.
        started_ms: When waiting started.
        timeout_ms: Maximum wait time.

    Returns:
        Timestamp when advance was detected, or None on timeout.
    """
    page = probe._require_page()
    while action_hooks.get_current_time_ms() - started_ms < timeout_ms:
        action_hooks.drain_buffered_messages(probe, probe.world)
        probe._update_state_from_world()
        ws = probe.get_world_state()
        if ws["timestamp_ms"] > baseline_ms:
            return action_hooks.get_current_time_ms()
        page.wait_for_timeout(float(_POLL_INTERVAL_MS))
    return None


def _determine_experiment_status(
    primary_ack_ms: int | None,
    secondary_ack_ms: int | None,
) -> Literal["both_processed", "second_dropped", "timeout"]:
    """Determine experiment outcome from command acknowledgements."""
    if primary_ack_ms is not None and secondary_ack_ms is not None:
        return "both_processed"
    if primary_ack_ms is not None:
        return "second_dropped"
    return "timeout"


def _build_command_timing(
    label: str,
    sent_ms: int,
    ack_ms: int | None,
) -> QueueCommandTimingDict:
    """Build a command timing entry."""
    return QueueCommandTimingDict(
        label=label,
        sent_ms=sent_ms,
        ack_ms=ack_ms,
        elapsed_ms=ack_ms - sent_ms if ack_ms is not None else None,
    )


def _require_self_state(probe: QueueExperimentProbeProtocol) -> SelfStateDict:
    """Return self state or raise QueueProbeError.

    Args:
        probe: Probe to query for self state.

    Returns:
        Current SelfStateDict.

    Raises:
        QueueProbeError: If self state is not available.
    """
    self_state = probe.get_self_state()
    if self_state is None:
        raise QueueProbeError("self state unavailable")
    return self_state


def run_shoot_then_pickup_experiment(
    probe: QueueExperimentProbeProtocol,
    *,
    timeout_ms: int,
) -> QueueExperimentResultDict:
    """Send shoot then pickup_fuel back-to-back.

    Args:
        probe: Probe with command dispatch and world state access.
        timeout_ms: Maximum wait for server response.

    Returns:
        Experiment result with timing data.

    Raises:
        QueueProbeError: If command dispatch fails.
    """
    self_state = _require_self_state(probe)
    x, y = self_state["x"], self_state["y"]
    fuel_before = self_state["fuel"]
    world_ts_before = probe.get_world_state()["timestamp_ms"]
    message_start = len(probe.messages)

    shoot_sent_ms = action_hooks.get_current_time_ms()
    if not probe.shoot(x, y):
        raise QueueProbeError("shoot command dispatch failed")

    pickup_sent_ms = action_hooks.get_current_time_ms()
    if not probe.pickup_fuel(x, y):
        raise QueueProbeError("pickup_fuel command dispatch failed")

    shoot_ack_ms = _wait_for_world_timestamp_advance(
        probe,
        baseline_ms=world_ts_before,
        started_ms=shoot_sent_ms,
        timeout_ms=timeout_ms,
    )

    pickup_ack_ms: int | None = None
    if shoot_ack_ms is not None:
        self_state_after = probe.get_self_state()
        if self_state_after is not None and self_state_after["fuel"] != fuel_before:
            pickup_ack_ms = shoot_ack_ms

    end_ms = action_hooks.get_current_time_ms()
    return QueueExperimentResultDict(
        kind="shoot_then_pickup",
        status=_determine_experiment_status(shoot_ack_ms, pickup_ack_ms),
        primary=_build_command_timing("shoot", shoot_sent_ms, shoot_ack_ms),
        secondary=_build_command_timing("pickup_fuel", pickup_sent_ms, pickup_ack_ms),
        inter_send_delay_ms=pickup_sent_ms - shoot_sent_ms,
        total_elapsed_ms=end_ms - shoot_sent_ms,
        message_start_index=message_start,
        message_end_index=len(probe.messages),
    )


def run_shoot_then_shoot_experiment(
    probe: QueueExperimentProbeProtocol,
    *,
    timeout_ms: int,
) -> QueueExperimentResultDict:
    """Send two shoot commands back-to-back.

    Args:
        probe: Probe with command dispatch and world state access.
        timeout_ms: Maximum wait for server response.

    Returns:
        Experiment result with timing data.

    Raises:
        QueueProbeError: If command dispatch fails.
    """
    self_state = _require_self_state(probe)
    x, y = self_state["x"], self_state["y"]
    world_ts_before = probe.get_world_state()["timestamp_ms"]
    message_start = len(probe.messages)

    shoot1_sent_ms = action_hooks.get_current_time_ms()
    if not probe.shoot(x, y):
        raise QueueProbeError("first shoot command dispatch failed")

    shoot2_sent_ms = action_hooks.get_current_time_ms()
    if not probe.shoot(x, y):
        raise QueueProbeError("second shoot command dispatch failed")

    shoot1_ack_ms = _wait_for_world_timestamp_advance(
        probe,
        baseline_ms=world_ts_before,
        started_ms=shoot1_sent_ms,
        timeout_ms=timeout_ms,
    )

    shoot2_ack_ms: int | None = None
    if shoot1_ack_ms is not None:
        world_ts_mid = probe.get_world_state()["timestamp_ms"]
        shoot2_ack_ms = _wait_for_world_timestamp_advance(
            probe,
            baseline_ms=world_ts_mid,
            started_ms=shoot1_ack_ms,
            timeout_ms=timeout_ms,
        )

    end_ms = action_hooks.get_current_time_ms()
    return QueueExperimentResultDict(
        kind="shoot_then_shoot",
        status=_determine_experiment_status(shoot1_ack_ms, shoot2_ack_ms),
        primary=_build_command_timing("shoot_1", shoot1_sent_ms, shoot1_ack_ms),
        secondary=_build_command_timing("shoot_2", shoot2_sent_ms, shoot2_ack_ms),
        inter_send_delay_ms=shoot2_sent_ms - shoot1_sent_ms,
        total_elapsed_ms=end_ms - shoot1_sent_ms,
        message_start_index=message_start,
        message_end_index=len(probe.messages),
    )


def run_move_then_pickup_experiment(
    probe: QueueExperimentProbeProtocol,
    *,
    timeout_ms: int,
) -> QueueExperimentResultDict:
    """Send move then pickup_fuel back-to-back.

    Args:
        probe: Probe with command dispatch and world state access.
        timeout_ms: Maximum wait for server response.

    Returns:
        Experiment result with timing data.

    Raises:
        QueueProbeError: If command dispatch fails.
    """
    self_state = _require_self_state(probe)
    x, y = self_state["x"], self_state["y"]
    fuel_before = self_state["fuel"]
    target_x = min(x + 1, 255)
    message_start = len(probe.messages)

    move_sent_ms = action_hooks.get_current_time_ms()
    if not probe.move_to(target_x, y):
        raise QueueProbeError("move command dispatch failed")

    pickup_sent_ms = action_hooks.get_current_time_ms()
    if not probe.pickup_fuel(x, y):
        raise QueueProbeError("pickup_fuel command dispatch failed")

    move_ack_ms = _wait_for_position_change(
        probe,
        baseline_x=x,
        baseline_y=y,
        started_ms=move_sent_ms,
        timeout_ms=timeout_ms,
    )

    pickup_ack_ms: int | None = None
    if move_ack_ms is not None:
        self_state_after = probe.get_self_state()
        if self_state_after is not None and self_state_after["fuel"] != fuel_before:
            pickup_ack_ms = move_ack_ms

    end_ms = action_hooks.get_current_time_ms()
    return QueueExperimentResultDict(
        kind="move_then_pickup",
        status=_determine_experiment_status(move_ack_ms, pickup_ack_ms),
        primary=_build_command_timing("move", move_sent_ms, move_ack_ms),
        secondary=_build_command_timing("pickup_fuel", pickup_sent_ms, pickup_ack_ms),
        inter_send_delay_ms=pickup_sent_ms - move_sent_ms,
        total_elapsed_ms=end_ms - move_sent_ms,
        message_start_index=message_start,
        message_end_index=len(probe.messages),
    )


def run_single_experiment(
    probe: QueueExperimentProbeProtocol,
    kind: QueueExperimentKind,
    *,
    timeout_ms: int,
) -> QueueExperimentResultDict:
    """Dispatch and run a single queue experiment.

    Args:
        probe: Probe with command dispatch and world state access.
        kind: Which experiment to run.
        timeout_ms: Maximum wait for server response.

    Returns:
        Experiment result with timing data.
    """
    page = probe._require_page()
    page.wait_for_timeout(float(_SETTLE_MS))
    action_hooks.drain_buffered_messages(probe, probe.world)

    if kind == "shoot_then_pickup":
        return run_shoot_then_pickup_experiment(probe, timeout_ms=timeout_ms)
    if kind == "shoot_then_shoot":
        return run_shoot_then_shoot_experiment(probe, timeout_ms=timeout_ms)
    return run_move_then_pickup_experiment(probe, timeout_ms=timeout_ms)


__all__ = [
    "QueueExperimentProbeProtocol",
    "QueueProbeError",
    "QueueWaitProbeProtocol",
    "run_move_then_pickup_experiment",
    "run_shoot_then_pickup_experiment",
    "run_shoot_then_shoot_experiment",
    "run_single_experiment",
]
