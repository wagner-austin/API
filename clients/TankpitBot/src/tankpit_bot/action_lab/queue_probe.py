"""Live command queue probe — tests multi-command batching.

Connects to the game, enters a room, and sends command pairs
back-to-back to measure how the server processes queued commands.

Experiments:
1. shoot + pickup — does the server process both in one tick?
2. shoot + shoot — are rapid shots queued or dropped?
3. move + pickup — can a pickup ride the same tick as a move?
"""

from __future__ import annotations

from typing import Literal, Protocol

from platform_core.logging import get_logger

from tankpit_bot._test_hooks.bot import BufferedMessageSourceProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.probe_base import ProbeBase, ProbeError
from tankpit_bot.action_lab.probe_entrypoint import run_and_save_standard_probe_session
from tankpit_bot.action_lab.probe_runtime import (
    ProbeCommandReadyContextDict,
    execute_live_probe_bootstrap,
)
from tankpit_bot.action_lab.probe_session import build_probe_session_envelope
from tankpit_bot.action_lab.queue_probe_types import (
    QueueCommandTimingDict,
    QueueExperimentKind,
    QueueExperimentResultDict,
    QueueProbeSessionDict,
    encode_queue_probe_session,
)
from tankpit_bot.state import SelfStateDict, WorldStateDict
from tankpit_bot.types import CapturedMessage

log = get_logger(__name__)

_POLL_INTERVAL_MS = 100
_SETTLE_MS = 500


class QueueWaitProbeProtocol(BufferedMessageSourceProtocol, Protocol):
    """Minimal probe protocol for queue experiment wait loops."""

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
        action_hooks.drain_buffered_messages(probe)
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
        action_hooks.drain_buffered_messages(probe)
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
        action_hooks.drain_buffered_messages(probe)
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
    action_hooks.drain_buffered_messages(probe)

    if kind == "shoot_then_pickup":
        return run_shoot_then_pickup_experiment(probe, timeout_ms=timeout_ms)
    if kind == "shoot_then_shoot":
        return run_shoot_then_shoot_experiment(probe, timeout_ms=timeout_ms)
    return run_move_then_pickup_experiment(probe, timeout_ms=timeout_ms)


class QueueProbe(ProbeBase):
    """Live probe that tests server command queue behavior."""

    def execute_probe(
        self,
        *,
        initial_sync_timeout_ms: int,
        experiment_timeout_ms: int,
        experiment_kinds: list[QueueExperimentKind],
    ) -> QueueProbeSessionDict:
        """Run the live queue probe session.

        Args:
            initial_sync_timeout_ms: Maximum wait for initial sync.
            experiment_timeout_ms: Maximum wait per experiment.
            experiment_kinds: Which experiments to run.

        Returns:
            Complete session with all experiment results.

        Raises:
            ValueError: If timeout is non-positive or kinds is empty.
        """
        if experiment_timeout_ms <= 0:
            raise ValueError("experiment_timeout_ms must be positive")
        if not experiment_kinds:
            raise ValueError("experiment_kinds must not be empty")

        def _run_ready_session(
            context: ProbeCommandReadyContextDict,
        ) -> QueueProbeSessionDict:
            experiments: list[QueueExperimentResultDict] = []
            for kind in experiment_kinds:
                result = run_single_experiment(
                    self,
                    kind,
                    timeout_ms=experiment_timeout_ms,
                )
                experiments.append(result)
                log.info(
                    "Experiment %s: %s (primary %sms, secondary %sms, gap %dms)",
                    kind,
                    result["status"],
                    result["primary"]["elapsed_ms"],
                    result["secondary"]["elapsed_ms"],
                    result["inter_send_delay_ms"],
                )

            first_started_ms = experiments[0]["primary"]["sent_ms"] if experiments else None
            envelope = build_probe_session_envelope(
                self,
                context=context,
                first_attempt_started_ms=first_started_ms,
            )
            return QueueProbeSessionDict(
                session_id=envelope.session_id,
                start_timestamp_ms=envelope.start_timestamp_ms,
                end_timestamp_ms=envelope.end_timestamp_ms,
                base_url=envelope.base_url,
                spawn_x=envelope.spawn_x,
                spawn_y=envelope.spawn_y,
                capture_session_path="",
                initial_sync_timeout_ms=initial_sync_timeout_ms,
                experiment_timeout_ms=experiment_timeout_ms,
                startup_timing=envelope.startup_timing,
                experiments=experiments,
            )

        return execute_live_probe_bootstrap(
            self,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            run_ready_session=_run_ready_session,
        )


_DEFAULT_EXPERIMENT_KINDS: list[QueueExperimentKind] = [
    "shoot_then_pickup",
    "shoot_then_shoot",
    "move_then_pickup",
]


def format_queue_probe_summary(session: QueueProbeSessionDict) -> str:
    """Format a human-readable summary of a queue probe session.

    Args:
        session: Completed queue probe session.

    Returns:
        Multi-line summary string.
    """
    lines = [
        f"Queue probe session {session['session_id']}",
        f"  Spawn: ({session['spawn_x']}, {session['spawn_y']})",
        f"  Experiments: {len(session['experiments'])}",
    ]
    for exp in session["experiments"]:
        p_ms = exp["primary"]["elapsed_ms"]
        s_ms = exp["secondary"]["elapsed_ms"]
        lines.append(
            f"  {exp['kind']}: {exp['status']}"
            f" (primary={p_ms}ms, secondary={s_ms}ms, gap={exp['inter_send_delay_ms']}ms)"
        )
    return "\n".join(lines)


def _create_queue_probe(
    target_url: str,
    *,
    headless: bool,
    prefer_account: bool,
) -> QueueProbe:
    """Factory for creating QueueProbe instances with injected services.

    Args:
        target_url: Browser target URL.
        headless: Whether to run headless.
        prefer_account: Whether to prefer account login.

    Returns:
        New QueueProbe instance with factory-wired services.
    """
    from tankpit_bot.action_lab.probe_factory import create_probe

    probe = create_probe(
        QueueProbe,
        target_url,
        headless=headless,
        prefer_account=prefer_account,
    )
    assert isinstance(probe, QueueProbe)
    return probe


def run_queue_probe(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    prefer_account: bool = False,
    initial_sync_timeout_ms: int = 10000,
    experiment_timeout_ms: int = 5000,
    experiment_kinds: list[QueueExperimentKind] | None = None,
) -> QueueProbeSessionDict:
    """Run a live queue probe and save the session JSON.

    Args:
        target_url: Browser target URL.
        output_path: JSON output path.
        headless: Whether to run headless.
        prefer_account: Whether to prefer account login.
        initial_sync_timeout_ms: Maximum wait for initial sync.
        experiment_timeout_ms: Maximum wait per experiment.
        experiment_kinds: Which experiments to run (defaults to all three).

    Returns:
        Completed and persisted session payload.
    """
    kinds = experiment_kinds if experiment_kinds is not None else _DEFAULT_EXPERIMENT_KINDS

    def _run_session(probe: QueueProbe) -> QueueProbeSessionDict:
        return probe.execute_probe(
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            experiment_timeout_ms=experiment_timeout_ms,
            experiment_kinds=kinds,
        )

    return run_and_save_standard_probe_session(
        probe_factory=_create_queue_probe,
        run_session=_run_session,
        encoder=encode_queue_probe_session,
        summary_formatter=format_queue_probe_summary,
        target_url=target_url,
        output_path=output_path,
        headless=headless,
        prefer_account=prefer_account,
    )


__all__ = [
    "QueueExperimentProbeProtocol",
    "QueueProbe",
    "QueueProbeError",
    "QueueWaitProbeProtocol",
    "_build_command_timing",
    "_determine_experiment_status",
    "_require_self_state",
    "_wait_for_fuel_change",
    "_wait_for_position_change",
    "_wait_for_world_timestamp_advance",
    "format_queue_probe_summary",
    "run_move_then_pickup_experiment",
    "run_queue_probe",
    "run_shoot_then_pickup_experiment",
    "run_shoot_then_shoot_experiment",
    "run_single_experiment",
]
