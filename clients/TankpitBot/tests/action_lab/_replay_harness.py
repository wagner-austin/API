"""Movement-probe replay harness on top of the shared replay core.

Drives a real :class:`tankpit_bot.action_lab.movement_probe.MovementProbe`
through a captured WebSocket session. All generic infrastructure
(:class:`ReplayPage`, :class:`FrameBatchSource`,
:class:`WorldStateDerivedCDP`, :class:`ReplayClock`,
:func:`load_capture`, :func:`prepare_probe_replay`) lives in
:mod:`tests.action_lab._replay_core`; this module only owns the
movement-probe-specific subclass and the attempt-orchestration
entry point.

This is not a fake-driven test surface: every observable result comes
from real production code reading real captured bytes. The harness is a
seam, not a substitute for logic.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tests.action_lab._replay_core import (
    DispatchCaptureMixin,
    ReplayResult,
    prepare_probe_replay,
)

from tankpit_bot.action_lab.movement_probe import MovementProbe
from tankpit_bot.action_lab.movement_probe_types import MovementProbeAttemptResultDict
from tankpit_bot.action_lab.types import TeleportTargetDict
from tankpit_bot.bot.world_sync import drain_messages
from tankpit_bot.sniffer.world_state import reset_world_state

__all__ = [
    "ReplayMovementProbe",
    "replay_movement_attempt",
]


class ReplayMovementProbe(DispatchCaptureMixin, MovementProbe):
    """Real :class:`MovementProbe` with WebSocket dispatch captured.

    Skips the browser bootstrap that :class:`MovementProbe.__init__`
    normally relies on by leaving Playwright untouched. The harness
    manually wires ``_magic``, ``_page`` and ``_cdp`` before issuing
    the attempt call. Dispatch capture and failure injection are
    inherited from :class:`DispatchCaptureMixin`.
    """

    def __init__(
        self,
        target_url: str = "https://tankpit.com/play",
        *,
        fail_command: Callable[[str], bool] | None = None,
    ) -> None:
        """Initialize the replay probe.

        Args:
            target_url: Game URL the probe records on its session.
            fail_command: Forwarded to :class:`DispatchCaptureMixin`.
        """
        MovementProbe.__init__(self, target_url, headless=True, prefer_account=False)
        self._init_dispatch_capture(fail_command)


def replay_movement_attempt(
    capture_path: Path,
    target: TeleportTargetDict,
    *,
    move_timeout_ms: int = 5000,
    settle_delay_ms: int = 0,
    queue_map_open_during_move: bool = False,
    map_open_delay_ms: int = 0,
    initial_sync_batches: int = 3,
    frames_per_wait: int = 1,
    fail_command: Callable[[str], bool] | None = None,
    omit_cdp: bool = False,
) -> ReplayResult[MovementProbeAttemptResultDict]:
    """Drive a real :class:`MovementProbe` through a captured session.

    Args:
        capture_path: Path to a captured ``*.capture_session.json``.
        target: Movement target the probe should walk to.
        move_timeout_ms: Move-completion timeout (matches production
            default tuning).
        settle_delay_ms: Optional settle-delay between attempt phases.
        queue_map_open_during_move: Whether to fire ``map_open`` during
            the move (forwarded to the probe attempt body).
        map_open_delay_ms: Delay before queued ``map_open``.
        initial_sync_batches: Maximum number of frame batches drained
            before the attempt begins, waiting for ``self_state`` to
            populate.
        frames_per_wait: Frames fed into ``_cdp_message_buffer`` on each
            wait helper poll.
        fail_command: Predicate over ``cmd_name`` labels. When
            supplied and it returns ``True`` for a label, the real
            ``_send_bytes`` returns ``False`` for that command --
            exactly what the live runtime does when the socket is down.
        omit_cdp: Build the probe without a CDP session. Exercises the
            real probe error path for "cdp session is unavailable".

    Returns:
        :class:`ReplayResult` parameterized by
        :class:`MovementProbeAttemptResultDict` -- the production
        attempt dict, the captured command labels, the count of
        recorded frames consumed, and the timeline of poll-wait
        durations.

    Raises:
        RuntimeError: If the capture has no magic key (cannot rebuild
            the XOR table) or if no ``self_state`` materializes within
            the configured initial-sync budget.
    """
    probe = ReplayMovementProbe(fail_command=fail_command)
    context = prepare_probe_replay(
        capture_path,
        probe,
        initial_sync_batches=initial_sync_batches,
        frames_per_wait=frames_per_wait,
        omit_cdp=omit_cdp,
        drain_messages=drain_messages,
        update_state_from_world=probe._update_state_from_world,
        reset_world_state=reset_world_state,
    )
    try:
        attempt = probe._probe_single_movement_target(
            target,
            move_timeout_ms=move_timeout_ms,
            queue_map_open_during_move=queue_map_open_during_move,
            map_open_delay_ms=map_open_delay_ms,
            settle_delay_ms=settle_delay_ms,
        )
    finally:
        context.restore_clock()

    return ReplayResult[MovementProbeAttemptResultDict](
        attempt=attempt,
        dispatched_commands=probe.dispatched_commands,
        frames_fed=context.initial_frames + context.page.frames_fed,
        waits_ms=context.page.waits_ms,
    )
