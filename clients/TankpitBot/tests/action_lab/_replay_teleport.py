"""Teleport-probe replay harness on top of the shared replay core.

Drives a real :class:`tankpit_bot.action_lab.teleport.TeleportProbe`
through a captured WebSocket session. All generic infrastructure
(:class:`ReplayPage`, :class:`FrameBatchSource`,
:class:`WorldStateDerivedCDP`, :class:`ReplayClock`,
:func:`load_capture`, :func:`prepare_probe_replay`) lives in
:mod:`tests.action_lab._replay_core`; this module only owns the
teleport-probe-specific subclass and the attempt-orchestration entry
point.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from tests.action_lab._replay_core import (
    DispatchCaptureMixin,
    ReplayResult,
    prepare_probe_replay,
)

from tankpit_bot.action_lab.teleport import TeleportProbe
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportTargetDict,
)
from tankpit_bot.bot.world_sync import drain_messages
from tankpit_bot.sniffer.world_service import WorldService

TeleportReplayResult = ReplayResult[TeleportAttemptResultDict]

__all__ = [
    "ReplayTeleportProbe",
    "TeleportReplayResult",
    "replay_teleport_attempt",
]


class ReplayTeleportProbe(DispatchCaptureMixin, TeleportProbe):
    """Real :class:`TeleportProbe` with WebSocket dispatch captured.

    Skips the browser bootstrap by leaving Playwright untouched. The
    harness manually wires ``_magic``, ``_page`` and ``_cdp`` before
    issuing the attempt call. Dispatch capture and failure injection
    are inherited from :class:`DispatchCaptureMixin`.
    """

    def __init__(
        self,
        target_url: str = "https://tankpit.com/play",
        *,
        fail_command: Callable[[str], bool] | None = None,
        world: WorldService | None = None,
    ) -> None:
        """Initialize the replay probe.

        Args:
            target_url: Game URL the probe records on its session.
            fail_command: Forwarded to :class:`DispatchCaptureMixin`.
            world: Injected WorldService. Created internally if None.
        """
        TeleportProbe.__init__(self, target_url, headless=True, prefer_account=False, world=world)
        self._init_dispatch_capture(fail_command)


def replay_teleport_attempt(
    capture_path: Path,
    target: TeleportTargetDict,
    *,
    map_sync_timeout_ms: int = 3_000,
    teleport_timeout_ms: int = 10_000,
    settle_delay_ms: int = 0,
    teleport_strategy: Literal[
        "sync_before_teleport", "immediate_after_map_open"
    ] = "immediate_after_map_open",
    initial_sync_batches: int = 20,
    frames_per_wait: int = 5,
    omit_cdp: bool = False,
    fail_command: Callable[[str], bool] | None = None,
) -> ReplayResult[TeleportAttemptResultDict]:
    """Drive a real :class:`TeleportProbe` through a captured session.

    Args:
        capture_path: Path to a captured ``*.capture_session.json``.
        target: Teleport target the probe should pivot to.
        map_sync_timeout_ms: Map-sync timeout (matches production
            default tuning).
        teleport_timeout_ms: Teleport-completion timeout.
        settle_delay_ms: Optional settle-delay after attempt completion.
        teleport_strategy: Teleport strategy selection forwarded to the
            production attempt body.
        initial_sync_batches: Maximum number of frame batches drained
            before the attempt begins, waiting for ``self_state`` to
            populate.
        frames_per_wait: Frames fed into ``_cdp_message_buffer`` on
            each wait helper poll.
        omit_cdp: Build the probe without a CDP session. Exercises the
            real probe error path for "cdp session is unavailable".

    Returns:
        :class:`ReplayResult` parameterized by
        :class:`TeleportAttemptResultDict` -- the production attempt
        dict,
        the captured command labels, the count of recorded frames
        consumed, and the timeline of poll-wait durations.

    Raises:
        RuntimeError: If the capture has no magic key (cannot rebuild
            the XOR table) or if no ``self_state`` materializes within
            the configured initial-sync budget.
    """
    ws = WorldService()
    probe = ReplayTeleportProbe(fail_command=fail_command, world=ws)
    context = prepare_probe_replay(
        capture_path,
        probe,
        initial_sync_batches=initial_sync_batches,
        frames_per_wait=frames_per_wait,
        omit_cdp=omit_cdp,
        drain_messages=drain_messages,
        update_state_from_world=probe._update_state_from_world,
    )
    try:
        attempt = probe._probe_single_target(
            target,
            teleport_strategy=teleport_strategy,
            map_sync_timeout_ms=map_sync_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
            settle_delay_ms=settle_delay_ms,
        )
    finally:
        context.restore_clock()

    return ReplayResult[TeleportAttemptResultDict](
        attempt=attempt,
        dispatched_commands=probe.dispatched_commands,
        frames_fed=context.initial_frames + context.page.frames_fed,
        waits_ms=context.page.waits_ms,
    )
