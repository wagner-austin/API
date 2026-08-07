"""Replay harness core: load a capture and drive a probe through it.

The capture loaders, the probe context and result types, the bootstrap
and dispatch mixins, and ``prepare_probe_replay``. The page and CDP
doubles are :mod:`tests.action_lab._replay_page` and
:mod:`tests.action_lab._replay_cdp`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import (
    dataclass,
    field,
)
from pathlib import Path
from typing import (
    Generic,
    Protocol,
    TypeVar,
)

from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
)
from tests.action_lab._replay_cdp import WorldStateDerivedCDP
from tests.action_lab._replay_page import (
    FrameBatchSource,
    ReplayClock,
    ReplayPage,
)

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import (
    BufferedMessageSourceProtocol,
    CDPSessionProtocol,
    PageProtocol,
)
from tankpit_bot.capture.xor import build_session_xor_table
from tankpit_bot.sniffer.world_state import get_world_state
from tankpit_bot.state import (
    SelfStateDict,
    WorldStateDict,
)
from tankpit_bot.types import (
    CaptureSession,
    decode_capture_session,
)


def load_capture(path: Path) -> CaptureSession:
    """Load and validate a captured session via the real codec.

    Args:
        path: Filesystem path to the capture JSON.

    Returns:
        Validated capture session.
    """
    text = core_hooks.read_text(path)
    return decode_capture_session(narrow_json_to_dict(load_json_str(text)))


def received_payloads(session: CaptureSession) -> list[str]:
    """Filter the message log to received frames in order.

    Args:
        session: Capture session payload.

    Returns:
        Base64-encoded received payloads in original order.
    """
    return [m["payload"] for m in session["messages"] if m["direction"] == "received"]


@dataclass
class ReplayProbeContext:
    """Wired-up real probe ready to run an attempt body.

    Returned by :func:`prepare_probe_replay`. Each field is a real
    production object or one of the shared seam substitutes:

    * ``probe`` -- the production probe class instance with its
      ``_magic``, ``_page`` and ``_cdp`` already seeded.
    * ``page`` -- the :class:`ReplayPage` driving frame ingestion.
    * ``clock`` -- the :class:`ReplayClock` substituted into
      :data:`tankpit_bot.action_lab._test_hooks.get_current_time_ms`.
    * ``frame_source`` -- the :class:`FrameBatchSource` cursor; tests
      can inspect ``frame_source.consumed`` and the harness can pull
      ``initial_frames`` to fold into the final result.
    * ``initial_frames`` -- frames drained during the pre-attempt sync.
    * ``restore_clock`` -- callable that puts the production
      ``get_current_time_ms`` hook back. The caller owns the unwind.
    """

    probe: BufferedMessageSourceProtocol
    page: ReplayPage
    clock: ReplayClock
    frame_source: FrameBatchSource
    initial_frames: int
    restore_clock: Callable[[], None]


_AttemptResultT = TypeVar("_AttemptResultT")


@dataclass
class ReplayResult(Generic[_AttemptResultT]):
    """Aggregated outcome of one probe replay attempt.

    Parameterized by the probe-specific attempt-result TypedDict
    (e.g. :class:`MovementProbeAttemptResultDict`,
    :class:`FuelProbeAttemptResultDict`). Each probe-specific replay
    function returns ``ReplayResult[<that probe's attempt result>]``
    -- one generic shape, zero forked copies.
    """

    attempt: _AttemptResultT
    dispatched_commands: list[str]
    frames_fed: int
    waits_ms: list[float] = field(default_factory=list)


class StubbedBootstrapMixin:
    """Stub the probe bootstrap so ``execute_probe`` runs without a browser.

    Sets ``action_hooks`` lifecycle stubs so ``prepare_live_probe_runtime``
    and ``execute_live_probe_bootstrap`` skip real browser interaction.

    Subclasses must call :meth:`_init_bootstrap_stubs` from their own
    ``__init__`` (after their real-probe ``__init__``).
    """

    cleanup_calls: int
    _magic: str | None

    def _init_bootstrap_stubs(self) -> None:
        """Install lifecycle hook stubs and initialize counters."""

        self.cleanup_calls = 0

        from tankpit_bot.action_lab import _test_hooks as action_hooks
        from tankpit_bot.browser.lifecycle import (
            navigate_and_login as _real_navigate,
        )
        from tankpit_bot.types import CapturedMessage

        def _bootstrap_navigate(
            page: PageProtocol,
            cdp: CDPSessionProtocol,
            *,
            target_url: str,
            prefer_account: bool,
            tank_name_prefix: str = "TP",
            auto_join_room: bool = True,
        ) -> None:
            _real_navigate(
                page,
                cdp,
                target_url=target_url,
                prefer_account=prefer_account,
                tank_name_prefix=tank_name_prefix,
                auto_join_room=False,
            )

        def _bootstrap_wait_ready(
            page: PageProtocol,
            messages: list[CapturedMessage],
        ) -> None:
            _ = (page, messages)

        def _bootstrap_gather_intel(
            page: PageProtocol,
            cdp: CDPSessionProtocol,
        ) -> str | None:
            _ = (page, cdp)
            return None

        action_hooks.navigate_and_login = _bootstrap_navigate
        action_hooks.wait_for_game_ready = _bootstrap_wait_ready
        action_hooks.gather_intel = _bootstrap_gather_intel

    def _setup_console_listener(self, cdp: CDPSessionProtocol) -> None:
        """Skip real console-listener wiring."""
        _ = cdp

    def _setup_cdp_handlers(self, cdp: CDPSessionProtocol) -> None:
        """Skip real CDP-handler wiring."""
        _ = cdp


class WorldStateOverrideMixin:
    """Override ``get_world_state``/``get_self_state`` from ``self._world_state``.

    Used by harnesses that bypass the world-state singleton and drive
    a controllable world dict directly. Subclasses must assign
    ``self._world_state`` before any production code path reads it.
    """

    _world_state: WorldStateDict

    def get_world_state(self) -> WorldStateDict:
        """Return the controllable world state."""
        return self._world_state

    def get_self_state(self) -> SelfStateDict | None:
        """Return the controllable self state."""
        return self._world_state["self_state"]


class DispatchCaptureMixin:
    """Shared ``_send_bytes`` capture + optional failure injection.

    Every ``Replay*Probe`` subclass exposes the same two test
    affordances: it records the structured ``cmd_name`` of every
    dispatched command, and it optionally fails specific commands at
    the real dispatch boundary to exercise production error paths.
    Lifted here so each replay subclass declares the affordances by
    inheriting this mixin instead of carrying its own copy.

    Subclasses must call :meth:`_init_dispatch_capture` from their
    own ``__init__`` (after their real-probe ``__init__``) to set up
    the per-instance state.
    """

    dispatched_commands: list[str]
    _fail_command: Callable[[str], bool] | None

    def _init_dispatch_capture(
        self,
        fail_command: Callable[[str], bool] | None,
    ) -> None:
        """Initialize per-instance capture state.

        Args:
            fail_command: Predicate over ``cmd_name`` labels. When
                supplied and it returns ``True`` for a label, the real
                ``_send_bytes`` returns ``False`` for that command --
                exactly what the live runtime does when the socket is
                down.
        """
        self.dispatched_commands = []
        self._fail_command = fail_command

    def _send_bytes(self, data: bytes, cmd_name: str) -> bool:
        """Capture the dispatched command label and optionally fail.

        Args:
            data: Encoded command bytes (ignored in replay).
            cmd_name: Structured command label produced by the
                production encoder (e.g. ``"map_open"``,
                ``"teleport(131,110)"``, ``"radar"``).

        Returns:
            False when ``fail_command`` matches the label; True
            otherwise. The probe's state machine then advances
            exactly as it would after a live success/failure.
        """
        _ = data
        self.dispatched_commands.append(cmd_name)
        return not (self._fail_command is not None and self._fail_command(cmd_name))


def _drain_until_self_state(
    probe: BufferedMessageSourceProtocol,
    frame_source: FrameBatchSource,
    initial_sync_batches: int,
    drain_messages: Callable[[BufferedMessageSourceProtocol], int],
) -> int:
    """Drain frame batches until the world's ``self_state`` materializes.

    Args:
        probe: Probe whose ``_cdp_message_buffer`` receives frames.
        frame_source: Cursor to pull batches from.
        initial_sync_batches: Maximum number of batches to drain.
        drain_messages: Production ``drain_messages`` callable.

    Returns:
        Number of frames consumed during the initial sync.

    Raises:
        RuntimeError: If ``self_state`` does not arrive within
            ``initial_sync_batches`` batches.
    """
    initial_frames = 0
    self_state_populated = False
    for _ in range(initial_sync_batches):
        batch = frame_source.next_batch()
        if not batch:
            break
        probe._cdp_message_buffer.extend(batch)
        drain_messages(probe)
        initial_frames += len(batch)
        if get_world_state()["self_state"] is not None:
            self_state_populated = True
            break
    if not self_state_populated:
        raise RuntimeError(
            f"replay did not produce a self_state within "
            f"{initial_sync_batches} frame batches; increase "
            f"initial_sync_batches or frames_per_wait"
        )
    return initial_frames


class _ReplayProbeShape(Protocol):
    """Production-shaped probe slots the harness writes into.

    The probe subclasses that the per-probe harnesses build all expose
    these attributes through their parent class hierarchy
    (:class:`tankpit_bot.bot.base.Bot`). Declaring the slots
    structurally lets :func:`prepare_probe_replay` write to them
    without giving up strict typing.
    """

    _cdp_message_buffer: list[str]
    _magic: str | None
    _page: PageProtocol | None
    _cdp: CDPSessionProtocol | None
    xor_table: bytes | None


def prepare_probe_replay(
    capture_path: Path,
    probe: _ReplayProbeShape,
    *,
    initial_sync_batches: int,
    frames_per_wait: int,
    omit_cdp: bool,
    drain_messages: Callable[[BufferedMessageSourceProtocol], int],
    update_state_from_world: Callable[[], None],
    reset_world_state: Callable[[], None],
) -> ReplayProbeContext:
    """Wire a real probe up to a captured session for one attempt.

    Loads the capture, rebuilds the XOR table from its magic key,
    drains initial frames until ``self_state`` populates, installs the
    :class:`ReplayPage` substitute, and points
    ``get_current_time_ms`` at the :class:`ReplayClock`.

    The caller is responsible for invoking
    ``context.restore_clock()`` after running the attempt body; the
    paired ``try/finally`` pattern lives in the per-probe harness so
    each probe's specific attempt call is visible at its own call
    site.

    Args:
        capture_path: Path to a captured ``*.capture_session.json``.
        probe: Probe instance to wire up. Must satisfy
            :class:`_ReplayProbeShape`.
        initial_sync_batches: Maximum number of frame batches drained
            before the attempt begins.
        frames_per_wait: Frames fed into ``_cdp_message_buffer`` on
            each wait helper poll.
        omit_cdp: When True, the probe's ``_cdp`` slot is left ``None``
            (exercises the real "cdp session is unavailable" error
            path).
        drain_messages: Production ``drain_messages`` callable.
        update_state_from_world: Production
            ``probe._update_state_from_world`` bound method.
        reset_world_state: Production ``reset_world_state`` callable.

    Returns:
        :class:`ReplayProbeContext` -- everything the caller needs to
        invoke the probe's attempt body and capture the result.

    Raises:
        RuntimeError: If the capture has no magic key, or if no
            ``self_state`` materializes within the initial-sync
            budget.
    """
    from tankpit_bot.action_lab import _test_hooks as action_hooks

    session = load_capture(capture_path)
    magic = session["magic"]
    if magic is None:
        raise RuntimeError(f"capture {capture_path.name} has no magic key")

    reset_world_state()
    probe.xor_table = build_session_xor_table(magic)

    probe._magic = magic
    probe._cdp = None if omit_cdp else WorldStateDerivedCDP()

    payloads = received_payloads(session)
    frame_source = FrameBatchSource(payloads, frames_per_wait)

    initial_frames = _drain_until_self_state(
        probe,
        frame_source,
        initial_sync_batches,
        drain_messages,
    )
    update_state_from_world()

    clock = ReplayClock()
    page = ReplayPage(probe, frame_source, clock)
    probe._page = page

    original_get_time = action_hooks.get_current_time_ms
    action_hooks.get_current_time_ms = clock

    def _restore_clock() -> None:
        """Put the production wall-clock hook back."""
        action_hooks.get_current_time_ms = original_get_time

    return ReplayProbeContext(
        probe=probe,
        page=page,
        clock=clock,
        frame_source=frame_source,
        initial_frames=initial_frames,
        restore_clock=_restore_clock,
    )
