"""Generic, probe-agnostic infrastructure for action-lab replay tests.

Anything in this module is shared by every probe-specific replay
harness:

* :class:`ReplayClock` -- a controlled wall-clock for hook substitution.
* :class:`FrameBatchSource` -- a mutable cursor over recorded payloads.
* :class:`ReplayPage` -- a :class:`PageProtocol` substitute whose
  ``wait_for_timeout`` advances the clock and feeds the next frame
  batch into the probe's CDP buffer.
* :class:`WorldStateDerivedCDP` -- a :class:`CDPSessionProtocol`
  substitute whose ``Runtime.evaluate`` responses are derived from the
  live world-state singleton (so snapshot capture sees a
  deterministic projection of the same truth the real bot would).
* :func:`load_capture` / :func:`received_payloads` -- shared capture
  decoding helpers.

The probe-specific harnesses (``_replay_movement.py``, etc.) wire these
into one ``replay_*_attempt`` function each. Keeping the core here
guarantees the seam tests in
:mod:`tests.action_lab.test_replay_harness_contracts` apply to every
probe equally.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Protocol, TypeVar

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    load_json_str,
    narrow_json_to_dict,
)

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import (
    BufferedMessageSourceProtocol,
    CDPSessionProtocol,
    KeyboardProtocol,
    PageProtocol,
    ResponseProtocol,
)
from tankpit_bot.sniffer.world_state import get_world_state
from tankpit_bot.state import SelfStateDict, WorldStateDict
from tankpit_bot.types import CaptureSession, decode_capture_session


@dataclass
class ReplayClock:
    """Monotonically advancing clock under harness control.

    Mirrors the production wall-clock signature
    (``Callable[[], int]``) so the action-lab hook
    ``action_hooks.get_current_time_ms`` can be pointed at it
    transparently.
    """

    now_ms: int = 0

    def __call__(self) -> int:
        """Return the current controlled timestamp in milliseconds."""
        return self.now_ms

    def advance(self, delta_ms: int) -> None:
        """Move the clock forward by ``delta_ms``.

        Args:
            delta_ms: Milliseconds to advance.
        """
        self.now_ms += delta_ms


class FrameBatchSource:
    """Mutable cursor over a captured-frame stream.

    The cursor walks a recorded payload list one batch at a time. The
    harness uses this instead of a generator/Iterator so the cursor is
    a first-class object with explicit state -- callers can inspect
    progress without consuming the source.
    """

    def __init__(self, payloads: list[str], batch_size: int) -> None:
        """Initialize the frame cursor.

        Args:
            payloads: Ordered base64-encoded received payloads.
            batch_size: Number of payloads returned per ``next_batch``
                call.
        """
        self._payloads = payloads
        self._batch_size = batch_size
        self._cursor = 0

    def next_batch(self) -> list[str]:
        """Pop the next batch.

        Returns:
            A list of up to ``batch_size`` payloads. Empty when the
            source is exhausted.
        """
        if self._cursor >= len(self._payloads):
            return []
        batch = self._payloads[self._cursor : self._cursor + self._batch_size]
        self._cursor += len(batch)
        return batch

    @property
    def consumed(self) -> int:
        """Return the number of payloads handed out so far."""
        return self._cursor


class _ReplayKeyboard:
    """No-op keyboard satisfying :class:`KeyboardProtocol`.

    The action-lab attempt loops never exercise keyboard input, so the
    methods are bare stubs that simply absorb their arguments.
    """

    def press(self, key: str, *, delay: float | None = None) -> None:
        """Absorb a key-press request."""
        _ = (key, delay)

    def type(self, text: str, *, delay: float | None = None) -> None:
        """Absorb a text-type request."""
        _ = (text, delay)


class ClockAdvancingPage:
    """``PageProtocol`` whose ``wait_for_timeout`` advances a clock.

    Lifted from per-test ``_FakePage`` forks that all share the same
    shape: every wait advances a :class:`ReplayClock`, optionally runs
    an ``on_wait`` callback (used by tests that sequence world-state
    providers between waits), and records the wait duration. All other
    PageProtocol methods are no-ops.

    Used by tests that don't need real frame replay (the
    :class:`ReplayPage` harness already covers that) but still need a
    page whose ``wait_for_timeout`` ticks deterministically.
    """

    url = "https://tankpit.com/play"

    def __init__(
        self,
        clock: ReplayClock,
        *,
        on_wait: Callable[[], None] | None = None,
    ) -> None:
        """Initialize with a clock and an optional wait-side-effect.

        Args:
            clock: Clock advanced by every ``wait_for_timeout`` call.
            on_wait: Optional callback invoked after each clock tick.
                Used by tests that sequence world-state snapshots
                between waits (the callback advances the provider).
                Tests that need to wire the callback after the page
                already exists can set ``page.on_wait`` directly.
        """
        self._clock = clock
        self.on_wait = on_wait
        self._keyboard = _ReplayKeyboard()
        self.waits: list[float] = []

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Return the no-op keyboard."""
        return self._keyboard

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> ResponseProtocol | None:
        """Absorb a navigation request; never returns a real response."""
        _ = (url, referer, timeout, wait_until)
        return None

    def wait_for_timeout(self, timeout: float) -> None:
        """Advance the clock by ``timeout`` ms and run ``on_wait``."""
        self.waits.append(timeout)
        self._clock.advance(int(timeout))
        if self.on_wait is not None:
            self.on_wait()

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Absorb an event-wait request."""
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Absorb a function-wait request."""
        _ = (expression, timeout)

    def close(
        self,
        *,
        reason: str | None = None,
        run_before_unload: bool | None = None,
    ) -> None:
        """Absorb a close request."""
        _ = (reason, run_before_unload)

    def evaluate(self, expression: str) -> JSONValue:
        """Absorb an evaluate request; never returns a real value."""
        _ = expression
        return None


class ReplayPage:
    """Page substitute that feeds frames each ``wait_for_timeout`` call.

    The replay harness owns the frame stream and the clock. Each time
    the action-lab wait helpers call ``page.wait_for_timeout(ms)``:

    1. The clock advances by ``ms``.
    2. The next batch of recorded frames is appended to the probe's
       ``_cdp_message_buffer``.

    When the frame source is exhausted, subsequent waits still advance
    the clock -- this is how the wait helpers reach their timeout in a
    recorded session that ends before the requested outcome.

    Implements the full :class:`tankpit_bot._test_hooks.PageProtocol`
    surface so the harness can assign ``probe._page = ReplayPage(...)``
    without weakening the production type. The methods action-lab waits
    do not call are simple stubs.
    """

    def __init__(
        self,
        probe: BufferedMessageSourceProtocol,
        frame_source: FrameBatchSource,
        clock: ReplayClock,
    ) -> None:
        """Initialize the replay page.

        Args:
            probe: Probe whose ``_cdp_message_buffer`` receives frames.
            frame_source: Mutable cursor over the recorded frame stream.
            clock: Shared replay clock advanced on every wait.
        """
        self._probe = probe
        self._frame_source = frame_source
        self._clock = clock
        self._keyboard = _ReplayKeyboard()
        self._url = "https://tankpit.com/play"
        self.waits_ms: list[float] = []
        self.frames_fed: int = 0

    @property
    def url(self) -> str:
        """Return the current URL."""
        return self._url

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Return the no-op keyboard satisfying ``KeyboardProtocol``."""
        return self._keyboard

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> ResponseProtocol | None:
        """Absorb a navigation request without doing any real network IO."""
        _ = (referer, timeout, wait_until)
        self._url = url
        return None

    def wait_for_timeout(self, timeout: float) -> None:
        """Advance the clock and feed the next batch of frames.

        Args:
            timeout: Milliseconds to advance.
        """
        self.waits_ms.append(timeout)
        self._clock.advance(int(timeout))
        batch = self._frame_source.next_batch()
        if not batch:
            return
        self._probe._cdp_message_buffer.extend(batch)
        self.frames_fed += len(batch)

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Absorb an event-wait request without blocking."""
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Absorb a function-wait request without evaluating anything."""
        _ = (expression, timeout)

    def close(
        self,
        *,
        reason: str | None = None,
        run_before_unload: bool | None = None,
    ) -> None:
        """Absorb a close request."""
        _ = (reason, run_before_unload)

    def evaluate(self, expression: str) -> JSONValue:
        """Return ``None`` for any JS expression -- nothing to evaluate."""
        _ = expression
        return None


def build_world_derived_snapshot() -> JSONObject:
    """Build a page-client snapshot from the current global world state.

    No browser is running during replay. The :class:`WorldStateDerivedCDP`
    therefore answers snapshot queries with a deterministic projection of
    what :mod:`tankpit_bot.sniffer.world_state` already knows. Field
    semantics mirror the live snapshot but every value here is reachable
    from the production world-state singletons.

    Returns:
        JSON-shaped page-client snapshot payload.
    """
    world = get_world_state()
    self_state = world["self_state"]
    self_fields: dict[str, JSONValue] = {}
    if self_state is not None:
        self_fields["x"] = self_state["x"]
        self_fields["y"] = self_state["y"]
        self_fields["fuel"] = self_state["fuel"]
    return {
        "timestamp_ms": world["timestamp_ms"],
        "client_present": True,
        "map_visible": False,
        "client_state": 0,
        "client_busy": False,
        "pending_actions": 0,
        "heartbeat_age_ms": 0,
        "last_page_client_send_age_ms": 0,
        "last_bot_send_age_ms": 0,
        "ws_ready_state": 1,
        "current_send_label": None,
        "sent_frame_meta_queue_length": 0,
        "self_fields": self_fields,
        "world_fields": {},
        "map_fields": {},
        "world_collections": {},
    }


_DEFAULT_PAGE_CLIENT_SNAPSHOT_VALUE: JSONObject = {
    "timestamp_ms": 1000,
    "client_present": True,
    "map_visible": False,
    "client_state": 13,
    "client_busy": False,
    "pending_actions": 0,
    "heartbeat_age_ms": 10,
    "last_page_client_send_age_ms": 20,
    "last_bot_send_age_ms": 5,
    "ws_ready_state": 1,
    "current_send_label": None,
    "sent_frame_meta_queue_length": 0,
    "self_fields": {},
    "world_fields": {},
    "map_fields": {},
    "world_collections": {},
}
"""Canonical identity page-client snapshot for stub CDP sessions.

Matches the production :class:`PageClientSnapshotDict` shape. Tests
that need a CDP session purely for type-system reasons (the bootstrap
or attempt body is fully stubbed) wire :class:`StubSnapshotCDPSession`
with this value -- one snapshot constant, not nine.
"""


class StubSnapshotCDPSession:
    """CDPSessionProtocol that returns a fixed snapshot on ``Runtime.evaluate``.

    Other CDP methods return an empty dict; ``on`` and ``detach`` are
    no-ops. Used by tests whose probe bootstrap or attempt body is
    fully stubbed (the CDP session is only present to satisfy the
    type system, never read by production logic).

    When ``snapshot`` is omitted, returns the shared
    :data:`_DEFAULT_PAGE_CLIENT_SNAPSHOT_VALUE` -- one constant, not
    eight forked copies.
    """

    def __init__(self, snapshot: JSONObject | None = None) -> None:
        """Initialize with an optional snapshot value.

        Args:
            snapshot: ``Runtime.evaluate`` response payload. When
                ``None``, the shared identity snapshot is returned.
        """
        self._snapshot = snapshot if snapshot is not None else _DEFAULT_PAGE_CLIENT_SNAPSHOT_VALUE

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Return the canned snapshot for ``Runtime.evaluate``.

        Args:
            method: CDP method name.
            params: CDP method params (ignored).

        Returns:
            ``{"result": {"value": snapshot}}`` for ``Runtime.evaluate``;
            an empty dict for every other method.
        """
        _ = params
        if method == "Runtime.evaluate":
            return {"result": {"value": self._snapshot}}
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """No-op event handler registration."""
        _ = (event, handler)

    def detach(self) -> None:
        """No-op CDP session detach."""


class WorldStateDerivedCDP:
    """CDP substitute that derives ``Runtime.evaluate`` results from world state.

    The harness routes every CDP call through this class. Snapshot
    queries return a payload built from the *current* world-state
    singletons (see :func:`build_world_derived_snapshot`); all other
    CDP methods are no-ops. The CDP substitute carries no behavior of
    its own -- it is a pure projection of world-state truth.
    """

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Service a CDP command.

        Args:
            method: CDP method name (only ``Runtime.evaluate`` is honored).
            params: Optional method params.

        Returns:
            Snapshot payload for snapshot queries; string for WebSocket
            send evaluations; otherwise an empty evaluate response.
        """
        if method == "Runtime.evaluate" and params is not None:
            expression = params.get("expression", "")
            if isinstance(expression, str) and "ws.send" in expression:
                return {"result": {"value": "SENT_REPLAY_BYTES"}}
            return {"result": {"value": build_world_derived_snapshot()}}
        return {"result": {"value": None}}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """No-op event subscription."""
        _ = (event, handler)

    def detach(self) -> None:
        """No-op detach."""
        return


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
        from tankpit_bot.action_lab import _test_hooks as action_hooks
        from tankpit_bot.types import CapturedMessage

        self.cleanup_calls = 0

        def _stub_navigate(
            page: PageProtocol,
            cdp: CDPSessionProtocol,
            *,
            target_url: str,
            prefer_account: bool,
            tank_name_prefix: str = "TP",
            auto_join_room: bool = True,
        ) -> None:
            _ = (page, cdp, target_url, prefer_account, tank_name_prefix, auto_join_room)

        def _stub_wait_ready(
            page: PageProtocol,
            messages: list[CapturedMessage],
        ) -> None:
            _ = (page, messages)

        action_hooks.navigate_and_login = _stub_navigate
        action_hooks.wait_for_game_ready = _stub_wait_ready

    def _setup_console_listener(self, cdp: CDPSessionProtocol) -> None:
        """Skip real console-listener wiring."""
        _ = cdp

    def _setup_cdp_handlers(self, cdp: CDPSessionProtocol) -> None:
        """Skip real CDP-handler wiring."""
        _ = cdp

    def _gather_intel(self, page: PageProtocol, cdp: CDPSessionProtocol) -> None:
        """Skip real intel capture and set a stub magic key."""
        _ = (page, cdp)
        self._magic = "fake-magic"


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
    build_global_xor_table: Callable[[str], None],
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
        build_global_xor_table: Production ``build_global_xor_table``
            callable.

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
    build_global_xor_table(magic)

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


__all__ = [
    "ClockAdvancingPage",
    "DispatchCaptureMixin",
    "FrameBatchSource",
    "ReplayClock",
    "ReplayPage",
    "ReplayProbeContext",
    "ReplayResult",
    "StubSnapshotCDPSession",
    "StubbedBootstrapMixin",
    "WorldStateDerivedCDP",
    "WorldStateOverrideMixin",
    "build_world_derived_snapshot",
    "load_capture",
    "prepare_probe_replay",
    "received_payloads",
]
