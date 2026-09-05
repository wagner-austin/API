"""Tests for :class:`tankpit_bot.service.session_runner.SessionRunner`.

Covers the state machine, the start/stop happy path, the
already-running rejection, the stale-stop-file scrub, and the idle
status publish that closes every session. No mocks — a small
``_RecordingBot`` fake implements the runnable-bot Protocol
structurally and captures the run() invocation, and the standard
``fake_fs`` fixture backs the stop-file writes.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks.browser import (
    SyncPlaywrightContextManagerProtocol,
    SyncPlaywrightFactoryProtocol,
)
from tankpit_bot.bus.mode_bridge import ModeBridge, ModeBridgeProtocol
from tankpit_bot.bus.status_bus import StatusBus, StatusBusProtocol
from tankpit_bot.service.session_runner import (
    RunnableBotProtocol,
    SessionAlreadyRunningError,
    SessionRunner,
)
from tests.conftest import FakeFileSystem

_STOP_FILE = Path("runs/bot/STOP")


@pytest.fixture()
def stop_file_hooks(fake_fs: FakeFileSystem) -> Generator[FakeFileSystem, None, None]:
    """Wire ``_test_hooks.remove_file`` to ``fake_fs.remove`` for the test.

    The stock ``fake_fs`` fixture installs the read/write/path-exists
    hooks but not ``remove_file`` — the tick loop and session runner
    both delete the stop-file sentinel, so tests exercising those
    paths need the extra hook.

    Yields:
        The underlying :class:`FakeFileSystem` for direct assertions.
    """
    original_remove_file = _test_hooks.remove_file
    _test_hooks.remove_file = fake_fs.remove
    try:
        yield fake_fs
    finally:
        _test_hooks.remove_file = original_remove_file


class _RecordingBot:
    """Minimal :class:`RunnableBotProtocol` for the runner tests."""

    def __init__(
        self,
        *,
        on_run: Callable[[], None] | None = None,
    ) -> None:
        """Track invocations of :meth:`run` for later assertions.

        Args:
            on_run: Optional side-effect callable executed inside
                ``run``. Used by tests that want the bot to touch the
                stop-file (or raise) to simulate the tick loop
                observing an external stop request.
        """
        self.calls: list[tuple[int, Path]] = []
        self._on_run = on_run

    def run(
        self,
        *,
        session_seconds: int,
        session_kills: int = 0,
        stop_file_path: Path,
    ) -> None:
        """Record the arguments and invoke the optional side-effect."""
        self.calls.append((session_seconds, stop_file_path))
        if self._on_run is not None:
            self._on_run()


class _NeverStartedSyncPlaywright:
    """``SyncPlaywrightFactoryProtocol`` fake for slot-identity tests.

    The bootstrap tests only assert WHICH object sits in the loader
    slot; nothing may actually launch Playwright, so calling the
    factory is a test bug by definition.
    """

    def __call__(self) -> SyncPlaywrightContextManagerProtocol:
        raise AssertionError("the bootstrap tests must never invoke the factory")


def _make_runner(
    bot: RunnableBotProtocol,
    *,
    bridge: ModeBridgeProtocol | None = None,
    bus: StatusBusProtocol | None = None,
) -> SessionRunner:
    """Assemble a runner bound to ``bot`` and the given (or fresh) channels."""
    used_bridge = bridge if bridge is not None else ModeBridge()
    used_bus = bus if bus is not None else StatusBus()

    def factory(
        *,
        mode_bridge: ModeBridgeProtocol,
        status_bus: StatusBusProtocol,
    ) -> RunnableBotProtocol:
        _ = (mode_bridge, status_bus)
        return bot

    return SessionRunner(
        bot_factory=factory,
        mode_bridge=used_bridge,
        status_bus=used_bus,
        stop_file_path=_STOP_FILE,
    )


class TestSessionRunnerStateMachine:
    """State transitions across the runner's lifecycle."""

    def test_initial_state_is_idle(self, fake_fs: FakeFileSystem) -> None:
        """A freshly constructed runner reports idle and not-running."""
        _ = fake_fs
        runner = _make_runner(_RecordingBot())
        assert runner.state() == "idle"
        assert runner.is_running() is False

    def test_start_calls_the_factory_and_runs_the_bot(self, fake_fs: FakeFileSystem) -> None:
        """The runner constructs a bot via the factory and calls ``run``."""
        _ = fake_fs
        bot = _RecordingBot()
        runner = _make_runner(bot)

        runner.start()

        assert bot.calls == [(0, _STOP_FILE)]

    def test_start_bootstraps_the_playwright_loader_hook(self, fake_fs: FakeFileSystem) -> None:
        """A None ``sync_playwright`` slot is populated via the loader.

        The service path missed this bootstrap while ``make run`` and
        the sniffer had it — every phone-driven START BOT raised
        ``PlaywrightNotInstalledError`` before its run log existed
        (2026-07-19). The runner now mirrors ``bot/entry.py``.
        """
        _ = fake_fs
        original_slot = _test_hooks.sync_playwright
        original_loader = _test_hooks.get_sync_playwright
        sentinel = _NeverStartedSyncPlaywright()

        def fake_loader() -> SyncPlaywrightFactoryProtocol:
            return sentinel

        try:
            _test_hooks.sync_playwright = None
            _test_hooks.get_sync_playwright = fake_loader
            _make_runner(_RecordingBot()).start()
            assert _test_hooks.sync_playwright is sentinel
        finally:
            _test_hooks.sync_playwright = original_slot
            _test_hooks.get_sync_playwright = original_loader

    def test_start_leaves_an_injected_playwright_fake_alone(self, fake_fs: FakeFileSystem) -> None:
        """A non-None slot (test-injected fake) is never clobbered."""
        _ = fake_fs
        original_slot = _test_hooks.sync_playwright
        injected = _NeverStartedSyncPlaywright()
        try:
            _test_hooks.sync_playwright = injected
            _make_runner(_RecordingBot()).start()
            assert _test_hooks.sync_playwright is injected
        finally:
            _test_hooks.sync_playwright = original_slot

    def test_start_queues_no_mode_so_the_bot_arbitrates_from_the_first_tick(
        self, fake_fs: FakeFileSystem
    ) -> None:
        """A service session starts PLAYING, not pinned to idle.

        The runner used to submit ``"UNSET"`` here so the first tick
        drained it into ``manual_mode = "UNSET"`` and the bot held
        position until an operator released it from the fiesta SPA's bot
        overlay. That overlay was deleted 2026-09-03, and the fleet runs
        this entry point for every child — so the pin outlived the only
        UI that could lift it and every fleet bot sat in the game doing
        nothing, logging ``reason=manual_hold`` forever.

        An empty bridge is the whole assertion: ``_apply_pending_mode_override``
        leaves ``manual_mode`` alone when the drain comes back empty, so
        the session keeps the ``None`` default and auto-arbitrates
        exactly like ``make run``.
        """
        _ = fake_fs
        bridge = ModeBridge()
        runner = _make_runner(_RecordingBot(), bridge=bridge)

        runner.start()

        assert bridge.drain() is None

    def test_start_returns_to_idle_after_bot_run_completes(self, fake_fs: FakeFileSystem) -> None:
        """After ``bot.run`` returns the runner is idle again."""
        _ = fake_fs
        runner = _make_runner(_RecordingBot())

        runner.start()

        assert runner.state() == "idle"
        assert runner.is_running() is False

    def test_start_publishes_idle_status_at_the_end(self, fake_fs: FakeFileSystem) -> None:
        """A session-end frame reaches every subscriber."""
        _ = fake_fs
        bus = StatusBus()
        subscriber = bus.subscribe()
        runner = _make_runner(_RecordingBot(), bus=bus)

        runner.start()

        frame = subscriber.next_frame(timeout=0.5)
        if frame is None:
            raise AssertionError("expected an idle status frame after session end")
        assert frame["running"] is False
        assert frame["active_mode"] == "UNSET"

    def test_start_returns_to_idle_when_bot_raises(self, fake_fs: FakeFileSystem) -> None:
        """A crash in ``bot.run`` still leaves the runner idle.

        The runner must not stay stuck in ``"running"`` after the tick
        loop dies — otherwise the next ``POST /start`` would forever
        return 409.
        """
        _ = fake_fs

        def blow_up() -> None:
            raise RuntimeError("simulated tick-loop crash")

        bot = _RecordingBot(on_run=blow_up)
        runner = _make_runner(bot)

        with pytest.raises(RuntimeError, match="simulated tick-loop crash"):
            runner.start()

        assert runner.state() == "idle"

    def test_second_start_while_running_raises(self, fake_fs: FakeFileSystem) -> None:
        """A concurrent ``start`` mid-session is rejected as 409-worthy."""
        _ = fake_fs
        rejection_messages: list[str] = []

        def try_concurrent_start() -> None:
            # pytest.raises works on any thread: if start() does NOT
            # raise, the Failed escapes into this worker thread and
            # rejection_messages stays empty, failing the main-thread
            # assertion below — same detection, no silent except.
            with pytest.raises(SessionAlreadyRunningError) as excinfo:
                runner.start()
            rejection_messages.append(str(excinfo.value))

        def block_until_second_start_attempted() -> None:
            import threading as _threading
            import time as _time

            waiter = _threading.Thread(target=try_concurrent_start)
            waiter.start()
            # Give the concurrent thread a moment to hit the state
            # check + raise. Without a barrier this can race, but
            # 50ms is plenty for a lock acquisition + raise.
            _time.sleep(0.05)
            waiter.join(timeout=1.0)

        bot = _RecordingBot(on_run=block_until_second_start_attempted)
        runner = _make_runner(bot)

        runner.start()

        assert len(rejection_messages) == 1
        assert "cannot start" in rejection_messages[0]
        assert "running" in rejection_messages[0]

    def test_start_reusable_after_a_session_ends(self, fake_fs: FakeFileSystem) -> None:
        """After a session completes, ``start`` accepts the next request."""
        _ = fake_fs
        bot = _RecordingBot()
        runner = _make_runner(bot)

        runner.start()
        runner.start()

        assert len(bot.calls) == 2


class TestSessionRunnerRequestStop:
    """The ``request_stop`` cross-thread signal."""

    def test_stop_while_running_writes_sentinel_and_flips_to_stopping(
        self, fake_fs: FakeFileSystem
    ) -> None:
        """A stop request during ``run`` writes the stop file."""
        captured_state: list[str] = []
        captured_written: list[bool] = []

        def stop_mid_run() -> None:
            runner.request_stop()
            captured_state.append(runner.state())
            captured_written.append(str(_STOP_FILE) in fake_fs.get_written_files())

        bot = _RecordingBot(on_run=stop_mid_run)
        runner = _make_runner(bot)

        runner.start()

        assert captured_state == ["stopping"]
        assert captured_written == [True]

    def test_stop_from_idle_is_a_noop(self, fake_fs: FakeFileSystem) -> None:
        """A stop request with no session running does not touch the disk."""
        runner = _make_runner(_RecordingBot())

        runner.request_stop()

        assert runner.state() == "idle"
        assert str(_STOP_FILE) not in fake_fs.get_written_files()


class TestSessionRunnerStaleStopFile:
    """The stop-file scrub that guards the next start."""

    def test_stale_stop_file_is_removed_before_bot_run(
        self, stop_file_hooks: FakeFileSystem
    ) -> None:
        """A leftover stop file does not survive into the next session."""
        stop_file_hooks.write_text(_STOP_FILE, "")
        observations: list[bool] = []

        def observe_when_bot_runs() -> None:
            # ``run`` is called AFTER ``_clear_stop_file``; the file
            # must be gone by now so the tick loop does not exit
            # instantly on its first path_exists poll.
            observations.append(str(_STOP_FILE) in stop_file_hooks.get_written_files())

        bot = _RecordingBot(on_run=observe_when_bot_runs)
        runner = _make_runner(bot)

        runner.start()

        assert observations == [False]


class TestSessionAlreadyRunningError:
    """The rejection error type contract."""

    def test_error_is_runtime_error_subclass(self) -> None:
        """The rejection error is a :class:`RuntimeError` for broad catchers."""
        assert issubclass(SessionAlreadyRunningError, RuntimeError)


class TestSessionRunnerConfiguresRunArtifacts:
    """Service sessions get the same run artifacts as ``make run``.

    Until 2026-07-28 the service path never called
    ``configure_bot_runtime_logging``, so phone-driven sessions ran
    with unconfigured logging — INFO lines vanished, no archive log or
    events file existed, and the session scorecard never reached
    ``_index.tsv``. Discovered when a live service session's
    screencast lifecycle lines were nowhere on disk.
    """

    def test_start_configures_bot_runtime_logging(self, fake_fs: FakeFileSystem) -> None:
        """``start`` installs the per-session bot artifact bundle."""
        from tankpit_bot.runtime_logging import get_bot_runtime_artifacts

        runner = _make_runner(_RecordingBot())

        runner.start()

        artifacts = get_bot_runtime_artifacts()
        if artifacts is None:
            raise AssertionError("start() must configure the bot runtime artifacts")
        assert Path(artifacts["latest_log_path"]) == Path("runs/bot/latest.log")
        assert Path(artifacts["latest_events_path"]) == Path("runs/bot/latest.events.jsonl")
        # The artifact text handler mirrors session log lines into the
        # latest log through the filesystem hooks — proof the configure
        # ran inside ``start`` BEFORE the session logging began.
        written = fake_fs.get_written_files()
        latest_log = written[str(Path(artifacts["latest_log_path"]))]
        assert "Session artifacts:" in latest_log
        assert "Session start: running bot" in latest_log
        assert "Session end: bot.run returned" in latest_log


class TestBotFactoryReceivesSharedChannels:
    """The factory sees the exact bridge / bus the runner owns."""

    def test_factory_gets_the_runners_channels(self, fake_fs: FakeFileSystem) -> None:
        """The factory kwargs are the runner's channels, not copies."""
        _ = fake_fs
        bridge = ModeBridge()
        bus = StatusBus()
        received: list[tuple[ModeBridgeProtocol, StatusBusProtocol]] = []

        def factory(
            *,
            mode_bridge: ModeBridgeProtocol,
            status_bus: StatusBusProtocol,
        ) -> RunnableBotProtocol:
            received.append((mode_bridge, status_bus))
            return _RecordingBot()

        runner = SessionRunner(
            bot_factory=factory,
            mode_bridge=bridge,
            status_bus=bus,
            stop_file_path=_STOP_FILE,
        )

        runner.start()

        assert len(received) == 1
        assert received[0][0] is bridge
        assert received[0][1] is bus
