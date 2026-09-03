"""Serialised single-session runner for the long-running bot service.

The service main starts once; the SPA fires ``POST /start`` when it
wants a game session, ``POST /stop`` when it wants that session to
end, and can drive several sessions during the process lifetime. This
class owns exactly one session's lifecycle at a time — it refuses to
start a second session while the first is still running and it holds
the ``ModeBridgeProtocol`` / ``StatusBusProtocol`` handles the HTTP
thread has to share with the tick loop.

Stop signalling reuses the existing stop-file channel :func:`Bot.run`
already polls: the runner writes the sentinel from the aiohttp
thread; the tick loop sees it and exits at the next boundary. That
keeps the cross-thread contract identical to today's ``make bot``
graceful-shutdown, so the tick loop needs no new signalling code.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal, Protocol

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot.browser.cdp_utils import get_current_time_ms
from tankpit_bot.bus.frame_bus import FrameBusProtocol
from tankpit_bot.bus.mode_bridge import ModeBridgeProtocol
from tankpit_bot.bus.session_status import idle_session_status
from tankpit_bot.bus.status_bus import StatusBusProtocol
from tankpit_bot.runtime_logging import configure_bot_runtime_logging

log = get_logger(__name__)

SessionRunnerState = Literal["idle", "running", "stopping"]


class BotFactoryProtocol(Protocol):
    """Factory callable that produces the Bot for one session.

    The runner keeps :class:`Bot` construction behind a callable so
    tests can inject a fake bot without wiring up Playwright. Every
    factory invocation must return a fresh instance — the runner
    calls ``run`` on it directly.
    """

    def __call__(
        self,
        *,
        mode_bridge: ModeBridgeProtocol,
        status_bus: StatusBusProtocol,
        frame_bus: FrameBusProtocol,
    ) -> RunnableBotProtocol:
        """Construct a bot bound to the shared bridge + buses.

        Args:
            mode_bridge: Cross-thread mode override channel.
            status_bus: Fan-out for :class:`SessionStatusDict` frames.
            frame_bus: Fan-out for screencast JPEG frames; its
                subscriber count is the tick loop's screencast demand
                signal.

        Returns:
            A bot whose :meth:`RunnableBotProtocol.run` will block the
            calling thread for the session's lifetime.
        """
        ...


class RunnableBotProtocol(Protocol):
    """Minimal surface :class:`SessionRunner` needs from :class:`Bot`."""

    def run(
        self,
        *,
        session_seconds: int,
        session_kills: int = 0,
        stop_file_path: Path,
    ) -> None:
        """Run one session until a bound trips or the stop file appears.

        Args:
            session_seconds: Bounded session length in seconds; zero or
                negative runs until externally stopped.
            session_kills: Kill-count bound; zero or negative means
                unbounded.
            stop_file_path: Sentinel file whose existence requests a
                graceful shutdown.
        """
        ...


class SessionAlreadyRunningError(RuntimeError):
    """Raised when :meth:`SessionRunner.start` is called mid-session.

    Signals the HTTP layer to return ``409 Conflict`` — a second
    ``POST /start`` cannot start a session while the first is still
    holding Playwright.
    """


class SessionRunner:
    """Coordinator for one active game session at a time.

    Thread model:

    * :meth:`start` runs on the service main thread. It transitions
      state to ``"running"``, invokes the bot factory, and calls
      :meth:`RunnableBotProtocol.run` — blocking until the session
      ends. On return it transitions back to ``"idle"`` and publishes
      one final :func:`idle_session_status` frame so the SPA sees
      "session ended."
    * :meth:`request_stop` runs on the aiohttp handler thread. It
      writes the stop-file sentinel; the tick loop sees it at the
      next boundary and unwinds.
    * :meth:`state` is thread-safe and cheap — the ``GET /status``
      SSE handler can call it every heartbeat without lock contention.
    """

    def __init__(
        self,
        *,
        bot_factory: BotFactoryProtocol,
        mode_bridge: ModeBridgeProtocol,
        status_bus: StatusBusProtocol,
        frame_bus: FrameBusProtocol,
        stop_file_path: Path,
    ) -> None:
        """Bind the runner to its cross-thread channels.

        Args:
            bot_factory: Callable that produces a fresh
                :class:`RunnableBotProtocol` per session. In
                production this is a lambda that constructs
                :class:`tankpit_bot.bot.base.Bot`; tests pass a
                factory that returns a fake bot.
            mode_bridge: Cross-thread mode override channel shared
                with the HTTP handler.
            status_bus: Fan-out for :class:`SessionStatusDict` frames
                shared with the SSE handler.
            frame_bus: Fan-out for screencast JPEG frames shared with
                the ``/video`` / ``/frame`` handlers.
            stop_file_path: Sentinel file the runner writes when the
                SPA requests a stop; the tick loop polls its
                existence.
        """
        self._bot_factory = bot_factory
        self._mode_bridge = mode_bridge
        self._status_bus = status_bus
        self._frame_bus = frame_bus
        self._stop_file_path = stop_file_path
        self._state_lock = threading.Lock()
        self._state: SessionRunnerState = "idle"

    def state(self) -> SessionRunnerState:
        """Return the current lifecycle state.

        Returns:
            ``"idle"`` when no session is running, ``"running"`` while
            :meth:`start` is executing, ``"stopping"`` while a stop
            has been requested but the tick loop has not yet exited.
        """
        with self._state_lock:
            return self._state

    def is_running(self) -> bool:
        """Return True when a session is currently active.

        Returns:
            True while the state is ``"running"`` or ``"stopping"``.
        """
        with self._state_lock:
            return self._state != "idle"

    def start(self, *, session_seconds: int = 0, session_kills: int = 0) -> None:
        """Run one session start-to-finish on the calling thread.

        Blocks until the tick loop exits (via stop file, tick budget,
        kill bound, or session error). Rejects a second concurrent
        call with :class:`SessionAlreadyRunningError` so the HTTP
        layer can translate to ``409 Conflict``.

        Args:
            session_seconds: Bounded session length in seconds; zero
                runs until externally stopped (the phone flow's
                default — ``POST /start`` with no body).
            session_kills: Kill-count bound; zero means unbounded.

        Raises:
            SessionAlreadyRunningError: A session is already running
                or stopping.
        """
        with self._state_lock:
            if self._state != "idle":
                raise SessionAlreadyRunningError(f"cannot start: SessionRunner is {self._state!r}")
            self._state = "running"

        try:
            self._clear_stop_file()
            # Bootstrap the Playwright loader hook exactly like the
            # ``make run`` entry point (``bot/entry.py``) and the
            # sniffer do. ``sync_playwright`` is None by design until
            # a bootstrap assigns it; ``Bot.run`` reads the slot
            # directly and raises ``PlaywrightNotInstalledError`` on
            # None. The service path was the ONLY session entry point
            # missing this — every phone-driven START BOT died
            # instantly, and the unawaited executor future swallowed
            # the traceback (found 2026-07-19; the crash was invisible
            # until the diag reproduced it in the foreground).
            if _test_hooks.sync_playwright is None:
                _test_hooks.sync_playwright = _test_hooks.get_sync_playwright()
            # NO IDLE PIN. A service session starts in auto-arbitration,
            # exactly like ``make run``, because nothing left in this
            # system can release a pin.
            #
            # It used to submit ``"UNSET"`` here (2026-07-18, at Austin's
            # request): phone-driven sessions started pinned to idle so
            # the operator could see where the tank spawned and then
            # release it from the fiesta SPA's bot overlay. That overlay
            # was deleted on 2026-09-03 when tankpit was decoupled from
            # fiesta, and the fleet runs THIS entry point for every
            # child — so the pin outlived the only UI that could lift
            # it. The visible result was a bot that logged in, entered
            # the game, and then held position forever, reporting
            # ``reason=manual_hold`` on every tick.
            #
            # Pinning is still reachable and still deliberate: an
            # operator POSTs ``/mode`` (UNSET / HUNT / COLLECT) through
            # the same bridge. What is gone is a default that waits for
            # somebody who is not coming.
            # Configure the per-session run artifacts (archive log,
            # events.jsonl, index row) exactly like the ``make run``
            # entry point does. Until 2026-07-28 the service path
            # skipped this, so every phone-driven session ran with
            # UNCONFIGURED logging: INFO lines vanished, no run
            # archive existed, and the scorecard never reached
            # ``_index.tsv`` — discovered when a service session's
            # screencast log lines were nowhere on disk.
            artifacts = configure_bot_runtime_logging()
            log.info("Session artifacts: %s", artifacts["archive_log_path"])
            bot = self._bot_factory(
                mode_bridge=self._mode_bridge,
                status_bus=self._status_bus,
                frame_bus=self._frame_bus,
            )
            log.info("Session start: running bot")
            bot.run(
                session_seconds=session_seconds,
                session_kills=session_kills,
                stop_file_path=self._stop_file_path,
            )
            log.info("Session end: bot.run returned")
        finally:
            with self._state_lock:
                self._state = "idle"
            self._publish_idle_status()

    def request_stop(self) -> None:
        """Request the running session end at the next tick boundary.

        Writes the stop-file sentinel and transitions state from
        ``"running"`` to ``"stopping"``. Idempotent — calling from
        an idle state or during a stop-in-progress is a no-op.
        """
        with self._state_lock:
            if self._state == "idle":
                log.info("Stop requested while idle; ignoring")
                return
            self._state = "stopping"
        _test_hooks.write_text(self._stop_file_path, "")
        log.info("Stop requested: wrote %s", self._stop_file_path)

    def _clear_stop_file(self) -> None:
        """Remove any stale stop-file from a previous session.

        A stop-file left behind by an earlier session (unclean shutdown
        or test) would cause the next :meth:`start` to exit on its
        very first tick. Clearing it here mirrors the same discipline
        :func:`entry.main` uses at bot boot.
        """
        if _test_hooks.path_exists(self._stop_file_path):
            _test_hooks.remove_file(self._stop_file_path)

    def _publish_idle_status(self) -> None:
        """Emit one ``running=False`` frame so subscribers see session end."""
        self._status_bus.publish(idle_session_status(get_current_time_ms()))


__all__ = [
    "BotFactoryProtocol",
    "RunnableBotProtocol",
    "SessionAlreadyRunningError",
    "SessionRunner",
    "SessionRunnerState",
]
