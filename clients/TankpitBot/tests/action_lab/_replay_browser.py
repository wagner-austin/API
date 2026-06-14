"""Recorded Playwright bootstrap stack for execute_probe tests.

The probe ``execute_probe`` path runs through
:func:`tankpit_bot.action_lab.probe_runtime.execute_live_probe_bootstrap`,
which expects the entire Playwright Protocol chain
(``sync_playwright()`` -> ``Playwright`` -> ``BrowserType`` ->
``Browser`` -> ``BrowserContext`` -> ``Page`` + ``CDPSession``) to be
available. The test harness in
:mod:`tests.action_lab._replay_core` only wires the inner
``probe._page``/``probe._cdp`` handles after the attempt body has
already started -- it does not exercise the bootstrap itself.

This module fills that gap. A single :class:`RecordedChromiumSession`
yields a bootstrap-compatible Playwright stack so probe subclasses can
run ``execute_probe`` end-to-end against a captured WebSocket
recording. The Page is a :class:`ReplayPage` driving the captured
frames; the CDP session is a
:class:`WorldStateDerivedCDP` for snapshot ``Runtime.evaluate``
responses with an additional handler registry so production
``cdp.on(...)`` calls land on a structural protocol rather than the
attribute-error path.

Every protocol implementation below satisfies the corresponding
:mod:`tankpit_bot._test_hooks.browser` protocol -- mypy treats the
stack as a real Playwright surface, no casts and no thin protocol
wrappers.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from pathlib import Path

from platform_core.json_utils import JSONObject
from tests.action_lab._replay_core import (
    FrameBatchSource,
    ReplayClock,
    ReplayPage,
    WorldStateDerivedCDP,
    load_capture,
    received_payloads,
)

from tankpit_bot._test_hooks import (
    BrowserContextProtocol,
    BrowserProtocol,
    BrowserTypeProtocol,
    BufferedMessageSourceProtocol,
    CDPSessionProtocol,
    PageProtocol,
    PlaywrightProtocol,
    SyncPlaywrightContextManagerProtocol,
)
from tankpit_bot.sniffer.world_state import reset_world_state
from tankpit_bot.sniffer.xor import build_global_xor_table
from tankpit_bot.types import CaptureSession

__all__ = [
    "RecordedBrowser",
    "RecordedBrowserContext",
    "RecordedBrowserType",
    "RecordedChromiumSession",
    "RecordedPlaywright",
    "RecordedSyncPlaywrightContextManager",
    "RecordedSyncPlaywrightFactory",
]


class _HandlerRegistryCDP(WorldStateDerivedCDP):
    """CDP substitute that also records ``cdp.on`` registrations.

    The production probe bootstrap calls ``cdp.on(...)`` for
    ``Network.webSocketCreated``, ``Network.webSocketFrameReceived``,
    ``Network.webSocketFrameSent``, and ``Runtime.consoleAPICalled``.
    The default :class:`WorldStateDerivedCDP` discards these calls,
    which is correct for the action-lab attempt-body tests but loses
    the registered handlers in the execute_probe path where the
    harness has to push frames through them.

    This variant retains the handlers so a future call can replay
    captured frames through the real production handler graph.
    """

    def __init__(self) -> None:
        """Initialize an empty handler registry."""
        self._handlers: dict[str, list[Callable[[JSONObject], None]]] = {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Record an event handler registration.

        Args:
            event: CDP event name (e.g. ``"Network.webSocketFrameReceived"``).
            handler: Callback invoked with one event's params dict.
        """
        self._handlers.setdefault(event, []).append(handler)

    def emit(self, event: str, params: JSONObject) -> None:
        """Dispatch ``params`` to every handler registered for ``event``.

        Args:
            event: CDP event name to fire.
            params: Event payload forwarded to each registered handler.
        """
        for handler in self._handlers.get(event, ()):
            handler(params)


class RecordedBrowserContext:
    """Browser context that hands back the harness page and CDP session."""

    def __init__(
        self,
        page: PageProtocol,
        cdp: CDPSessionProtocol,
    ) -> None:
        """Initialize the recorded context.

        Args:
            page: Page substitute the bootstrap should adopt.
            cdp: CDP session substitute the bootstrap should adopt.
        """
        self._page = page
        self._cdp = cdp
        self.close_calls: int = 0

    def new_page(self) -> PageProtocol:
        """Return the harness page (one per context)."""
        return self._page

    def new_cdp_session(self, page: PageProtocol) -> CDPSessionProtocol:
        """Return the harness CDP session paired with the harness page.

        Args:
            page: Page the CDP session attaches to (must be the same
                page the context handed out).

        Returns:
            The harness CDP session.

        Raises:
            RuntimeError: If the caller passes a page the context did
                not produce -- catches accidental cross-wiring early.
        """
        if page is not self._page:
            raise RuntimeError(
                "RecordedBrowserContext.new_cdp_session received a page "
                "that the context did not produce"
            )
        return self._cdp

    def close(self, *, reason: str | None = None) -> None:
        """Record a context-close request without raising.

        Args:
            reason: Optional reason recorded by production cleanup.
        """
        _ = reason
        self.close_calls += 1


class RecordedBrowser:
    """Browser that yields the configured browser context."""

    def __init__(self, context: BrowserContextProtocol) -> None:
        """Initialize the recorded browser.

        Args:
            context: Browser context the bootstrap should adopt.
        """
        self._context = context
        self.close_calls: int = 0

    def new_context(self) -> BrowserContextProtocol:
        """Return the harness browser context."""
        return self._context

    def close(self, *, reason: str | None = None) -> None:
        """Record a browser-close request without raising.

        Args:
            reason: Optional reason recorded by production cleanup.
        """
        _ = reason
        self.close_calls += 1


class RecordedBrowserType:
    """BrowserType that produces the configured browser on ``launch``."""

    def __init__(self, browser: BrowserProtocol) -> None:
        """Initialize the recorded browser type.

        Args:
            browser: Browser the bootstrap should adopt.
        """
        self._browser = browser
        self.launches: list[bool | None] = []

    def launch(
        self,
        *,
        headless: bool | None = None,
        slow_mo: float | None = None,
        timeout: float | None = None,
    ) -> BrowserProtocol:
        """Record the launch request and return the harness browser.

        Args:
            headless: Recorded for assertion purposes.
            slow_mo: Ignored (production uses defaults).
            timeout: Ignored (production uses defaults).

        Returns:
            The harness browser.
        """
        _ = (slow_mo, timeout)
        self.launches.append(headless)
        return self._browser


class RecordedPlaywright:
    """Playwright instance exposing the recorded chromium browser type."""

    def __init__(self, browser_type: BrowserTypeProtocol) -> None:
        """Initialize the recorded Playwright instance.

        Args:
            browser_type: Browser type returned via ``chromium``.
        """
        self._chromium = browser_type
        self.stop_calls: int = 0

    @property
    def chromium(self) -> BrowserTypeProtocol:
        """Return the harness browser type."""
        return self._chromium

    def stop(self) -> None:
        """Record a Playwright-stop request."""
        self.stop_calls += 1


class RecordedSyncPlaywrightContextManager:
    """Context manager that yields the recorded Playwright instance."""

    def __init__(self, playwright: PlaywrightProtocol) -> None:
        """Initialize the recorded context manager.

        Args:
            playwright: Playwright instance to yield.
        """
        self._playwright = playwright

    def start(self) -> PlaywrightProtocol:
        """Return the recorded Playwright instance."""
        return self._playwright

    def __enter__(self) -> PlaywrightProtocol:
        """Yield the recorded Playwright instance to the ``with`` block."""
        return self._playwright

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit the context without suppressing exceptions.

        Args:
            exc_type: Propagated by the runtime; ignored.
            exc_val: Propagated by the runtime; ignored.
            exc_tb: Propagated by the runtime; ignored.
        """
        _ = (exc_type, exc_val, exc_tb)


class RecordedSyncPlaywrightFactory:
    """Factory that produces a context manager for one bootstrap call.

    Satisfies
    :class:`tankpit_bot._test_hooks.SyncPlaywrightFactoryProtocol` so it
    can be assigned to ``_test_hooks.sync_playwright`` for the duration
    of a test.
    """

    def __init__(self, manager: SyncPlaywrightContextManagerProtocol) -> None:
        """Initialize the recorded factory.

        Args:
            manager: Context manager returned on every call.
        """
        self._manager = manager
        self.calls: int = 0

    def __call__(self) -> SyncPlaywrightContextManagerProtocol:
        """Return the recorded context manager and record the call."""
        self.calls += 1
        return self._manager


class RecordedChromiumSession:
    """End-to-end recorded Playwright stack for one execute_probe run.

    Constructed once per test; the
    :attr:`sync_playwright_factory` attribute is the entry point
    test code assigns to ``_test_hooks.sync_playwright``. Every layer
    of the bootstrap chain (``Playwright``, ``BrowserType``,
    ``Browser``, ``BrowserContext``, ``Page``, ``CDPSession``) is
    materialized once and shared across the entire ``with`` block --
    matching production semantics, no per-call rebuilds.

    The session is anchored to a captured WebSocket recording: the
    capture's magic key bootstraps the global XOR table on
    construction, and the recorded received-payloads feed into the
    page substitute's frame source so the replay timeline matches the
    live game.
    """

    def __init__(
        self,
        probe: BufferedMessageSourceProtocol,
        capture: CaptureSession,
        *,
        frames_per_wait: int = 5,
    ) -> None:
        """Initialize the recorded session against a captured run.

        Args:
            probe: Probe instance whose ``_cdp_message_buffer`` the
                page substitute should populate on each
                ``wait_for_timeout`` call.
            capture: Loaded capture session whose magic key and
                received payloads drive the bootstrap.
            frames_per_wait: Number of recorded frames fed into the
                probe's CDP buffer on each ``wait_for_timeout`` poll.

        Raises:
            RuntimeError: If the capture has no magic key.
        """
        magic = capture["magic"]
        if magic is None:
            raise RuntimeError("capture has no magic key; cannot bootstrap XOR table")
        reset_world_state()
        build_global_xor_table(magic)

        self.clock = ReplayClock()
        self.frame_source = FrameBatchSource(received_payloads(capture), frames_per_wait)
        self.page = ReplayPage(probe, self.frame_source, self.clock)
        self.cdp = _HandlerRegistryCDP()
        self.context = RecordedBrowserContext(self.page, self.cdp)
        self.browser = RecordedBrowser(self.context)
        self.browser_type = RecordedBrowserType(self.browser)
        self.playwright = RecordedPlaywright(self.browser_type)
        self.manager = RecordedSyncPlaywrightContextManager(self.playwright)
        self.sync_playwright_factory = RecordedSyncPlaywrightFactory(self.manager)

    @classmethod
    def from_capture_path(
        cls,
        probe: BufferedMessageSourceProtocol,
        capture_path: Path,
        *,
        frames_per_wait: int = 5,
    ) -> RecordedChromiumSession:
        """Build a recorded session from a capture file on disk.

        Args:
            probe: Probe instance whose ``_cdp_message_buffer`` the
                page substitute should populate.
            capture_path: Path to a ``*.capture_session.json``.
            frames_per_wait: Frames fed per poll.

        Returns:
            Recorded session bound to the loaded capture.
        """
        return cls(probe, load_capture(capture_path), frames_per_wait=frames_per_wait)
