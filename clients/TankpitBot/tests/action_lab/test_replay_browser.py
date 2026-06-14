"""Contract tests for the recorded Playwright bootstrap stack.

Pins the seams of :mod:`tests.action_lab._replay_browser` so the
bootstrap-compatible substitutes stay aligned with the real
Playwright Protocol surface that
:func:`tankpit_bot.action_lab.probe_runtime.execute_live_probe_bootstrap`
consumes.

The execute_probe pipeline tests in test_*_probe.py exercise the
substitutes end-to-end; these tests pin the unit-level behavior
(handler registry, factory plumbing, close accounting, frame replay
through registered handlers) so a regression in any single seam fails
loudly here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject
from tests.action_lab._replay_browser import RecordedChromiumSession

from tankpit_bot._test_hooks import (
    BrowserContextProtocol,
    BrowserProtocol,
    BrowserTypeProtocol,
    CDPSessionProtocol,
    PlaywrightProtocol,
    SyncPlaywrightContextManagerProtocol,
    SyncPlaywrightFactoryProtocol,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FUEL_CAPTURE = REPO_ROOT / "fuel_probe.capture_session.json"


class _BufferProbe:
    """Minimal probe substitute that exposes ``_cdp_message_buffer``.

    Lets the contract tests drive a RecordedChromiumSession without
    pulling in a real :class:`FuelProbe`/etc -- only the buffer
    protocol matters here.
    """

    def __init__(self) -> None:
        """Initialize an empty message buffer."""
        self._cdp_message_buffer: list[str] = []


def _make_session() -> RecordedChromiumSession:
    """Build a session bound to the committed fuel-probe capture.

    Returns:
        A :class:`RecordedChromiumSession` bootstrapped against the
        committed ``fuel_probe.capture_session.json``.
    """
    probe = _BufferProbe()
    return RecordedChromiumSession.from_capture_path(probe, FUEL_CAPTURE)


def test_factory_is_callable_and_returns_context_manager() -> None:
    """The factory satisfies ``SyncPlaywrightFactoryProtocol``.

    Calls the factory and uses the returned object as a context
    manager -- if it isn't really CM-shaped, the ``with`` block
    fails. Stronger than an isinstance check.
    """
    session = _make_session()
    factory: SyncPlaywrightFactoryProtocol = session.sync_playwright_factory

    with factory() as playwright:
        assert playwright is session.playwright
    assert session.sync_playwright_factory.calls == 1


def test_context_manager_yields_same_playwright_via_enter_and_start() -> None:
    """The context manager surface is stable across ``__enter__`` and ``start``."""
    session = _make_session()
    manager: SyncPlaywrightContextManagerProtocol = session.manager

    with manager as via_enter:
        assert via_enter is session.playwright
    assert manager.start() is session.playwright


def test_playwright_chromium_returns_recorded_browser_type() -> None:
    """``playwright.chromium`` returns the recorded browser type."""
    session = _make_session()
    playwright: PlaywrightProtocol = session.playwright

    browser_type = playwright.chromium

    assert browser_type is session.browser_type


def test_playwright_stop_is_recorded() -> None:
    """``playwright.stop`` increments ``stop_calls`` for assertions."""
    session = _make_session()

    session.playwright.stop()
    session.playwright.stop()

    assert session.playwright.stop_calls == 2


def test_browser_type_launch_records_headless_choice() -> None:
    """``BrowserType.launch`` records ``headless`` so tests can assert it."""
    session = _make_session()
    browser_type: BrowserTypeProtocol = session.browser_type

    returned = browser_type.launch(headless=False, slow_mo=10, timeout=1000)

    assert returned is session.browser
    assert session.browser_type.launches == [False]


def test_browser_new_context_returns_recorded_context() -> None:
    """``Browser.new_context`` returns the recorded context."""
    session = _make_session()
    browser: BrowserProtocol = session.browser

    context = browser.new_context()

    assert context is session.context


def test_browser_close_records_calls_without_raising() -> None:
    """``Browser.close`` accepts ``reason`` and records each call."""
    session = _make_session()

    session.browser.close()
    session.browser.close(reason="end of test")

    assert session.browser.close_calls == 2


def test_context_new_page_and_new_cdp_session_pairs_correctly() -> None:
    """``new_page`` and ``new_cdp_session`` return the harness page + CDP."""
    session = _make_session()
    context: BrowserContextProtocol = session.context

    page = context.new_page()
    cdp = context.new_cdp_session(page)

    assert page is session.page
    assert cdp is session.cdp


def test_context_new_cdp_session_rejects_unknown_page() -> None:
    """The CDP session raises if asked to attach to a foreign page."""
    session = _make_session()
    other_session = _make_session()

    with pytest.raises(RuntimeError, match="did not produce"):
        session.context.new_cdp_session(other_session.page)


def test_context_close_records_calls_without_raising() -> None:
    """``BrowserContext.close`` accepts ``reason`` and records each call."""
    session = _make_session()

    session.context.close()
    session.context.close(reason="end of test")

    assert session.context.close_calls == 2


def test_cdp_records_handler_registrations_and_replays_them() -> None:
    """The CDP handler registry replays events through registered handlers."""
    session = _make_session()
    received: list[JSONObject] = []

    def _handler(params: JSONObject) -> None:
        """Capture each replayed event for assertion."""
        received.append(params)

    cdp: CDPSessionProtocol = session.cdp
    cdp.on("Network.webSocketFrameReceived", _handler)

    payload: JSONObject = {"requestId": "rid", "timestamp": 12.5}
    session.cdp.emit("Network.webSocketFrameReceived", payload)

    assert received == [payload]


def test_cdp_emit_for_unregistered_event_is_silent() -> None:
    """Emitting an event with no registered handlers is a no-op."""
    session = _make_session()

    session.cdp.emit("Unregistered.Event", {"any": "payload"})


def test_session_rejects_capture_without_magic_key() -> None:
    """Sessions cannot bootstrap without a magic key for the XOR table."""
    from tankpit_bot.types import CaptureSession

    probe = _BufferProbe()
    bad_capture: CaptureSession = {
        "session_id": "no-magic",
        "start_timestamp_ms": 0,
        "end_timestamp_ms": None,
        "base_url": "https://tankpit.com",
        "messages": [],
        "magic": None,
        "game_log": [],
        "tank_names": {},
    }

    with pytest.raises(RuntimeError, match="no magic key"):
        RecordedChromiumSession(probe, bad_capture)


def test_session_wires_full_bootstrap_chain_end_to_end() -> None:
    """One factory call walks the entire chain back to the harness handles.

    Exercises every layer the production bootstrap touches in
    :func:`execute_live_probe_bootstrap` -- factory -> CM -> Playwright
    -> chromium -> launch -> new_context -> new_page +
    new_cdp_session -- and asserts each step returns exactly the
    handle the session prebuilt, proving the chain is wired
    correctly without weak type-identity checks.
    """
    session = _make_session()

    with session.sync_playwright_factory() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        cdp = context.new_cdp_session(page)

    assert playwright is session.playwright
    assert browser is session.browser
    assert context is session.context
    assert page is session.page
    assert cdp is session.cdp
    assert page.url == "https://tankpit.com/play"
    assert session.browser_type.launches == [True]


def test_page_wait_for_timeout_feeds_buffer_through_session() -> None:
    """The session's page drives the configured probe's CDP buffer."""
    probe = _BufferProbe()
    session = RecordedChromiumSession.from_capture_path(
        probe,
        FUEL_CAPTURE,
        frames_per_wait=3,
    )

    session.page.wait_for_timeout(50.0)

    assert len(probe._cdp_message_buffer) == 3
    assert session.clock.now_ms == 50
