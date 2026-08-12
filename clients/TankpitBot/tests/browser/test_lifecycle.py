"""Tests for browser lifecycle standalone functions."""

from __future__ import annotations

import logging
from collections.abc import Generator

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import BrowserProtocol, CDPSessionProtocol, PageProtocol
from tankpit_bot._test_hooks.cdp import RouteFulfillTarget
from tankpit_bot.browser.lifecycle import (
    _handle_teardown_hang,
    cleanup_browser,
    navigate_and_login,
    wait_for_game_ready,
)
from tankpit_bot.browser.types import GameNotJoinedError
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.types import CapturedMessage
from tests.action_lab._replay_page import (
    ClockAdvancingPage,
    ReplayClock,
)
from tests.conftest import FakeFileSystem
from tests.fakes import FakeBrowser, FakeCDPSession

_PAGE_HTML = (
    '<!DOCTYPE html><html><head><script src="/tpclient.js"></script></head><body></body></html>'
)
_TEST_PAGE_URL = "http://localhost:9999/test-page"


@pytest.fixture(scope="module")
def headless_browser() -> Generator[BrowserProtocol, None, None]:
    """Launch one real headless Chromium shared by this module's browser tests.

    ``launch()`` is the expensive call and its cost is paid per launch,
    not per test: on hosts where a filesystem minifilter inspects the
    browser's teardown, terminating one has been measured at tens of
    seconds. The real-browser tests below therefore share one instance
    and take their own context each, mirroring the ``live_cdp`` fixture
    in ``tests/conftest.py``. ``--dist loadscope`` keeps a module's
    tests on one xdist worker, so a module-scoped browser is never
    shared across processes.

    A context is the isolation these tests actually need -- cache,
    cookies and routes are all per-context -- and none of them asserts
    anything about launch or close semantics.

    Yields:
        A live headless Chromium browser.
    """
    factory = _test_hooks.sync_playwright
    if factory is None:
        factory = _test_hooks.get_sync_playwright()
    if factory is None:
        pytest.skip("Playwright not available")
    with factory() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


def _intel_page(
    browser: BrowserProtocol,
    js_content: str,
) -> tuple[PageProtocol, CDPSessionProtocol]:
    """Open a page in a fresh context serving ``js_content`` as ``tpclient.js``.

    The page markup, both fulfilled routes and the navigation are
    identical across the real-browser tests; only the script body
    differs, so it is the only parameter.

    Args:
        browser: Browser to open a fresh context in.
        js_content: Body served for ``tpclient.js``.

    Returns:
        The navigated page and a CDP session attached to it.
    """

    def _fulfill_page(route: RouteFulfillTarget) -> None:
        route.fulfill(content_type="text/html", body=_PAGE_HTML)

    def _fulfill_tpclient(route: RouteFulfillTarget) -> None:
        route.fulfill(content_type="application/javascript", body=js_content)

    context = browser.new_context()
    page = context.new_page()
    cdp = context.new_cdp_session(page)
    page.route("**/test-page", _fulfill_page)
    page.route("**/tpclient.js", _fulfill_tpclient)
    page.goto(_TEST_PAGE_URL)
    return page, cdp


class TestWaitForGameReady:
    def test_returns_when_messages_stabilize(self) -> None:
        messages: list[CapturedMessage] = [
            CapturedMessage(
                timestamp_ms=1,
                direction="received",
                payload="x",
                ws_url="wss://test",
            ),
        ]
        page = ClockAdvancingPage(ReplayClock())
        wait_for_game_ready(page, messages)
        assert len(page.waits) >= 4

    def test_raises_when_no_messages(self) -> None:
        messages: list[CapturedMessage] = []
        page = ClockAdvancingPage(ReplayClock())
        with pytest.raises(GameNotJoinedError, match="No WebSocket messages"):
            wait_for_game_ready(page, messages)

    def test_resets_stable_checks_when_messages_arrive(self) -> None:
        messages: list[CapturedMessage] = [
            CapturedMessage(
                timestamp_ms=1,
                direction="received",
                payload="x",
                ws_url="wss://test",
            ),
        ]
        call_count = 0

        def _on_wait() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                messages.append(
                    CapturedMessage(
                        timestamp_ms=2,
                        direction="received",
                        payload="y",
                        ws_url="wss://test",
                    )
                )

        page = ClockAdvancingPage(ReplayClock(), on_wait=_on_wait)
        wait_for_game_ready(page, messages)
        assert len(messages) == 2


class TestNavigateAndLogin:
    def test_success_with_fake_page(self) -> None:
        from tests.fakes import FakePage

        cdp = FakeCDPSession()
        page = FakePage(cdp_session=cdp)
        navigate_and_login(
            page,
            cdp,
            WorldService(),
            target_url="https://tankpit.com/play",
            prefer_account=False,
        )

    def test_raises_on_login_failure(self) -> None:
        page = ClockAdvancingPage(ReplayClock())
        page.url = "https://tankpit.com/before-playing"
        cdp = FakeCDPSession()
        with pytest.raises(GameNotJoinedError, match="login or room join"):
            navigate_and_login(
                page,
                cdp,
                WorldService(),
                target_url="https://tankpit.com/play",
                prefer_account=False,
            )


class _FailCloseBrowser(FakeBrowser):
    def close(self, *, reason: str | None = None) -> None:
        _ = reason
        raise OSError("browser already closed")


class _RuntimeErrorCloseBrowser(FakeBrowser):
    def close(self, *, reason: str | None = None) -> None:
        _ = reason
        raise RuntimeError("browser teardown failed")


class TestCleanupBrowser:
    def test_closes_browser(self) -> None:
        browser = FakeBrowser()
        cleanup_browser(browser)

    def test_handles_os_error(self) -> None:
        cleanup_browser(_FailCloseBrowser())

    def test_handles_runtime_error(self) -> None:
        cleanup_browser(_RuntimeErrorCloseBrowser())


class TestGatherIntel:
    def test_returns_none_when_no_tpclient(self, fake_fs: FakeFileSystem) -> None:
        """``gather_intel`` yields no key and saves no client source.

        The second route into the tpclient writer, and the second one
        that reached the REAL filesystem when the URL type check was
        mutated away -- ``gather_intel`` delegates to
        ``_capture_static_key``, so both entry points need the fixture
        and both need to say that nothing was written.
        """
        from tankpit_bot.browser.lifecycle import gather_intel

        page = ClockAdvancingPage(ReplayClock())
        cdp = FakeCDPSession()

        assert gather_intel(page, cdp) is None
        written = [path for path in fake_fs.get_written_files() if path.endswith("tpclient.js")]
        assert written == []

    def test_debug_js_websocket_logs_without_error(self) -> None:
        from tankpit_bot.browser.lifecycle import _debug_js_websocket

        cdp = FakeCDPSession()
        _debug_js_websocket(cdp)

    def test_log_script_urls_with_page(self) -> None:
        from tankpit_bot.browser.lifecycle import _log_script_urls
        from tests.fakes import FakePage

        cdp = FakeCDPSession()
        page = FakePage(cdp_session=cdp)
        _log_script_urls(page)

    def test_capture_static_key_returns_none(
        self,
        fake_fs: FakeFileSystem,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A page with no tpclient URL says so, and saves nothing.

        Two things are pinned here, and they came from opposite ends.

        The write assertion is the one with a scar: the 2026-08-12
        mutation sweep removed this return and truncated the checked-in
        160KB ``tpclient.js`` to zero bytes, because the fetch was then
        attempted against the string ``'None'``, came back empty, and the
        empty result was written to the CWD-relative ``Path`` -- the
        repository root during a test run. Every test still passed. The
        suite could not tell the guard from absent; the working tree
        could.

        The message assertion is what still distinguishes this return
        now that an empty fetch is refused downstream. Both paths end in
        ``None`` and neither writes, so the diagnostic is the whole
        difference -- and it is the accurate one. Reporting "fetched no
        source from None" for a page that has no tpclient script at all
        would send a reader looking at the network instead of the page.
        """
        from tankpit_bot.browser.lifecycle import _capture_static_key

        page = ClockAdvancingPage(ReplayClock())

        with caplog.at_level(logging.WARNING):
            assert _capture_static_key(page) is None

        messages = [record.message for record in caplog.records]
        assert any("Could not find tpclient script URL" in message for message in messages)
        assert not any("Fetched no tpclient source" in message for message in messages)
        written = [path for path in fake_fs.get_written_files() if path.endswith("tpclient.js")]
        assert written == []

    @pytest.mark.usefixtures("fake_fs")
    def test_capture_static_key_with_real_headless_browser(
        self,
        headless_browser: BrowserProtocol,
    ) -> None:
        """Real Playwright headless browser extracts a 1000-char static key."""
        from tankpit_bot.browser.lifecycle import _capture_static_key

        static_key = "K" * 1000
        page, _ = _intel_page(headless_browser, f'var config = "{static_key}";')
        assert _capture_static_key(page) == static_key

    @pytest.mark.usefixtures("fake_fs")
    def test_gather_intel_with_real_headless_browser(
        self,
        headless_browser: BrowserProtocol,
    ) -> None:
        """Real Playwright headless browser runs gather_intel end to end."""
        from tankpit_bot.browser.lifecycle import gather_intel

        static_key = "J" * 1000
        page, cdp = _intel_page(headless_browser, f'var config = "{static_key}";')
        assert gather_intel(page, cdp) == static_key

    def test_an_empty_tpclient_body_is_not_saved_over_the_tracked_copy(
        self,
        fake_fs: FakeFileSystem,
        headless_browser: BrowserProtocol,
    ) -> None:
        """Real browser: a tpclient.js that serves nothing is not written.

        The script tag exists and its URL resolves, so the fetch runs and
        legitimately returns the empty string. The checked-in
        ``tpclient.js`` is the reference copy later sessions read, so
        saving an empty fetch over it destroys the artifact -- which is
        what the old ``else ""`` did with a fetch that returned nothing
        at all.
        """
        from tankpit_bot.browser.lifecycle import _capture_static_key

        page, _ = _intel_page(headless_browser, "")

        assert _capture_static_key(page) is None
        written = [path for path in fake_fs.get_written_files() if path.endswith("tpclient.js")]
        assert written == []

    def test_control_a_served_body_is_saved(
        self,
        fake_fs: FakeFileSystem,
        headless_browser: BrowserProtocol,
    ) -> None:
        """Control: real source IS written, so the silence above is the check."""
        from tankpit_bot.browser.lifecycle import _capture_static_key

        static_key = "L" * 1000
        page, _ = _intel_page(headless_browser, f'var config = "{static_key}";')

        assert _capture_static_key(page) == static_key
        written = [path for path in fake_fs.get_written_files() if path.endswith("tpclient.js")]
        assert len(written) == 1

    @pytest.mark.usefixtures("fake_fs")
    def test_capture_static_key_no_key_in_content(
        self,
        headless_browser: BrowserProtocol,
    ) -> None:
        """Real browser: tpclient.js exists but has no 1000-char string."""
        from tankpit_bot.browser.lifecycle import _capture_static_key

        page, _ = _intel_page(headless_browser, "var x = 1;")
        assert _capture_static_key(page) is None


class TestHandleTeardownHang:
    def test_calls_force_exit(self) -> None:
        from tankpit_bot import _test_hooks

        calls: list[int] = []
        original = _test_hooks.force_exit
        _test_hooks.force_exit = lambda code: calls.append(code)
        try:
            _handle_teardown_hang()
            assert calls == [75]
        finally:
            _test_hooks.force_exit = original
