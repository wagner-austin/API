"""Tests for browser lifecycle standalone functions."""

from __future__ import annotations

import pytest

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
from tests.fakes import FakeBrowser, FakeCDPSession


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
    def test_returns_none_when_no_tpclient(self) -> None:
        from tankpit_bot.browser.lifecycle import gather_intel

        page = ClockAdvancingPage(ReplayClock())
        cdp = FakeCDPSession()
        result = gather_intel(page, cdp)
        assert result is None

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

    def test_capture_static_key_returns_none(self) -> None:
        from tankpit_bot.browser.lifecycle import _capture_static_key

        page = ClockAdvancingPage(ReplayClock())
        result = _capture_static_key(page)
        assert result is None

    def test_capture_static_key_with_real_headless_browser(self) -> None:
        """Real Playwright headless browser extracts a 1000-char static key."""
        from tankpit_bot import _test_hooks
        from tankpit_bot.browser.lifecycle import _capture_static_key

        sync_pw = _test_hooks.sync_playwright
        if sync_pw is None:
            sync_pw = _test_hooks.get_sync_playwright()
        if sync_pw is None:
            pytest.skip("Playwright not available")

        from tests.conftest import FakeFileSystem

        fake_fs = FakeFileSystem()
        orig_write = _test_hooks.write_text
        _test_hooks.write_text = fake_fs.write_text

        static_key = "K" * 1000
        js_content = f'var config = "{static_key}";'

        with sync_pw() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_context().new_page()
            from tankpit_bot._test_hooks.cdp import RouteFulfillTarget

            def _fulfill_tpclient(route: RouteFulfillTarget) -> None:
                route.fulfill(
                    content_type="application/javascript",
                    body=js_content,
                )

            page_html = (
                "<!DOCTYPE html><html><head>"
                '<script src="/tpclient.js"></script>'
                "</head><body></body></html>"
            )

            def _fulfill_page(route: RouteFulfillTarget) -> None:
                route.fulfill(content_type="text/html", body=page_html)

            page.route("**/test-page", _fulfill_page)
            page.route("**/tpclient.js", _fulfill_tpclient)
            page.goto("http://localhost:9999/test-page")
            result = _capture_static_key(page)
            browser.close()

        _test_hooks.write_text = orig_write
        assert result == static_key

    def test_gather_intel_with_real_headless_browser(self) -> None:
        """Real Playwright headless browser runs gather_intel end to end."""
        from tankpit_bot import _test_hooks
        from tankpit_bot.browser.lifecycle import gather_intel

        sync_pw = _test_hooks.sync_playwright
        if sync_pw is None:
            sync_pw = _test_hooks.get_sync_playwright()
        if sync_pw is None:
            pytest.skip("Playwright not available")

        from tests.conftest import FakeFileSystem

        fake_fs = FakeFileSystem()
        orig_write = _test_hooks.write_text
        _test_hooks.write_text = fake_fs.write_text

        static_key = "J" * 1000
        js_content = f'var config = "{static_key}";'
        page_html = (
            "<!DOCTYPE html><html><head>"
            '<script src="/tpclient.js"></script>'
            "</head><body></body></html>"
        )

        from tankpit_bot._test_hooks.cdp import RouteFulfillTarget

        def _fulfill_page(route: RouteFulfillTarget) -> None:
            route.fulfill(content_type="text/html", body=page_html)

        def _fulfill_tpclient(route: RouteFulfillTarget) -> None:
            route.fulfill(content_type="application/javascript", body=js_content)

        with sync_pw() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context()
            page = ctx.new_page()
            cdp = ctx.new_cdp_session(page)
            page.route("**/test-page", _fulfill_page)
            page.route("**/tpclient.js", _fulfill_tpclient)
            page.goto("http://localhost:9999/test-page")
            result = gather_intel(page, cdp)
            browser.close()

        _test_hooks.write_text = orig_write
        assert result == static_key

    def test_capture_static_key_no_key_in_content(self) -> None:
        """Real browser: tpclient.js exists but has no 1000-char string."""
        from tankpit_bot import _test_hooks
        from tankpit_bot.browser.lifecycle import _capture_static_key
        from tests.conftest import FakeFileSystem

        sync_pw = _test_hooks.sync_playwright
        if sync_pw is None:
            sync_pw = _test_hooks.get_sync_playwright()
        if sync_pw is None:
            pytest.skip("Playwright not available")

        fake_fs = FakeFileSystem()
        orig_write = _test_hooks.write_text
        _test_hooks.write_text = fake_fs.write_text

        from tankpit_bot._test_hooks.cdp import RouteFulfillTarget

        page_html = (
            "<!DOCTYPE html><html><head>"
            '<script src="/tpclient.js"></script>'
            "</head><body></body></html>"
        )

        def _fulfill_page(route: RouteFulfillTarget) -> None:
            route.fulfill(content_type="text/html", body=page_html)

        def _fulfill_short_js(route: RouteFulfillTarget) -> None:
            route.fulfill(content_type="application/javascript", body="var x = 1;")

        with sync_pw() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_context().new_page()
            page.route("**/test-page", _fulfill_page)
            page.route("**/tpclient.js", _fulfill_short_js)
            page.goto("http://localhost:9999/test-page")
            result = _capture_static_key(page)
            browser.close()

        _test_hooks.write_text = orig_write
        assert result is None


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
