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
from tankpit_bot.types import CapturedMessage
from tests.action_lab._replay_core import ClockAdvancingPage, ReplayClock
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
