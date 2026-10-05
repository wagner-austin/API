"""Tests for browser lifecycle standalone functions.

The cases that need a real headless Chromium live in
``tests/browser/test_real_chromium.py``, which launches the suite's only
browser; the cases here run against recorded and fake pages.
"""

from __future__ import annotations

import logging

import pytest

from tankpit_bot.browser.lifecycle import (
    navigate_and_login,
    wait_for_game_ready,
)
from tankpit_bot.browser.types import GameNotJoinedError
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.types import CapturedMessage
from tankpit_bot.types.literals import MessageDirection
from tests.action_lab._replay_page import (
    ClockAdvancingPage,
    ReplayClock,
)
from tests.conftest import FakeFileSystem
from tests.fakes import FakeCDPSession


class TestWaitForGameReady:
    def test_returns_when_messages_stabilize(self) -> None:
        messages: list[CapturedMessage] = [
            CapturedMessage(
                timestamp_ms=1,
                direction=MessageDirection.RECEIVED,
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
                direction=MessageDirection.RECEIVED,
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
                        direction=MessageDirection.RECEIVED,
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

    def test_an_unrotated_key_writes_nothing_and_says_nothing(
        self,
        fake_fs: FakeFileSystem,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The ordinary case: the client's key is the one the wheel ships.

        This is EVERY session until the game rotates its key, and it is
        the case that used to write. The write went into the installed
        package, which a source checkout allows and a container refuses,
        so a bot that was playing fine on the host died four seconds into
        the game in the fleet. Asserting that nothing is written is
        therefore the whole point of the test, not a detail of it.
        """
        from tankpit_bot.browser.key_discovery import load_static_key
        from tankpit_bot.browser.lifecycle import _check_shipped_static_key

        before = set(fake_fs.get_written_files())
        with caplog.at_level(logging.WARNING):
            _check_shipped_static_key(load_static_key())

        assert set(fake_fs.get_written_files()) == before
        assert [record.message for record in caplog.records] == []

    def test_a_rotated_key_is_reported_and_captured_beside_the_run(
        self,
        fake_fs: FakeFileSystem,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A key that no longer matches is drift, and drift is loud.

        The recovery matters as much as the warning: every session is
        decoding against the stale shipped key from here on, and the
        operator needs the new one to rebuild the asset. It lands in the
        run's own artifact directory — never back in the package, which
        is the write this whole change exists to remove.
        """
        from tankpit_bot.browser.lifecycle import _check_shipped_static_key
        from tankpit_bot.resources import static_key_file_path

        rotated = "Z" * 1000
        before = set(fake_fs.get_written_files())
        with caplog.at_level(logging.WARNING):
            _check_shipped_static_key(rotated)

        after = fake_fs.get_written_files()
        fresh = set(after) - before
        assert len(fresh) == 1
        captured = fresh.pop()
        assert captured.endswith("rotated_xor_static_key.txt")
        assert after[captured] == rotated + "\n"
        # The shipped asset is untouched — that write is the defect.
        assert after[str(static_key_file_path())] == "Y" + "A" * 999
        assert any("STATIC KEY ROTATED" in record.message for record in caplog.records)
