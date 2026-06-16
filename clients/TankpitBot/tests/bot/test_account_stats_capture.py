"""End-to-end tests for the bot's ``C``-panel account stats capture.

Every test drives the REAL pipeline: ``Bot._capture_account_stats`` ->
CDP keypress dispatch + DOM scrape -> real parsing -> JSONL artifact via
:class:`tests.conftest.FakeFileSystem`. The panel fixture mirrors the
live probe capture from 20260610-2348.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import JSONValue

from tankpit_bot._test_hooks import KeyboardProtocol, ResponseProtocol
from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot.bot import Bot
from tankpit_bot.diagnostics.event_stream import load_event_records
from tankpit_bot.runtime_logging import (
    RuntimeEventRecordDict,
    configure_bot_runtime_logging,
)
from tests.conftest import FakeCDPSessionSimple, FakeEnv, FakeFileSystem


class _NoOpKeyboard:
    """Minimal keyboard stub."""

    def press(self, key: str, *, delay: float | None = None) -> None:
        """No-op."""

    def type(self, text: str, *, delay: float | None = None) -> None:
        """No-op."""


class _MinimalPage:
    """Page stub for account-stats tests that only need wait_for_timeout."""

    def __init__(self) -> None:
        """Initialize."""
        self._url = "https://test.tankpit.com/play"
        self._keyboard = _NoOpKeyboard()

    @property
    def url(self) -> str:
        """Get URL."""
        return self._url

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Return keyboard."""
        return self._keyboard

    def wait_for_timeout(self, timeout: float) -> None:
        """No-op wait."""

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """No-op wait for event."""

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """No-op wait for function."""

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> ResponseProtocol | None:
        """No-op goto."""
        self._url = url
        return None

    def evaluate(self, expression: str) -> JSONValue:
        """No-op evaluate."""
        return None

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """No-op close."""


_PANEL_TEXT = (
    " Rank: private (160)\n"
    " Statistics:\n"
    " Play time: 41:25:23\n"
    " Destroyed enemies: 65\n"
    " Deactivated: 0\n"
    " Promotion points: 121314\n"
)


def _sample_records(latest_events_path: str) -> list[RuntimeEventRecordDict]:
    """Return every ``session_account_stats`` record from the artifact."""
    return [
        record
        for record in load_event_records(Path(latest_events_path))
        if record["fields"].get("diagnostic_kind") == "session_account_stats"
    ]


def test_capture_toggles_panel_and_emits_stats(fake_env: FakeEnv, fake_fs: FakeFileSystem) -> None:
    """The capture opens the panel, scrapes, closes it, and emits."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    bot = Bot("https://test.tankpit.com/", headless=True)
    cdp = FakeCDPSessionSimple()
    cdp.add_response({})
    cdp.add_response({})
    cdp.add_response({"result": {"value": _PANEL_TEXT}})
    bot._cdp = cdp
    bot._page = _MinimalPage()

    bot._capture_account_stats("startup")

    methods = [method for method, _ in cdp.get_calls()]
    assert methods == [
        "Input.dispatchKeyEvent",
        "Input.dispatchKeyEvent",
        "Runtime.evaluate",
        "Input.dispatchKeyEvent",
        "Input.dispatchKeyEvent",
    ]
    records = _sample_records(artifacts["latest_events_path"])
    assert len(records) == 1
    assert records[0]["fields"]["panel_visible"] is True
    assert records[0]["fields"]["promotion_points"] == 121314
    assert records[0]["fields"]["destroyed_enemies"] == 65
    assert records[0]["fields"]["phase"] == "startup"


def test_capture_emits_absence_when_panel_not_rendered(
    fake_env: FakeEnv, fake_fs: FakeFileSystem
) -> None:
    """A scrape that missed the panel emits a loud absence marker."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    bot = Bot("https://test.tankpit.com/", headless=True)
    cdp = FakeCDPSessionSimple()
    cdp.add_response({})
    cdp.add_response({})
    cdp.add_response({"result": {"value": "LOCATION: 1,2"}})
    bot._cdp = cdp
    bot._page = _MinimalPage()

    bot._capture_account_stats("startup")

    records = _sample_records(artifacts["latest_events_path"])
    assert len(records) == 1
    assert records[0]["fields"] == {
        "diagnostic_kind": "session_account_stats",
        "phase": "startup",
        "panel_visible": False,
        "marker_present": False,
        "scrape_chars": 0,
    }


def test_capture_without_cdp_is_a_no_op(fake_env: FakeEnv, fake_fs: FakeFileSystem) -> None:
    """No CDP session means nothing to scrape and nothing emitted."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    bot = Bot("https://test.tankpit.com/", headless=True)
    bot._cdp = None
    bot._page = None

    bot._capture_account_stats("startup")

    assert _sample_records(artifacts["latest_events_path"]) == []


def test_capture_without_page_is_a_no_op(fake_env: FakeEnv, fake_fs: FakeFileSystem) -> None:
    """A CDP session without a page cannot wait for the panel render."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    bot = Bot("https://test.tankpit.com/", headless=True)
    bot._cdp = FakeCDPSessionSimple()
    bot._page = None

    bot._capture_account_stats("startup")

    assert _sample_records(artifacts["latest_events_path"]) == []
