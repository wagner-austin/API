"""Tests for tankpit_bot.browser module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject

from tankpit_bot import _test_hooks
from tankpit_bot.browser import (
    BrowserError,
    BrowserSession,
    GameNotJoinedError,
    PlaywrightNotInstalledError,
    cdp_timestamp_to_ms,
    get_current_time_ms,
)
from tankpit_bot.types import CapturedMessage
from tests.fakes import FakeBrowser, FakeBrowserContext, FakeCDPSession, FakePage

# =============================================================================
# Helper Function Tests
# =============================================================================


def test_get_current_time_ms_returns_int() -> None:
    """Test get_current_time_ms returns an integer."""
    result = get_current_time_ms()
    assert type(result) is int
    assert result > 0


def test_cdp_timestamp_to_ms() -> None:
    """Test cdp_timestamp_to_ms converts seconds to milliseconds."""
    result = cdp_timestamp_to_ms(12345.678)
    assert result == 12345678


# =============================================================================
# Error Class Tests
# =============================================================================


def test_browser_error_is_exception() -> None:
    """Test BrowserError is a subclass of Exception."""
    assert issubclass(BrowserError, Exception)
    err = BrowserError("test error")
    assert str(err) == "test error"


def test_playwright_not_installed_error_is_browser_error() -> None:
    """Test PlaywrightNotInstalledError is a BrowserError."""
    assert issubclass(PlaywrightNotInstalledError, BrowserError)


def test_game_not_joined_error_is_browser_error() -> None:
    """Test GameNotJoinedError is a BrowserError."""
    assert issubclass(GameNotJoinedError, BrowserError)


# =============================================================================
# BrowserSession Tests
# =============================================================================


def test_browser_session_init() -> None:
    """Test BrowserSession initialization."""
    session = BrowserSession("https://example.com", headless=True, prefer_account=False)
    assert session._target_url == "https://example.com"
    assert session._headless is True
    assert session._prefer_account is False
    assert len(session.session_id) == 36  # UUID format
    assert session.messages == []
    assert session.magic is None


def test_browser_session_properties() -> None:
    """Test BrowserSession property accessors."""
    session = BrowserSession("https://example.com")
    session._magic = "test_magic"
    assert session.magic == "test_magic"
    assert len(session.session_id) == 36  # UUID format


def test_browser_session_on_websocket_created() -> None:
    """Test _on_websocket_created records WebSocket URL."""
    session = BrowserSession("https://example.com")
    params: JSONObject = {
        "requestId": "req1",
        "url": "wss://example.com/ws",
    }
    session._on_websocket_created(params)
    assert session._ws_urls["req1"] == "wss://example.com/ws"


def test_browser_session_on_websocket_frame_received() -> None:
    """Test _on_websocket_frame_received records message."""
    session = BrowserSession("https://example.com")
    session._ws_urls["req1"] = "wss://example.com/ws"
    params: JSONObject = {
        "requestId": "req1",
        "timestamp": 12345.678,
        "response": {"opcode": 1, "mask": False, "payloadData": "test_payload"},
    }
    session._on_websocket_frame_received(params)
    assert len(session.messages) == 1
    msg = session.messages[0]
    assert msg["direction"] == "received"
    assert msg["payload"] == "test_payload"
    assert msg["ws_url"] == "wss://example.com/ws"


def test_browser_session_on_websocket_frame_sent() -> None:
    """Test _on_websocket_frame_sent records message."""
    session = BrowserSession("https://example.com")
    session._ws_urls["req1"] = "wss://example.com/ws"
    params: JSONObject = {
        "requestId": "req1",
        "timestamp": 12345.678,
        "response": {"opcode": 1, "mask": True, "payloadData": "sent_payload"},
    }
    session._on_websocket_frame_sent(params)
    assert len(session.messages) == 1
    msg = session.messages[0]
    assert msg["direction"] == "sent"
    assert msg["payload"] == "sent_payload"


def test_browser_session_on_message_captured_default() -> None:
    """Test _on_message_captured does nothing by default."""
    session = BrowserSession("https://example.com")
    msg = CapturedMessage(
        timestamp_ms=12345,
        direction="sent",
        payload="test",
        ws_url="wss://example.com/ws",
    )
    # Should not raise
    session._on_message_captured(msg)


def test_browser_session_setup_cdp_handlers() -> None:
    """Test _setup_cdp_handlers registers event handlers."""
    session = BrowserSession("https://example.com")
    cdp = FakeCDPSession()
    session._setup_cdp_handlers(cdp)

    assert "Network.enable" in cdp._sent_methods
    assert "Network.webSocketCreated" in cdp._handlers
    assert "Network.webSocketFrameReceived" in cdp._handlers
    assert "Network.webSocketFrameSent" in cdp._handlers


def test_browser_session_capture_magic_key() -> None:
    """Test _capture_magic_key captures tankpit.magic from page."""
    session = BrowserSession("https://example.com")
    cdp = FakeCDPSession()
    page = FakePage(cdp, magic="test_magic_key_12345")
    session._capture_magic_key(page)
    assert session.magic == "test_magic_key_12345"


def test_browser_session_capture_magic_key_empty() -> None:
    """Test _capture_magic_key handles empty magic."""
    session = BrowserSession("https://example.com")
    cdp = FakeCDPSession()
    page = FakePage(cdp, magic="")
    session._capture_magic_key(page)
    assert session.magic is None


def test_browser_session_capture_magic_key_not_string() -> None:
    """Test _capture_magic_key handles non-string magic."""
    session = BrowserSession("https://example.com")
    cdp = FakeCDPSession()
    # FakePage returns None for magic by default - simulates non-string result
    page = FakePage(cdp, magic=None)
    session._capture_magic_key(page)
    assert session.magic is None


def test_browser_session_wait_for_game_ready_success() -> None:
    """Test _wait_for_game_ready succeeds when messages captured."""
    session = BrowserSession("https://example.com")
    # Pre-populate messages to simulate game loaded
    session._messages = [
        CapturedMessage(timestamp_ms=1, direction="received", payload="msg1", ws_url="ws://test"),
        CapturedMessage(timestamp_ms=2, direction="received", payload="msg2", ws_url="ws://test"),
    ]
    cdp = FakeCDPSession()
    page = FakePage(cdp)
    session._wait_for_game_ready(page)
    # Should not raise


def test_browser_session_wait_for_game_ready_no_messages() -> None:
    """Test _wait_for_game_ready raises when no messages captured."""
    session = BrowserSession("https://example.com")
    cdp = FakeCDPSession()
    page = FakePage(cdp)
    with pytest.raises(GameNotJoinedError):
        session._wait_for_game_ready(page)


def test_browser_session_launch_browser_no_playwright() -> None:
    """Test _launch_browser raises when Playwright not installed."""
    session = BrowserSession("https://example.com")
    original = _test_hooks.sync_playwright
    _test_hooks.sync_playwright = None
    try:
        with pytest.raises(PlaywrightNotInstalledError):
            session._launch_browser()
    finally:
        _test_hooks.sync_playwright = original


def test_browser_session_launch_browser_success() -> None:
    """Test _launch_browser launches browser and sets up CDP handlers."""
    from tests.fakes import fake_sync_playwright

    session = BrowserSession("https://example.com", headless=True)
    original = _test_hooks.sync_playwright
    _test_hooks.sync_playwright = fake_sync_playwright
    try:
        browser, context, page, cdp = session._launch_browser()
        # Simulate a WebSocket creation event to verify handlers are working
        ws_created_event: JSONObject = {
            "requestId": "test_req",
            "url": "wss://test.com/ws",
        }
        session._on_websocket_created(ws_created_event)
        assert session._ws_urls["test_req"] == "wss://test.com/ws"

        # Simulate a WebSocket frame event to verify message capture works
        ws_frame_event: JSONObject = {
            "requestId": "test_req",
            "timestamp": 1000.0,
            "response": {"opcode": 1, "mask": False, "payloadData": "test_data"},
        }
        session._on_websocket_frame_received(ws_frame_event)
        assert len(session.messages) == 1
        assert session.messages[0]["payload"] == "test_data"
        assert session.messages[0]["ws_url"] == "wss://test.com/ws"

        # Verify cleanup works correctly
        session._cleanup(cdp, page, context, browser)
    finally:
        _test_hooks.sync_playwright = original


def test_browser_session_cleanup() -> None:
    """Test _cleanup closes all browser resources."""
    session = BrowserSession("https://example.com")
    cdp = FakeCDPSession()
    page = FakePage(cdp)
    context = FakeBrowserContext()
    browser = FakeBrowser()

    session._cleanup(cdp, page, context, browser)

    assert cdp._detached is True
    assert page._closed is True
    assert context._closed is True
    assert browser._closed is True
