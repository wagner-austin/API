"""Tests for BrowserSession basic functionality."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject

from tankpit_bot import _test_hooks
from tankpit_bot.browser import (
    BrowserSession,
    GameNotJoinedError,
    PlaywrightNotInstalledError,
)
from tankpit_bot.types import CapturedMessage
from tests.fakes import (
    FakeBrowser,
    FakeBrowserContext,
    FakeCDPSession,
    FakePage,
    FakePageGrowingMessages,
)


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


def test_browser_session_on_message_captured_non_auth_sent() -> None:
    """Test _on_message_captured ignores non-AUTH sent messages."""
    session = BrowserSession("https://example.com")
    msg = CapturedMessage(
        timestamp_ms=12345,
        direction="sent",
        payload="test",
        ws_url="wss://example.com/ws",
    )
    session._on_message_captured(msg)
    assert session._magic is None


def test_browser_session_on_message_captured_empty_payload() -> None:
    """Test _on_message_captured ignores empty payloads."""
    session = BrowserSession("https://example.com")
    msg = CapturedMessage(
        timestamp_ms=12345,
        direction="sent",
        payload="",
        ws_url="wss://example.com/ws",
    )
    session._on_message_captured(msg)
    assert session._magic is None


def test_browser_session_on_message_captured_invalid_base64() -> None:
    """Test _on_message_captured ignores invalid base64 payloads."""
    session = BrowserSession("https://example.com")
    msg = CapturedMessage(
        timestamp_ms=12345,
        direction="sent",
        payload="not!valid@base64",
        ws_url="wss://example.com/ws",
    )
    session._on_message_captured(msg)
    assert session._magic is None


def test_browser_session_on_message_captured_extracts_magic() -> None:
    """Test _on_message_captured extracts magic from AUTH message."""
    import base64

    session = BrowserSession("https://example.com")
    body = "%AUTH !be session|hash|ts test_magic_key_12345"
    body_bytes = body.encode("utf-8")
    length_prefix = len(body_bytes).to_bytes(2, "little")
    payload = base64.b64encode(length_prefix + body_bytes).decode("ascii")

    msg = CapturedMessage(
        timestamp_ms=12345,
        direction="sent",
        payload=payload,
        ws_url="wss://example.com/ws",
    )
    session._on_message_captured(msg)
    assert session._magic == "test_magic_key_12345"


def test_browser_session_on_message_captured_skips_received() -> None:
    """Test _on_message_captured does not extract magic from received messages."""
    import base64

    session = BrowserSession("https://example.com")
    body = "%AUTH !be session|hash|ts test_magic_key_12345"
    body_bytes = body.encode("utf-8")
    length_prefix = len(body_bytes).to_bytes(2, "little")
    payload = base64.b64encode(length_prefix + body_bytes).decode("ascii")

    msg = CapturedMessage(
        timestamp_ms=12345,
        direction="received",
        payload=payload,
        ws_url="wss://example.com/ws",
    )
    session._on_message_captured(msg)
    assert session._magic is None


def test_browser_session_on_message_captured_only_first_magic() -> None:
    """Test _on_message_captured only captures magic once."""
    import base64

    session = BrowserSession("https://example.com")

    def make_auth_payload(magic: str) -> str:
        body = f"%AUTH !be session|hash|ts {magic}"
        body_bytes = body.encode("utf-8")
        length_prefix = len(body_bytes).to_bytes(2, "little")
        return base64.b64encode(length_prefix + body_bytes).decode("ascii")

    msg1 = CapturedMessage(
        timestamp_ms=1,
        direction="sent",
        payload=make_auth_payload("first_magic_key_12345"),
        ws_url="wss://example.com/ws",
    )
    msg2 = CapturedMessage(
        timestamp_ms=2,
        direction="sent",
        payload=make_auth_payload("second_magic_key_1234"),
        ws_url="wss://example.com/ws",
    )
    session._on_message_captured(msg1)
    session._on_message_captured(msg2)
    assert session._magic == "first_magic_key_12345"


def test_browser_session_on_magic_captured_called() -> None:
    """Test _on_magic_captured is called when magic is extracted."""
    import base64

    captured_magics: list[str] = []

    class TestSession(BrowserSession):
        def _on_magic_captured(self, magic: str) -> None:
            captured_magics.append(magic)

    session = TestSession("https://example.com")
    body = "%AUTH !be session|hash|ts test_magic_key_12345"
    body_bytes = body.encode("utf-8")
    length_prefix = len(body_bytes).to_bytes(2, "little")
    payload = base64.b64encode(length_prefix + body_bytes).decode("ascii")

    msg = CapturedMessage(
        timestamp_ms=12345,
        direction="sent",
        payload=payload,
        ws_url="wss://example.com/ws",
    )
    session._on_message_captured(msg)
    assert captured_magics == ["test_magic_key_12345"]


def test_browser_session_on_magic_captured_default_noop() -> None:
    """Test _on_magic_captured does nothing by default."""
    session = BrowserSession("https://example.com")
    # Should not raise
    session._on_magic_captured("test_magic")


def test_browser_session_setup_cdp_handlers() -> None:
    """Test _setup_cdp_handlers registers event handlers."""
    session = BrowserSession("https://example.com")
    cdp = FakeCDPSession()
    session._setup_cdp_handlers(cdp)

    assert "Network.enable" in cdp._sent_methods
    assert "Network.webSocketCreated" in cdp._handlers
    assert "Network.webSocketFrameReceived" in cdp._handlers
    assert "Network.webSocketFrameSent" in cdp._handlers


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


def test_browser_session_wait_for_game_ready_stabilization_reset() -> None:
    """Test _wait_for_game_ready resets when new messages arrive during wait."""
    session = BrowserSession("https://example.com")
    session._messages = [
        CapturedMessage(timestamp_ms=1, direction="received", payload="msg1", ws_url="ws://test"),
    ]
    page = FakePageGrowingMessages(session._messages, add_on_call=2)
    session._wait_for_game_ready(page)
    assert len(session._messages) == 2


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


def test_browser_session_static_key_property() -> None:
    """Test static_key property returns captured static key."""
    session = BrowserSession("https://example.com")
    # Initially None
    assert session.static_key is None

    # After setting
    session._static_key = "test_static_key"
    assert session.static_key == "test_static_key"


class _FakeCDPNoResultDict(FakeCDPSession):
    """FakeCDPSession that returns empty response (no 'result' key)."""

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Return CDP response without a result dict.

        Args:
            method: CDP method name.
            params: Optional parameters.

        Returns:
            Empty JSONObject (no 'result' key).
        """
        _ = params
        self._sent_methods.append(method)
        return {}


def test_debug_js_websocket_no_result_dict() -> None:
    """_debug_js_websocket handles CDP response without a result dict."""
    session = BrowserSession("https://example.com", headless=True)
    cdp = _FakeCDPNoResultDict()
    session._debug_js_websocket(cdp)
    assert len(cdp._sent_methods) == 1
