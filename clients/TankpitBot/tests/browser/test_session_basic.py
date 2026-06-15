"""Tests for BrowserSession basic functionality."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from platform_core.json_utils import JSONObject

from tankpit_bot import _test_hooks
from tankpit_bot.browser import (
    BrowserSession,
    GameNotJoinedError,
    PlaywrightNotInstalledError,
)
from tankpit_bot.browser.inject_script import BROWSER_HOOK_SOURCE
from tankpit_bot.browser.session import (
    _pop_sent_frame_metadata,
    get_captured_raw_messages,
)
from tankpit_bot.types import CapturedMessage
from tests.fakes import (
    FakeBrowser,
    FakeBrowserContext,
    FakeCDPSession,
    FakePage,
    FakePageGrowingMessages,
)


class _MetadataCDP:
    """Minimal CDP fake for outbound metadata pop tests."""

    def __init__(self, value: JSONObject | None) -> None:
        """Store one runtime-evaluate result value."""
        self._value = value

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Return the configured runtime-evaluate value."""
        _ = params
        if method != "Runtime.evaluate":
            return {"result": {"value": ""}}
        return {"result": {"value": self._value}}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Ignore event registration for this minimal fake."""
        _ = (event, handler)

    def detach(self) -> None:
        """Ignore detach for this minimal fake."""


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
    session._cdp = _MetadataCDP(
        {
            "origin": "bot_injected",
            "label": "teleport(129,106)",
            "stack": "Error\\n at send",
        }
    )
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
    assert msg["sent_origin"] == "bot_injected"
    assert msg["sent_label"] == "teleport(129,106)"
    assert msg["sent_stack"] == "Error\\n at send"


def test_browser_session_on_websocket_frame_sent_without_optional_metadata() -> None:
    """Test sent frame metadata omits empty label and stack values."""
    session = BrowserSession("https://example.com")
    session._cdp = _MetadataCDP(
        {
            "origin": "page_client",
            "label": "",
            "stack": "",
        }
    )
    session._ws_urls["req1"] = "wss://example.com/ws"
    params: JSONObject = {
        "requestId": "req1",
        "timestamp": 12345.678,
        "response": {"opcode": 1, "mask": True, "payloadData": "sent_payload"},
    }

    session._on_websocket_frame_sent(params)

    msg = session.messages[0]
    assert msg["sent_origin"] == "page_client"
    assert "sent_label" not in msg
    assert "sent_stack" not in msg


def test_pop_sent_frame_metadata_returns_none_when_queue_empty() -> None:
    """Test empty send metadata queue returns None."""
    cdp = _MetadataCDP(None)

    result = _pop_sent_frame_metadata(cdp)

    assert result is None


def test_pop_sent_frame_metadata_decodes_record() -> None:
    """Test send metadata is decoded with strict validation."""
    cdp = _MetadataCDP(
        {
            "origin": "bot_injected",
            "label": "teleport(129,106)",
            "stack": "Error\\n at send",
        }
    )

    result = _pop_sent_frame_metadata(cdp)

    assert result == {
        "origin": "bot_injected",
        "label": "teleport(129,106)",
        "stack": "Error\\n at send",
    }


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


def test_browser_hook_source_captures_closure_scoped_game_client() -> None:
    """Injected browser hook stores the active game client outside tpclient closure scope."""
    assert "window.__tankpitActiveGame = null;" in BROWSER_HOOK_SOURCE
    assert "function maybeCaptureGameClient(candidate)" in BROWSER_HOOK_SOURCE
    assert "installClientProbe('map');" in BROWSER_HOOK_SOURCE
    assert "installClientProbe('Ha');" in BROWSER_HOOK_SOURCE
    assert "window.__tankpitActiveGame = candidate;" in BROWSER_HOOK_SOURCE


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


def test_get_captured_raw_messages_requires_value_field() -> None:
    """Captured raw-message helper rejects CDP results without a value field."""

    class _FakeCDPMissingValue:
        def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
            _ = (method, params)
            return {"result": {}}

        def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
            _ = (event, handler)

        def detach(self) -> None:
            return None

    with pytest.raises(ValueError, match="missing value"):
        get_captured_raw_messages(_FakeCDPMissingValue())


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


def test_browser_session_navigate_and_login_raises_when_login_flow_fails() -> None:
    """Navigate/login raises when the login flow reports failure."""
    from tests.login.conftest import FakeCDPLogin, FakePageLogin

    session = BrowserSession("https://example.com")
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin(include_practice_room=False)

    with pytest.raises(GameNotJoinedError, match="did not complete successfully"):
        session._navigate_and_login(page, cdp, auto_join_room=True)


def test_browser_session_cleanup() -> None:
    """Teardown arms the watchdog and closes only the browser.

    ``browser.close()`` subsumes page/context/CDP teardown; the old
    four-step sequence gave sync Playwright four chances to deadlock
    (runs 20260611-083908/092159 hung 10+ minutes after saving).
    """
    armed: list[float] = []

    def record_watchdog(seconds: float, on_fire: Callable[[], None]) -> None:
        del on_fire
        armed.append(seconds)

    _test_hooks.start_watchdog = record_watchdog
    session = BrowserSession("https://example.com")
    cdp = FakeCDPSession()
    page = FakePage(cdp)
    context = FakeBrowserContext()
    browser = FakeBrowser()

    session._cleanup(cdp, page, context, browser)

    assert browser._closed is True
    assert cdp._detached is False
    assert page._closed is False
    assert context._closed is False
    assert armed == [30.0]


def test_teardown_hang_handler_forces_distinct_exit_code() -> None:
    """A fired teardown watchdog forces a recorded, distinct exit."""
    from tankpit_bot.browser.session import _handle_teardown_hang

    exit_codes: list[int] = []
    _test_hooks.force_exit = exit_codes.append

    _handle_teardown_hang()

    assert exit_codes == [75]


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


class _FakeBrowserRaising(FakeBrowser):
    """Browser fake that raises RuntimeError on close."""

    def close(self, *, reason: str | None = None) -> None:
        """Raise RuntimeError to simulate already-closed browser."""
        _ = reason
        msg = "already closed"
        raise RuntimeError(msg)


class TestCleanup:
    """Tests for BrowserSession._cleanup error handling."""

    def test_cleanup_handles_browser_close_error(self) -> None:
        """_cleanup catches RuntimeError from an already-closed browser."""
        session = BrowserSession("https://example.com", headless=True)
        cdp = FakeCDPSession()
        page = FakePage(cdp)
        context = FakeBrowserContext()
        browser = _FakeBrowserRaising()
        # Should not raise — the close error is caught and logged
        session._cleanup(cdp, page, context, browser)


class TestConsoleListener:
    """Tests for BrowserSession._setup_console_listener."""

    def test_console_listener_registers_handler(self) -> None:
        """_setup_console_listener enables Runtime and registers handler."""
        session = BrowserSession("https://example.com", headless=True)
        calls: list[str] = []

        class _TrackingCDP(FakeCDPSession):
            def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
                calls.append(method)
                return {}

        cdp = _TrackingCDP()
        session._setup_console_listener(cdp)
        assert "Runtime.enable" in calls

    def test_console_handler_logs_ws_messages(self) -> None:
        """Console handler processes WebSocket-related messages."""
        session = BrowserSession("https://example.com", headless=True)
        registered_handlers: dict[str, Callable[[JSONObject], None]] = {}

        class _CapturingCDP(FakeCDPSession):
            def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
                registered_handlers[event] = handler

        cdp = _CapturingCDP()
        session._setup_console_listener(cdp)
        handler = registered_handlers.get("Runtime.consoleAPICalled")
        if handler is None:
            raise AssertionError("expected Runtime.consoleAPICalled handler")
        handler({"type": "log", "args": [{"value": "WS connected"}]})
        handler({"type": "log", "args": [{"value": "normal message"}]})
        handler({"type": "log", "args": []})
        handler({"type": "log", "args": "not_a_list"})
        handler({"type": "log", "args": [{"description": "Hook fired"}]})
        handler({"type": "log", "args": [{"value": None}]})
        handler({"type": "log", "args": ["raw_string_arg"]})


class TestGetArgv:
    """Tests for _test_hooks._real_get_argv."""

    def test_real_get_argv_returns_sys_argv(self) -> None:
        """_real_get_argv returns the actual sys.argv list."""
        import sys

        from tankpit_bot._test_hooks.runtime import _real_get_argv

        result = _real_get_argv()
        assert result is sys.argv
