"""Tests for BrowserSession basic functionality."""

from __future__ import annotations

import logging
from collections.abc import Callable

import pytest
from platform_core.json_utils import JSONObject

from tankpit_bot.browser import (
    BrowserSession,
)
from tankpit_bot.browser.cdp_utils import (
    _pop_sent_frame_metadata,
    get_captured_raw_messages,
)
from tankpit_bot.browser.inject_script import BROWSER_HOOK_SOURCE
from tankpit_bot.types import CapturedMessage
from tests.fakes import (
    FakeCDPSession,
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
    session._cdp_service._on_websocket_created(params)
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
    session._cdp_service._on_websocket_frame_received(params)
    assert len(session.messages) == 1
    msg = session.messages[0]
    assert msg["direction"] == "received"
    assert msg["payload"] == "test_payload"
    assert msg["ws_url"] == "wss://example.com/ws"


def test_browser_session_on_websocket_frame_sent() -> None:
    """Test _on_websocket_frame_sent records message."""
    session = BrowserSession("https://example.com")
    session._cdp_service.cdp = _MetadataCDP(
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
    session._cdp_service._on_websocket_frame_sent(params)
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
    session._cdp_service.cdp = _MetadataCDP(
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

    session._cdp_service._on_websocket_frame_sent(params)

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
    session._cdp_service._extract_magic_and_notify(msg)
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
    session._cdp_service._extract_magic_and_notify(msg)
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
    session._cdp_service._extract_magic_and_notify(msg)
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
    session._cdp_service._extract_magic_and_notify(msg)
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
    session._cdp_service._extract_magic_and_notify(msg)
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
    session._cdp_service._extract_magic_and_notify(msg1)
    session._cdp_service._extract_magic_and_notify(msg2)
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
    session._cdp_service._extract_magic_and_notify(msg)
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

    def test_page_errors_survive_the_noise_filter(self, caplog: pytest.LogCaptureFixture) -> None:
        """An error is kept whatever it says; chatty logs are still filtered.

        The substring filter is noise control for the game's own info
        logging, and it used to apply at every level — so a page-side
        ``TypeError`` was dropped for not containing "WS". The bot drove
        a page whose failures it could not see, which is why a frozen
        canvas had no explanation anywhere in the run log.
        """
        session = BrowserSession("https://example.com", headless=True)
        registered: dict[str, Callable[[JSONObject], None]] = {}

        class _CapturingCDP(FakeCDPSession):
            def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
                registered[event] = handler

        session._setup_console_listener(_CapturingCDP())
        handler = registered.get("Runtime.consoleAPICalled")
        if handler is None:
            raise AssertionError("expected Runtime.consoleAPICalled handler")

        with caplog.at_level(logging.WARNING):
            handler({"type": "error", "args": [{"value": "TypeError: x is not a function"}]})
            handler({"type": "warning", "args": [{"value": "deprecated thing"}]})
            handler({"type": "log", "args": [{"value": "chatty game info"}]})

        kept = [record.message for record in caplog.records]
        assert any("TypeError: x is not a function" in message for message in kept)
        assert any("deprecated thing" in message for message in kept)
        assert not any("chatty game info" in message for message in kept)

    def test_uncaught_page_exceptions_are_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        """Nothing listened to ``Runtime.exceptionThrown`` at all.

        rAF does not reschedule after a throw, so one uncaught error in
        the game's render loop freezes the canvas permanently while the
        WebSocket keeps updating game state — the bot plays on and the
        picture does not. That failure had no listener, so it left no
        trace in the run log.
        """
        session = BrowserSession("https://example.com", headless=True)
        registered: dict[str, Callable[[JSONObject], None]] = {}

        class _CapturingCDP(FakeCDPSession):
            def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
                registered[event] = handler

        session._setup_console_listener(_CapturingCDP())
        handler = registered.get("Runtime.exceptionThrown")
        if handler is None:
            raise AssertionError("expected Runtime.exceptionThrown handler")

        with caplog.at_level(logging.WARNING):
            handler({"exceptionDetails": {"text": "Uncaught TypeError: draw failed"}})
            handler({"exceptionDetails": {}})
            handler({"exceptionDetails": "not an object"})

        kept = [record.message for record in caplog.records]
        assert any("Uncaught TypeError: draw failed" in message for message in kept)
        # A malformed event still reports rather than vanishing.
        assert sum("[Page exception]" in message for message in kept) == 2


class TestGetArgv:
    """Tests for _test_hooks._real_get_argv."""

    def test_real_get_argv_returns_sys_argv(self) -> None:
        """_real_get_argv returns the actual sys.argv list."""
        import sys

        from tankpit_bot._test_hooks.runtime import _real_get_argv

        result = _real_get_argv()
        assert result is sys.argv


def test_browser_session_captured_message_count() -> None:
    """BrowserSession.captured_message_count delegates to message list length."""
    session = BrowserSession("https://example.com")
    assert session.captured_message_count() == 0
    session._messages.append(
        CapturedMessage(
            timestamp_ms=1,
            direction="received",
            payload="x",
            ws_url="wss://test",
        )
    )
    assert session.captured_message_count() == 1


def test_browser_session_static_key_property() -> None:
    """BrowserSession.static_key returns the stored static key."""
    session = BrowserSession("https://example.com")
    assert session.static_key is None
    session._static_key = "test_key"
    assert session.static_key == "test_key"


def test_browser_session_poll_game_log_empty() -> None:
    """BrowserSession._poll_game_log returns empty when no scraper."""
    session = BrowserSession("https://example.com")
    assert session._poll_game_log() == []


def test_browser_session_process_game_log_entry_logs_each_entry() -> None:
    """BrowserSession._process_game_log_entry logs every entry it receives.

    Combat-tracker integration was moved into the sniffer subclass
    2026-06-19 (only the sniffer needs per-capture forensic stats);
    the base class now just logs each entry at INFO.
    """
    from tankpit_bot.browser.dom_scraper import GameLogEntry

    session = BrowserSession("https://example.com")
    combat = GameLogEntry(text="You hit Tank123 for 50 damage", category="combat")
    session._process_game_log_entry(combat)
    non_combat = GameLogEntry(text="Zoom in", category="action")
    session._process_game_log_entry(non_combat)


def test_browser_session_poll_game_log_iterates_entries() -> None:
    """BrowserSession._poll_game_log iterates and processes scraper results."""

    class _GameLogCDP:
        def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
            _ = params
            if method == "Runtime.evaluate":
                return {"result": {"value": "Game Log\nZoom in\nTank full"}}
            return {"result": {"value": ""}}

        def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
            _ = (event, handler)

        def detach(self) -> None:
            pass

    session = BrowserSession("https://example.com")
    session._init_game_log_scraper(_GameLogCDP())
    entries = session._poll_game_log()
    assert len(entries) == 3


def test_browser_session_send_websocket_bytes() -> None:
    """BrowserSession._send_websocket_bytes delegates to send_websocket_bytes."""
    session = BrowserSession("https://example.com")
    cdp = FakeCDPSession()
    result = session._send_websocket_bytes(cdp, b"\x01\x02", "test_label")
    assert "SENT" in result or "no capturedWS" in result.lower() or result != ""
