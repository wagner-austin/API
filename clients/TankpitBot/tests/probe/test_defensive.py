"""Tests for defensive branches in probe module."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.probe import (
    _log_discovered_commands,
)
from tankpit_bot.types import CapturedMessage, MouseInput, ProbeInput, ProbeResult
from tests.conftest import FakeEnv, FakeFileSystem


def test_log_discovered_commands_key_with_none_input() -> None:
    """Test _log_discovered_commands handles key result with None key_input."""
    result = ProbeResult(
        input=ProbeInput(input_type="key", key_input=None, mouse_input=None),
        timestamp_ms=12345,
        messages_before_count=0,
        messages_after=[],
    )
    # Should not raise, just skip logging
    _log_discovered_commands([result])


def test_log_discovered_commands_mouse_with_none_input() -> None:
    """Test _log_discovered_commands handles mouse result with None mouse_input."""
    result = ProbeResult(
        input=ProbeInput(input_type="mouse", key_input=None, mouse_input=None),
        timestamp_ms=12345,
        messages_before_count=0,
        messages_after=[],
    )
    # Should not raise, just skip logging
    _log_discovered_commands([result])


def test_log_discovered_commands_mouse_with_messages(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test _log_discovered_commands logs mouse results with messages."""
    import logging

    result = ProbeResult(
        input=ProbeInput(
            input_type="mouse",
            key_input=None,
            mouse_input=MouseInput(x=100, y=200, button="left"),
        ),
        timestamp_ms=12345,
        messages_before_count=0,
        messages_after=[
            CapturedMessage(
                timestamp_ms=12346,
                direction="sent",
                payload="test_payload",
                ws_url="wss://test.com/ws",
            )
        ],
    )
    with caplog.at_level(logging.INFO):
        _log_discovered_commands([result])

    assert "Discovered: Mouse (100,200) -> 1 msg(s)" in caplog.text


def test_send_websocket_bytes_returns_false_on_non_dict_result(
    fake_fs: FakeFileSystem,
) -> None:
    """Test _send_websocket_bytes raises on malformed CDP result objects."""
    from tankpit_bot.browser import BrowserSession

    class FakeCDPNonDictResult:
        """Fake CDP session that returns non-dict result for ws.send."""

        def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
            """Return non-dict result for ws.send evaluation."""
            _ = method
            _ = params
            # Return result where "result" is a string, not a dict
            return {"result": "NOT_A_DICT"}

        def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
            """Stub handler registration."""
            _ = event
            _ = handler

        def detach(self) -> None:
            """Stub detach."""

    session = BrowserSession("https://tankpit.com/play", headless=True)
    cdp = FakeCDPNonDictResult()

    with pytest.raises(JSONTypeError, match="Field 'result' must be an object"):
        session._send_websocket_bytes(cdp, b"test_data")


def test_protocol_probe_toggle_key_close(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
) -> None:
    """Test that second press of toggle key uses JS keypress to close."""
    from tankpit_bot.probe import TOGGLE_KEYS, ProtocolProbe
    from tests.fakes import FakeCDPSessionProbe, FakePageProbe

    cdp = FakeCDPSessionProbe(emit_on_key=True)
    page = FakePageProbe(cdp)
    probe = ProtocolProbe("https://tankpit.com/play", headless=True)

    # Setup: mark 'f' as already open
    probe._open_toggles.add("f")
    probe._magic = "test_magic"
    probe._xor_table = b"\x00" * 1000

    # First, verify 'f' is a toggle key
    assert "f" in TOGGLE_KEYS

    # Probe 'f' when it's already open - should use JS keypress to close
    probe._probe_single_key(page, cdp, "f")

    # After closing, 'f' should be removed from open toggles
    assert "f" not in probe._open_toggles


def test_send_js_keypress_returns_question_on_non_dict_result(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
) -> None:
    """Test _send_js_keypress returns '?' when CDP result is not a dict."""
    from tankpit_bot.probe import ProtocolProbe

    class FakeCDPNonDictResultLocal:
        """Fake CDP session that returns non-dict result."""

        def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
            """Return non-dict result."""
            _ = method
            _ = params
            return {"result": "NOT_A_DICT"}

        def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
            """Stub handler registration."""
            _ = event
            _ = handler

        def detach(self) -> None:
            """Stub detach."""

    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    cdp = FakeCDPNonDictResultLocal()

    # Call _send_js_keypress with the fake CDP that returns non-dict
    result = probe._send_js_keypress(cdp, "f")

    # Should return "?" since result["result"] is not a dict
    assert result == "?"


def test_protocol_probe_toggle_key_close_no_wait_on_failed_js(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
) -> None:
    """Test that toggle close skips wait when JS keypress returns error."""
    from tankpit_bot.probe import ProtocolProbe
    from tests.fakes import FakeCDPSessionProbe, FakePageProbe

    # Use js_keypress_fails=True to make JS keypress return "ERROR"
    cdp = FakeCDPSessionProbe(emit_on_key=True, js_keypress_fails=True)
    page = FakePageProbe(cdp)
    probe = ProtocolProbe("https://tankpit.com/play", headless=True)

    # Setup: mark 'f' as already open
    probe._open_toggles.add("f")
    probe._magic = "test_magic"
    probe._xor_table = b"\x00" * 1000

    # Probe 'f' when it's already open - JS keypress will return "ERROR"
    # This covers the branch where result.startswith("JS_KEYPRESS_") is False
    probe._probe_single_key(page, cdp, "f")

    # Toggle should still be removed (close attempted)
    assert "f" not in probe._open_toggles


def test_console_listener_logs_websocket_messages(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test _setup_console_listener logs WebSocket-related console messages."""
    import logging

    from tankpit_bot.browser import BrowserSession
    from tests.fakes import FakeCDPSessionProbe

    class FakeCDPWithConsole(FakeCDPSessionProbe):
        """Fake CDP that can emit console events."""

        def emit_console(self, msg_type: str, text: str) -> None:
            """Emit a console event."""
            if "Runtime.consoleAPICalled" in self._handlers:
                for handler in self._handlers["Runtime.consoleAPICalled"]:
                    handler(
                        {
                            "type": msg_type,
                            "args": [{"value": text}],
                        }
                    )

    session = BrowserSession("https://tankpit.com/play", headless=True)
    cdp = FakeCDPWithConsole(emit_on_key=True)

    # Set up console listener
    session._setup_console_listener(cdp)

    # Emit console messages - WebSocket-related should be logged
    with caplog.at_level(logging.INFO):
        cdp.emit_console("log", "[WS Hook] Captured WebSocket via send")
        cdp.emit_console("log", "Some unrelated message")
        cdp.emit_console("info", "WebSocket connected")

    # WebSocket-related messages should be logged (may wrap across lines)
    assert "[Console log]" in caplog.text and "WS Hook" in caplog.text
    assert "[Console info]" in caplog.text and "WebSocket connected" in caplog.text
    # Unrelated messages should NOT be logged
    assert "Some unrelated message" not in caplog.text


def test_console_listener_handles_missing_value(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test _setup_console_listener handles args with description instead of value."""
    import logging

    from tankpit_bot.browser import BrowserSession
    from tests.fakes import FakeCDPSessionProbe

    class FakeCDPWithConsole(FakeCDPSessionProbe):
        """Fake CDP that can emit console events."""

        def emit_console_with_description(self, msg_type: str, desc: str) -> None:
            """Emit a console event with description instead of value."""
            if "Runtime.consoleAPICalled" in self._handlers:
                for handler in self._handlers["Runtime.consoleAPICalled"]:
                    handler(
                        {
                            "type": msg_type,
                            "args": [{"description": desc}],
                        }
                    )

        def emit_console_with_none_value(self, msg_type: str) -> None:
            """Emit a console event with None value."""
            if "Runtime.consoleAPICalled" in self._handlers:
                for handler in self._handlers["Runtime.consoleAPICalled"]:
                    handler(
                        {
                            "type": msg_type,
                            "args": [{"value": None}],
                        }
                    )

    session = BrowserSession("https://tankpit.com/play", headless=True)
    cdp = FakeCDPWithConsole(emit_on_key=True)

    # Set up console listener
    session._setup_console_listener(cdp)

    # Emit with description (fallback path)
    with caplog.at_level(logging.INFO):
        cdp.emit_console_with_description("log", "WS Hook description")
        # Emit with None value (? fallback path)
        cdp.emit_console_with_none_value("log")

    # Description should be used as fallback
    assert "[Console log]" in caplog.text and "WS Hook description" in caplog.text


def test_console_listener_handles_non_list_args(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
) -> None:
    """Test _setup_console_listener handles args that is not a list."""
    from tankpit_bot.browser import BrowserSession
    from tests.fakes import FakeCDPSessionProbe

    class FakeCDPWithBadConsole(FakeCDPSessionProbe):
        """Fake CDP that emits console events with non-list args."""

        def emit_console_non_list_args(self, msg_type: str) -> None:
            """Emit a console event with non-list args."""
            if "Runtime.consoleAPICalled" in self._handlers:
                for handler in self._handlers["Runtime.consoleAPICalled"]:
                    handler(
                        {
                            "type": msg_type,
                            "args": "not a list",  # Wrong type
                        }
                    )

    session = BrowserSession("https://tankpit.com/play", headless=True)
    cdp = FakeCDPWithBadConsole(emit_on_key=True)

    # Set up console listener
    session._setup_console_listener(cdp)

    # Emit with non-list args - should not crash
    cdp.emit_console_non_list_args("log")
    # Test passes if no exception raised


def test_console_listener_handles_non_dict_arg_element(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test _setup_console_listener handles arg elements that are not dicts."""
    import logging

    from tankpit_bot.browser import BrowserSession
    from tests.fakes import FakeCDPSessionProbe

    class FakeCDPWithMixedArgs(FakeCDPSessionProbe):
        """Fake CDP that emits console events with mixed arg types."""

        def emit_console_mixed_args(self, msg_type: str) -> None:
            """Emit a console event with mixed arg types."""
            if "Runtime.consoleAPICalled" in self._handlers:
                for handler in self._handlers["Runtime.consoleAPICalled"]:
                    handler(
                        {
                            "type": msg_type,
                            "args": [
                                "not a dict",  # Skip this
                                {"value": "WS valid dict"},  # Process this
                                123,  # Skip this
                            ],
                        }
                    )

    session = BrowserSession("https://tankpit.com/play", headless=True)
    cdp = FakeCDPWithMixedArgs(emit_on_key=True)

    # Set up console listener
    session._setup_console_listener(cdp)

    # Emit with mixed args - should only process dict elements
    with caplog.at_level(logging.INFO):
        cdp.emit_console_mixed_args("log")

    # Only the dict element should be processed
    assert "WS valid dict" in caplog.text
