"""Shared fixtures and fake classes for login testing."""

from __future__ import annotations

import base64
import re
from collections.abc import Callable

from platform_core.json_utils import JSONObject, JSONValue

from tankpit_bot._test_hooks import KeyboardProtocol, ResponseProtocol
from tankpit_bot.protocol.framing import decode_frame, encode_frame
from tests.no_op_keyboard import NoOpKeyboard


class FakeCDPLogin:
    """Fake CDP session for login testing."""

    def __init__(
        self,
        *,
        rate_limited: bool = False,
        login_error: str = "",
        include_practice_room: bool = True,
        emit_join_confirm: bool = True,
        emit_enter_response: bool = True,
        select_send_result: str = "SENT_4_BYTES via wss://tankpit.com/ws/",
        enter_send_result: str | None = None,
    ) -> None:
        """Initialize fake CDP session."""
        self._eval_count = 0
        self._rate_limited = rate_limited
        self._login_error = login_error
        self._account_login_started = False
        self._include_practice_room = include_practice_room
        self._emit_join_confirm = emit_join_confirm
        self._emit_enter_response = emit_enter_response
        self._select_send_result = select_send_result
        self._enter_send_result = enter_send_result
        self._selected_room: str | None = None
        self._entered_room: str | None = None
        self.join_room_called = False
        self.enter_room_called = False
        self.selected_room_id: str | None = None
        self.entered_room_id: str | None = None

    def _captured_payloads(self) -> list[JSONValue]:
        """Return the synthetic captured raw-message buffer."""
        payloads: list[JSONValue] = [
            base64.b64encode(
                encode_frame(b"+4|World (President Trump)|24|5,1,0,0,0,0,0|2|n|field24.gif|2026")
            ).decode("utf-8")
        ]
        if self._include_practice_room:
            payloads.append(
                base64.b64encode(
                    encode_frame(b"+1|Practice|1|0,0,0,0,0,0,0|2|p|field01.gif|2026")
                ).decode("utf-8")
            )
        if self._emit_join_confirm and self._selected_room is not None:
            confirm = f"={self._selected_room}|Sep. 25, 2012|Artax|4|9|9|9|9".encode()
            payloads.append(base64.b64encode(encode_frame(confirm)).decode("utf-8"))
        if self._emit_enter_response and self._entered_room is not None:
            response = f"${self._entered_room}|0".encode()
            payloads.append(base64.b64encode(encode_frame(response)).decode("utf-8"))
        return payloads

    def _handle_websocket_send(self, expression: str) -> JSONObject | None:
        """Handle injected WebSocket send expressions.

        Args:
            expression: JavaScript expression under evaluation.

        Returns:
            CDP Runtime.evaluate result, or None when the expression is not a
            WebSocket send helper invocation.
        """
        if "window.__capturedWS" not in expression or "atob('" not in expression:
            return None
        match = re.search(r"atob\('([^']+)'\)", expression)
        if match is None:
            raise ValueError(f"missing websocket payload in expression: {expression}")
        framed = base64.b64decode(match.group(1))
        body, remaining = decode_frame(framed)
        if remaining:
            raise ValueError(f"unexpected trailing framed data: {remaining.hex()}")
        if body.startswith(b"*"):
            self.join_room_called = True
            self._selected_room = body[1:].decode("utf-8")
            self.selected_room_id = self._selected_room
            return {"result": {"value": self._select_send_result}}
        if body.startswith(b"+"):
            parts = body[1:].split(b"|", 4)
            if len(parts) != 5:
                raise ValueError(f"unexpected room enter payload: {body!r}")
            self.enter_room_called = True
            self._entered_room = parts[0].decode("utf-8")
            self.entered_room_id = self._entered_room
            if self._enter_send_result is not None:
                return {"result": {"value": self._enter_send_result}}
            return {"result": {"value": f"SENT_{len(framed)}_BYTES via wss://tankpit.com/ws/"}}
        raise ValueError(f"unexpected websocket send body: {body!r}")

    def _handle_runtime_evaluate(self, expression: str) -> JSONObject:
        """Handle Runtime.evaluate calls for login tests.

        Args:
            expression: JavaScript expression under evaluation.

        Returns:
            Synthetic CDP response object.
        """
        if "window.__rawMsgs" in expression:
            return {"result": {"value": self._captured_payloads()}}
        websocket_result = self._handle_websocket_send(expression)
        if websocket_result is not None:
            return websocket_result
        metadata_result = self._handle_metadata_evaluate(expression)
        if metadata_result is not None:
            return metadata_result
        error_result = self._handle_error_evaluate(expression)
        if error_result is not None:
            return error_result
        if "#login" in expression:
            self._account_login_started = True
        if "login-username" in expression and "offsetParent" in expression:
            return {"result": {"value": "ready"}}
        return {"result": {"value": "success"}}

    def _handle_metadata_evaluate(self, expression: str) -> JSONObject | None:
        """Handle metadata-related runtime lookups for login tests."""
        if "tankpit.magic" in expression:
            return {"result": {"value": "test_magic_12345678"}}
        if "script[src]" in expression and "tpclient" in expression:
            return {"result": {"value": "https://tankpit.com/game/tpclient-test.js"}}
        if "fetch(" in expression and "tpclient-test.js" in expression:
            static_key = "A" * 1000
            return {"result": {"value": f'window.fakeTpclientKey="{static_key}";'}}
        return None

    def _handle_error_evaluate(self, expression: str) -> JSONObject | None:
        """Handle error-banner queries for guest and account login tests."""
        if "errors" not in expression and "error" not in expression:
            return None
        if self._rate_limited and not self._account_login_started:
            return {"result": {"value": "There are too many tanks"}}
        if self._account_login_started:
            return {"result": {"value": self._login_error}}
        return {"result": {"value": ""}}

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Send CDP command."""
        if method == "Runtime.evaluate":
            self._eval_count += 1
            expression = str(params.get("expression", "")) if params else ""
            return self._handle_runtime_evaluate(expression)
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler."""
        _ = (event, handler)

    def detach(self) -> None:
        """Detach CDP session."""


class FakeCDPLoginNonDictResult:
    """Fake CDP session that returns malformed raw-message snapshots.

    This tests the strict validation branch for the protocol-driven join path.
    """

    def __init__(self) -> None:
        """Initialize fake CDP session."""
        self.join_room_called = False

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Send CDP command returning malformed room-capture data."""
        if method == "Runtime.evaluate":
            expression = str(params.get("expression", "")) if params else ""
            if "window.__rawMsgs" in expression:
                return {"result": ["not", "a", "dict"]}
            return {"result": {"value": "success"}}
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler."""
        _ = (event, handler)

    def detach(self) -> None:
        """Detach CDP session."""


class FakePageLogin:
    """Fake page for login testing."""

    def __init__(
        self,
        *,
        start_url: str = "https://tankpit.com/play",
        stays_on_before_playing: bool = False,
    ) -> None:
        """Initialize fake page."""
        self._url = start_url
        self._stays_on_before_playing = stays_on_before_playing
        self._wait_count = 0

    @property
    def url(self) -> str:
        """Get current URL."""
        return self._url

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> ResponseProtocol | None:
        """Navigate to URL."""
        _ = (referer, timeout, wait_until)
        self._url = url
        return None

    def wait_for_timeout(self, timeout: float) -> None:
        """Wait for timeout."""
        _ = timeout
        self._wait_count += 1
        # After 2nd wait (post-submit), update URL unless staying on before-playing
        if (
            self._wait_count == 2
            and not self._stays_on_before_playing
            and "before-playing" in self._url
        ):
            self._url = "https://tankpit.com/play"

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Wait for an event - returns immediately in tests."""
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Wait for a JavaScript function - returns immediately in tests."""
        _ = (expression, timeout)

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)

    def evaluate(self, expression: str) -> JSONValue:
        """Evaluate JavaScript expression - returns empty list in tests."""
        _ = expression
        return []

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Get keyboard for typing."""
        return NoOpKeyboard()


class FakeCDPNonDictResult:
    """Fake CDP that returns non-dict results."""

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Send CDP command returning non-dict result."""
        _ = params
        if method == "Runtime.evaluate":
            # Return result as a list instead of dict
            return {"result": ["not", "a", "dict"]}
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler."""
        _ = (event, handler)

    def detach(self) -> None:
        """Detach CDP session."""
