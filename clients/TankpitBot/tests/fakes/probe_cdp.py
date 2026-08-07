"""Fake CDP session for the action-lab probes."""

from __future__ import annotations

import base64
from collections.abc import Callable

from platform_core.json_utils import (
    JSONObject,
)

from tankpit_bot.protocol.framing import decode_frame
from tests.fakes.payloads import (
    _FAKE_MAGIC,
    _FAKE_STATIC_KEY,
    _FAKE_TPCLIENT_URL,
    _build_captured_raw_messages,
    _extract_enter_room_id,
)


class FakeCDPSessionProbe:
    """Fake CDP session for probe testing that responds to WebSocket sends.

    Now detects WebSocket injection (ws.send) instead of JavaScript KeyboardEvents,
    since the probe uses WebSocket injection for sending commands.
    """

    def __init__(
        self,
        *,
        emit_on_key: bool = True,
        emit_on_mouse: bool = False,
        viewport_result: JSONObject | None = None,
        return_invalid_result: bool = False,
        return_missing_value: bool = False,
        js_keypress_fails: bool = False,
    ) -> None:
        """Initialize fake CDP session for probing.

        Args:
            emit_on_key: Whether to emit messages when WebSocket commands are sent.
            emit_on_mouse: Whether to emit messages when mouse inputs are injected.
            viewport_result: Custom viewport result to return, None uses default.
            return_invalid_result: Return non-dict result for Runtime.evaluate.
            return_missing_value: Return dict without value for Runtime.evaluate.
            js_keypress_fails: If True, JS keypress returns ERROR instead of JS_KEYPRESS_X.
        """
        self._handlers: dict[str, list[Callable[[JSONObject], None]]] = {}
        self._sent_methods: list[str] = []
        self._detached = False
        self._emit_on_key = emit_on_key
        self._emit_on_mouse = emit_on_mouse
        self._viewport_result = viewport_result
        self._return_invalid_result = return_invalid_result
        self._return_missing_value = return_missing_value
        self._js_keypress_fails = js_keypress_fails
        self._input_count = 0
        self._ws_url = "wss://tankpit.com/ws/"
        self._selected_room: str | None = None
        self._entered_room: str | None = None
        self._raw_messages_ready = False

    def _handle_runtime_evaluate(self, params: JSONObject) -> JSONObject:
        """Handle Runtime.evaluate CDP command."""
        expression = params.get("expression", "")
        expr_str = str(expression)

        captured_raw_result = self._handle_captured_raw_messages(expr_str)
        if captured_raw_result is not None:
            return captured_raw_result

        viewport_result = self._handle_viewport_evaluate(expr_str)
        if viewport_result is not None:
            return viewport_result

        websocket_result = self._handle_websocket_evaluate(expr_str)
        if websocket_result is not None:
            return websocket_result

        if self._return_invalid_result:
            return {"error": "simulated error"}

        if self._return_missing_value:
            return {"result": {}}

        keypress_result = self._handle_js_keypress_evaluate(expr_str)
        if keypress_result is not None:
            return keypress_result

        return {"result": {"value": "success"}}

    def _handle_captured_raw_messages(self, expression: str) -> JSONObject | None:
        """Handle synthetic raw-message snapshot queries."""
        if "window.__rawMsgs" not in expression:
            return None
        if not self._raw_messages_ready:
            return {"result": {"value": []}}
        return {
            "result": {
                "value": _build_captured_raw_messages(self._selected_room, self._entered_room)
            }
        }

    def _handle_viewport_evaluate(self, expression: str) -> JSONObject | None:
        """Handle viewport-size evaluation queries."""
        if "innerWidth" not in expression:
            return None
        if self._viewport_result is not None:
            return self._viewport_result
        return {"result": {"value": '{"w":800,"h":600}'}}

    def _handle_websocket_evaluate(self, expression: str) -> JSONObject | None:
        """Handle injected websocket send expressions."""
        if "window.__capturedWS" not in expression or "atob('" not in expression:
            if "tankpit.magic" in expression:
                return {"result": {"value": _FAKE_MAGIC}}
            if "script[src]" in expression and "tpclient" in expression:
                return {"result": {"value": _FAKE_TPCLIENT_URL}}
            if "fetch(" in expression and "tpclient-test.js" in expression:
                return {"result": {"value": f'window.fakeTpclientKey="{_FAKE_STATIC_KEY}";'}}
            return None
        framed = base64.b64decode(expression.split("atob('", 1)[1].split("')", 1)[0])
        body, remaining = decode_frame(framed)
        if remaining:
            raise ValueError(f"unexpected trailing framed data: {remaining.hex()}")
        if body.startswith(b"*"):
            self._selected_room = body[1:].decode("utf-8")
            return {"result": {"value": f"SENT_4_BYTES via {self._ws_url}"}}
        if body.startswith(b"+"):
            self._entered_room = _extract_enter_room_id(body)
            return {"result": {"value": f"SENT_{len(framed)}_BYTES via {self._ws_url}"}}
        if self._emit_on_key:
            self._input_count += 1
            self._emit_ws_sent(f"key_input_{self._input_count}")
            self._emit_ws_received(f"key_response_{self._input_count}")
        return {"result": {"value": f"SENT_{len(framed)}_BYTES via {self._ws_url}"}}

    def _handle_js_keypress_evaluate(self, expression: str) -> JSONObject | None:
        """Handle DOM keypress fallback used to close toggle menus."""
        if "KeyboardEvent" not in expression or "dispatchEvent" not in expression:
            return None
        if self._js_keypress_fails:
            return {"result": {"value": "ERROR"}}
        if "'f'" in expression or '"f"' in expression:
            return {"result": {"value": "JS_KEYPRESS_F"}}
        return {"result": {"value": "JS_KEYPRESS_?"}}

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Send CDP command and optionally emit WebSocket response."""
        self._sent_methods.append(method)

        if method == "Runtime.evaluate" and params is not None:
            return self._handle_runtime_evaluate(params)

        if method == "Input.dispatchMouseEvent" and self._emit_on_mouse:
            event_type = params.get("type", "") if params else ""
            if event_type == "mousePressed":
                self._input_count += 1
                self._emit_ws_sent(f"mouse_input_{self._input_count}")
                self._emit_ws_received(f"mouse_response_{self._input_count}")

        return {"result": {"value": ""}}

    def _emit_ws_sent(self, payload: str) -> None:
        """Emit a WebSocket sent event."""
        if "Network.webSocketFrameSent" in self._handlers:
            for handler in self._handlers["Network.webSocketFrameSent"]:
                handler(
                    {
                        "requestId": "1.1",
                        "timestamp": 100.0 + self._input_count,
                        "response": {"opcode": 1, "mask": True, "payloadData": payload},
                    }
                )

    def _emit_ws_received(self, payload: str) -> None:
        """Emit a WebSocket received event."""
        if "Network.webSocketFrameReceived" in self._handlers:
            for handler in self._handlers["Network.webSocketFrameReceived"]:
                handler(
                    {
                        "requestId": "1.1",
                        "timestamp": 100.0 + self._input_count + 0.001,
                        "response": {"opcode": 1, "mask": False, "payloadData": payload},
                    }
                )

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def detach(self) -> None:
        """Detach CDP session."""
        self._detached = True

    def emit_event(self, event: str, params: JSONObject) -> None:
        """Emit a CDP event for testing."""
        if event in self._handlers:
            for handler in self._handlers[event]:
                handler(params)
