"""Fake CDP sessions.

The standard fake and the rate-limited variant that replays the
throttling the real endpoint applies.
"""

from __future__ import annotations

from collections.abc import Callable

from platform_core.json_utils import (
    JSONObject,
)

from tankpit_bot.protocol.command_builders import build_query_command
from tankpit_bot.protocol.commands import CMD_ENTER_GAME
from tests.fakes.payloads import (
    _decode_injected_websocket_body,
    _extract_enter_room_id,
    _extract_injected_websocket_payload_data,
    _runtime_metadata_result,
    _runtime_raw_messages_result,
)


class FakeCDPSession:
    """Fake Playwright CDPSession."""

    def __init__(self, *, emit_runtime_frames: bool = True) -> None:
        """Initialize fake CDP session."""
        self._handlers: dict[str, list[Callable[[JSONObject], None]]] = {}
        self._sent_methods: list[str] = []
        self._detached = False
        self._selected_room: str | None = None
        self._entered_room: str | None = None
        self._ws_url = "wss://tankpit.com/ws/"
        self._runtime_frame_count = 0
        self._raw_messages_ready = False
        self._emit_runtime_frames = emit_runtime_frames
        self._page_text = ""
        self._page_text_queue: list[str] = []

    def set_page_text(self, page_text: str) -> None:
        """Set the rendered page text returned for body-text scrapes.

        Args:
            page_text: Value returned for ``document.body`` evaluate
                expressions (the ``C`` statistics panel scrape).
        """
        self._page_text = page_text
        self._page_text_queue = []

    def set_page_text_sequence(self, page_texts: list[str]) -> None:
        """Queue successive page texts for consecutive body-text scrapes.

        Each ``document.body`` evaluate consumes the next entry; the
        final entry then repeats. Models a panel that paints
        incrementally across scrapes (the ``C`` statistics panel).

        Args:
            page_texts: Non-empty ordered scrape results.
        """
        self._page_text_queue = list(page_texts)

    def _emit_runtime_frame_sent(self, payload: str) -> None:
        """Emit a synthetic websocket-sent CDP event for injected payloads."""
        if not self._emit_runtime_frames:
            return
        self._runtime_frame_count += 1
        if "Network.webSocketFrameSent" not in self._handlers:
            return
        event: JSONObject = {
            "requestId": "1.1",
            "timestamp": 200.0 + self._runtime_frame_count,
            "response": {"opcode": 1, "mask": True, "payloadData": payload},
        }
        for handler in self._handlers["Network.webSocketFrameSent"]:
            handler(event)

    def _handle_runtime_injected_websocket(self, expression: str) -> JSONObject | None:
        """Handle websocket helper sends from Runtime.evaluate."""
        body = _decode_injected_websocket_body(expression)
        if body is None:
            return None
        payload_data = _extract_injected_websocket_payload_data(expression)
        if payload_data is not None:
            self._emit_runtime_frame_sent(payload_data)
        if body.startswith(b"*"):
            self._selected_room = body[1:].decode("utf-8")
            return {"result": {"value": f"SENT_4_BYTES via {self._ws_url}"}}
        if body.startswith(b"+"):
            self._entered_room = _extract_enter_room_id(body)
            return {"result": {"value": f"SENT_{len(body) + 2}_BYTES via {self._ws_url}"}}
        if body == build_query_command(CMD_ENTER_GAME)[2:]:
            return {"result": {"value": f"SENT_5_BYTES via {self._ws_url}"}}
        return {"result": {"value": f"SENT_4_BYTES via {self._ws_url}"}}

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Send CDP command.

        Returns a valid CDP response with ``{"result": {"value": ...}}``,
        matching the real Chrome DevTools Protocol contract. The
        ``Browser.getWindowForTarget`` / ``Browser.setWindowBounds``
        pair used by ``_maximize_via_cdp`` returns a stable
        ``windowId`` so the streamed-display bootstrap path can be
        exercised through this fake.
        """
        self._sent_methods.append(method)
        if method == "Browser.getWindowForTarget":
            return {"windowId": 1}
        if method != "Runtime.evaluate" or params is None:
            return {"result": {"value": ""}}
        expression = str(params.get("expression", ""))
        if "document.body" in expression:
            if self._page_text_queue:
                self._page_text = self._page_text_queue.pop(0)
            return {"result": {"value": self._page_text}}
        raw_messages_result = _runtime_raw_messages_result(
            expression,
            raw_messages_ready=self._raw_messages_ready,
            selected_room=self._selected_room,
            entered_room=self._entered_room,
        )
        if raw_messages_result is not None:
            return raw_messages_result
        metadata_result = _runtime_metadata_result(expression)
        if metadata_result is not None:
            return metadata_result
        injected_result = self._handle_runtime_injected_websocket(expression)
        if injected_result is not None:
            return injected_result
        return {"result": {"value": ""}}

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


class FakeCDPSessionRateLimited:
    """Fake CDP session that simulates rate-limiting error then successful login."""

    def __init__(self, *, login_fails: bool = False, emit_runtime_frames: bool = True) -> None:
        """Initialize fake CDP session."""
        self._handlers: dict[str, list[Callable[[JSONObject], None]]] = {}
        self._sent_methods: list[str] = []
        self._detached = False
        self._eval_count = 0
        self._login_fails = login_fails
        self._selected_room: str | None = None
        self._entered_room: str | None = None
        self._ws_url = "wss://tankpit.com/ws/"
        self._runtime_frame_count = 0
        self._raw_messages_ready = False
        self._emit_runtime_frames = emit_runtime_frames

    def _emit_runtime_frame_sent(self, payload: str) -> None:
        """Emit a synthetic websocket-sent CDP event for injected payloads."""
        if not self._emit_runtime_frames:
            return
        self._runtime_frame_count += 1
        if "Network.webSocketFrameSent" not in self._handlers:
            return
        event: JSONObject = {
            "requestId": "1.1",
            "timestamp": 200.0 + self._runtime_frame_count,
            "response": {"opcode": 1, "mask": True, "payloadData": payload},
        }
        for handler in self._handlers["Network.webSocketFrameSent"]:
            handler(event)

    def _handle_runtime_fixture(self, expression: str) -> JSONObject | None:
        """Handle raw-message, metadata, and websocket helper runtime calls."""
        raw_messages_result = _runtime_raw_messages_result(
            expression,
            raw_messages_ready=self._raw_messages_ready,
            selected_room=self._selected_room,
            entered_room=self._entered_room,
        )
        if raw_messages_result is not None:
            return raw_messages_result
        metadata_result = _runtime_metadata_result(expression)
        if metadata_result is not None:
            return metadata_result
        return self._handle_runtime_injected_websocket(expression)

    def _handle_runtime_injected_websocket(self, expression: str) -> JSONObject | None:
        """Handle injected websocket sends for the rate-limited fake session."""
        body = _decode_injected_websocket_body(expression)
        if body is None:
            return None
        payload_data = _extract_injected_websocket_payload_data(expression)
        if payload_data is not None:
            self._emit_runtime_frame_sent(payload_data)
        if body.startswith(b"*"):
            self._selected_room = body[1:].decode("utf-8")
            return {"result": {"value": f"SENT_4_BYTES via {self._ws_url}"}}
        if body.startswith(b"+"):
            self._entered_room = _extract_enter_room_id(body)
            return {"result": {"value": f"SENT_{len(body) + 2}_BYTES via {self._ws_url}"}}
        if body == build_query_command(CMD_ENTER_GAME)[2:]:
            return {"result": {"value": f"SENT_5_BYTES via {self._ws_url}"}}
        return None

    def _handle_runtime_default(self) -> JSONObject:
        """Handle the guest/account login polling sequence."""
        self._eval_count += 1
        if self._eval_count == 3:
            return {"result": {"value": "There are too many tanks"}}
        if self._eval_count == 7:
            if self._login_fails:
                return {"result": {"value": "Invalid username or password"}}
            return {"result": {"value": ""}}
        return {"result": {"value": "success"}}

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Send CDP command, returning rate limit error on 3rd Runtime.evaluate."""
        self._sent_methods.append(method)
        if method == "Browser.getWindowForTarget":
            return {"windowId": 1}
        if method != "Runtime.evaluate":
            result: JSONObject = {}
            return result
        expression = str(params.get("expression", "")) if params is not None else ""
        fixture_result = self._handle_runtime_fixture(expression)
        if fixture_result is not None:
            return fixture_result
        return self._handle_runtime_default()

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
