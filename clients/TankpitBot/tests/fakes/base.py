"""Base fake Playwright classes for testing.

Provides core fake implementations of Playwright protocols that don't require
real browser installation. All fakes match the protocol signatures in
tankpit_bot._test_hooks.
"""

from __future__ import annotations

import base64
import re
import types
from collections.abc import Callable

from platform_core.json_utils import JSONObject, JSONValue

from tankpit_bot._test_hooks import (
    BrowserContextProtocol,
    BrowserProtocol,
    BrowserTypeProtocol,
    CDPSessionProtocol,
    KeyboardProtocol,
    PageProtocol,
    PlaywrightProtocol,
    ResponseProtocol,
    SyncPlaywrightContextManagerProtocol,
)
from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot.protocol.commands import CMD_ENTER_GAME, build_query_command
from tankpit_bot.protocol.framing import decode_frame, encode_frame
from tankpit_bot.types import CapturedMessage

_FAKE_TPCLIENT_URL = "https://tankpit.com/game/tpclient-test.js"
_FAKE_STATIC_KEY = "A" * 1000
_FAKE_MAGIC = "test_magic_12345678"


def _make_auth_payload(magic: str) -> str:
    """Create a base64-encoded AUTH payload containing the magic key.

    The AUTH message format is:
    - 2-byte length prefix (little-endian)
    - Text body: %AUTH !be <session>|<hash>|<ts> <magic>

    Args:
        magic: The magic key to include in the AUTH payload.

    Returns:
        Base64-encoded AUTH payload string.
    """
    body = f"%AUTH !be test_session|test_hash|12345 {magic}"
    body_bytes = body.encode("utf-8")
    length_prefix = len(body_bytes).to_bytes(2, "little")
    return base64.b64encode(length_prefix + body_bytes).decode("ascii")


_WEBSOCKET_INJECTION_PATTERN = re.compile(r"atob\('([^']+)'\)")


def _build_captured_raw_messages(
    selected_room: str | None,
    entered_room: str | None = None,
) -> list[JSONValue]:
    """Build the synthetic raw-message buffer used by protocol join tests.

    Args:
        selected_room: Joined room ID, if any.
        entered_room: Entered room ID, if any.

    Returns:
        Base64-encoded framed room-discovery payloads.
    """
    payloads: list[JSONValue] = [
        base64.b64encode(
            encode_frame(b"+4|World (President Trump)|24|5,1,0,0,0,0,0|2|n|field24.gif|2026")
        ).decode("utf-8"),
        base64.b64encode(encode_frame(b"+1|Practice|1|0,0,0,0,0,0,0|2|p|field01.gif|2026")).decode(
            "utf-8"
        ),
    ]
    if selected_room is not None:
        confirm = f"={selected_room}|Sep. 25, 2012|Artax|4|9|9|9|9".encode()
        payloads.append(base64.b64encode(encode_frame(confirm)).decode("utf-8"))
    if entered_room is not None:
        response = f"${entered_room}|0".encode()
        payloads.append(base64.b64encode(encode_frame(response)).decode("utf-8"))
    return payloads


def _extract_injected_websocket_payload_data(expression: str) -> str | None:
    """Return the base64 websocket payload injected by the runtime helper.

    Args:
        expression: JavaScript passed to ``Runtime.evaluate``.

    Returns:
        Base64 payload string, or ``None`` when the helper is not used.
    """
    if "window.__capturedWS" not in expression or "atob('" not in expression:
        return None
    match = _WEBSOCKET_INJECTION_PATTERN.search(expression)
    if match is None:
        raise ValueError(f"missing websocket payload in expression: {expression}")
    return match.group(1)


def _extract_enter_room_id(body: bytes) -> str:
    """Return the room ID from a ``+room|...`` enter packet body.

    Args:
        body: Decoded websocket body beginning with ``+``.

    Returns:
        Room ID portion of the enter packet.
    """
    parts = body[1:].split(b"|", 4)
    if len(parts) != 5:
        raise ValueError(f"unexpected room enter payload: {body!r}")
    return parts[0].decode("utf-8")


def _runtime_raw_messages_result(
    expression: str,
    *,
    raw_messages_ready: bool,
    selected_room: str | None,
    entered_room: str | None,
) -> JSONObject | None:
    """Return the fake ``window.__rawMsgs`` snapshot response."""
    if "window.__rawMsgs" not in expression:
        return None
    if not raw_messages_ready:
        return {"result": {"value": []}}
    return {
        "result": {
            "value": _build_captured_raw_messages(
                selected_room,
                entered_room,
            )
        }
    }


_PAGE_CLIENT_SNAPSHOT_VALUE: JSONObject = {
    "timestamp_ms": 1000,
    "client_present": True,
    "map_visible": False,
    "client_state": 1,
    "client_busy": False,
    "pending_actions": 0,
    "heartbeat_age_ms": 50,
    "last_page_client_send_age_ms": 100,
    "last_bot_send_age_ms": 100,
    "ws_ready_state": 1,
    "current_send_label": None,
    "sent_frame_meta_queue_length": 0,
    "self_fields": {},
    "world_fields": {},
    "map_fields": {},
    "world_collections": {},
}


def _runtime_metadata_result(expression: str) -> JSONObject | None:
    """Return fake metadata/script lookup responses for login helpers."""
    if "MAX_DEPTH" in expression:
        # Structure survey walk: the real page returns null until the
        # game object exists; the fake models the pre-capture state.
        return {"result": {"value": None}}
    if "tankpit.magic" in expression:
        return {"result": {"value": _FAKE_MAGIC}}
    if "script[src]" in expression and "tpclient" in expression:
        return {"result": {"value": _FAKE_TPCLIENT_URL}}
    if "fetch(" in expression and "tpclient-test.js" in expression:
        return {"result": {"value": f'window.fakeTpclientKey="{_FAKE_STATIC_KEY}";'}}
    if "window.__tankpitActiveGame" in expression and "map_visible" in expression:
        return {"result": {"value": _PAGE_CLIENT_SNAPSHOT_VALUE}}
    return None


def _decode_injected_websocket_body(expression: str) -> bytes | None:
    """Decode the browser helper's injected websocket payload.

    Args:
        expression: JavaScript passed to ``Runtime.evaluate``.

    Returns:
        The decoded framed-message body for framed payloads, the raw byte
        payload for non-framed sends, or ``None`` when the expression does not
        invoke the websocket helper.
    """
    if "window.__capturedWS" not in expression or "atob('" not in expression:
        return None
    match = _WEBSOCKET_INJECTION_PATTERN.search(expression)
    if match is None:
        raise ValueError(f"missing websocket payload in expression: {expression}")
    framed = base64.b64decode(match.group(1))
    if len(framed) < 2:
        return framed
    expected_total = 2 + int.from_bytes(framed[:2], "little")
    if expected_total != len(framed):
        return framed
    body, remaining = decode_frame(framed)
    if remaining:
        raise ValueError(f"unexpected trailing framed data: {remaining.hex()}")
    return body


class FakeResponse:
    """Fake Playwright Response."""

    def __init__(self, status: int = 200, url: str = "https://example.com") -> None:
        """Initialize fake response.

        Args:
            status: HTTP status code.
            url: Response URL.
        """
        self._status = status
        self._url = url

    @property
    def status(self) -> int:
        """Get status code."""
        return self._status

    @property
    def url(self) -> str:
        """Get URL."""
        return self._url


class FakeKeyboard:
    """Fake Playwright Keyboard for testing.

    In tests, keyboard input is typically handled via CDP synthetic events,
    but this class satisfies the KeyboardProtocol for type checking.
    """

    def press(self, key: str, *, delay: float | None = None) -> None:
        """Press a keyboard key (no-op in tests).

        Args:
            key: Key name.
            delay: Time between keydown and keyup.
        """
        _ = (key, delay)

    def type(self, text: str, *, delay: float | None = None) -> None:
        """Type text (no-op in tests).

        Args:
            text: Text to type.
            delay: Time between key presses.
        """
        _ = (text, delay)


class InMemoryTerrainMap:
    """In-memory ``TerrainMapProtocol`` for tests.

    Real implementation backed by a ``dict[(x, y), str]`` plus a
    configurable default tile. Callers seed the coordinates they care
    about; every other tile resolves to ``default``. Not a fake of
    behavior; this is what a small terrain map would look like in
    production if the data fit in memory.

    The ``default`` parameter lifts two common test shapes into one
    class: "all ground" (the default) and "passable set as ground,
    rest as water" (via :meth:`from_passable_set`).
    """

    ROCK: str = "#"
    GROUND: str = "."
    WATER: str = "W"

    def __init__(
        self,
        terrain_data: dict[tuple[int, int], str] | None = None,
        *,
        default: str = GROUND,
    ) -> None:
        """Initialize an in-memory terrain map.

        Args:
            terrain_data: Dict mapping ``(x, y)`` to terrain character.
                Unmapped tiles resolve to ``default``.
            default: Tile returned for any coordinate not present in
                ``terrain_data``. Defaults to ``GROUND``.
        """
        self._terrain_data = terrain_data or {}
        self._default = default

    @classmethod
    def from_passable_set(
        cls,
        passable: set[tuple[int, int]],
    ) -> InMemoryTerrainMap:
        """Build a map where ``passable`` tiles are ground, rest is water.

        Args:
            passable: Coordinates considered walkable ground.

        Returns:
            Terrain map whose ``passable`` tiles resolve to ``GROUND``
            and every other coordinate resolves to ``WATER``.
        """
        return cls(
            dict.fromkeys(passable, cls.GROUND),
            default=cls.WATER,
        )

    def get_terrain(self, x: int, y: int) -> str:
        """Get terrain at coordinates.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            Terrain character.
        """
        return self._terrain_data.get((x, y), self._default)

    def is_passable(self, x: int, y: int) -> bool:
        """Check if terrain is passable.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            True if passable.
        """
        terrain = self.get_terrain(x, y)
        return terrain not in (self.ROCK, self.WATER)

    def render_viewport(
        self,
        center_x: int,
        center_y: int,
        width: int = 16,
        height: int = 16,
    ) -> list[list[str]]:
        """Render viewport grid.

        Args:
            center_x: Center X.
            center_y: Center Y.
            width: Viewport width.
            height: Viewport height.

        Returns:
            2D list of terrain characters.
        """
        left = center_x - width // 2
        top = center_y - height // 2
        grid: list[list[str]] = []
        for row in range(height):
            row_data: list[str] = []
            for col in range(width):
                x = left + col
                y = top + row
                row_data.append(self.get_terrain(x, y))
            grid.append(row_data)
        return grid


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
        matching the real Chrome DevTools Protocol contract.
        """
        self._sent_methods.append(method)
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


class FakePage:
    """Fake Playwright Page that emits WebSocket events."""

    def __init__(
        self,
        cdp_session: FakeCDPSession | FakeCDPSessionRateLimited,
        *,
        script_urls: list[JSONValue] | None = None,
        magic: str | None = None,
    ) -> None:
        """Initialize fake page.

        Args:
            cdp_session: CDP session to use for events.
            script_urls: Optional list of script URLs to return from evaluate.
            magic: Optional magic key to embed in AUTH messages.
        """
        self._cdp_session = cdp_session
        self._goto_url: str | None = None
        self._wait_timeout: float | None = None
        self._closed = False
        self._url = ""
        self._script_urls: list[JSONValue] = script_urls if script_urls is not None else []
        self._magic = magic
        self._emitted_initial_messages = False

    @property
    def url(self) -> str:
        """Get the current URL of the page."""
        return self._url

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Get the keyboard interface.

        Returns:
            FakeKeyboard instance.
        """
        return FakeKeyboard()

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
        self._goto_url = url
        self._url = url
        return FakeResponse(url=url)

    def wait_for_timeout(self, timeout: float) -> None:
        """Wait and emit WebSocket events."""
        self._wait_timeout = timeout
        if self._emitted_initial_messages:
            return
        self._emitted_initial_messages = True
        self._cdp_session.emit_event(
            "Network.webSocketCreated",
            {"requestId": "1.1", "url": "wss://example.com/ws"},
        )
        # Emit AUTH message with magic if configured
        sent_payload = "sent message"
        if self._magic:
            sent_payload = _make_auth_payload(self._magic)
        self._cdp_session.emit_event(
            "Network.webSocketFrameSent",
            {
                "requestId": "1.1",
                "timestamp": 100.0,
                "response": {"opcode": 1, "mask": True, "payloadData": sent_payload},
            },
        )
        self._cdp_session.emit_event(
            "Network.webSocketFrameReceived",
            {
                "requestId": "1.1",
                "timestamp": 100.5,
                "response": {"opcode": 1, "mask": False, "payloadData": "received message"},
            },
        )
        self._cdp_session._raw_messages_ready = True

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Wait for an event - also emit WebSocket events like wait_for_timeout."""
        _ = (event, timeout)
        if self._emitted_initial_messages:
            return
        self._emitted_initial_messages = True
        # Emit the same events as wait_for_timeout for test compatibility
        self._cdp_session.emit_event(
            "Network.webSocketCreated",
            {"requestId": "1.1", "url": "wss://example.com/ws"},
        )
        # Emit AUTH message with magic if configured
        sent_payload = "sent message"
        if self._magic:
            sent_payload = _make_auth_payload(self._magic)
        self._cdp_session.emit_event(
            "Network.webSocketFrameSent",
            {
                "requestId": "1.1",
                "timestamp": 100.0,
                "response": {"opcode": 1, "mask": True, "payloadData": sent_payload},
            },
        )
        self._cdp_session.emit_event(
            "Network.webSocketFrameReceived",
            {
                "requestId": "1.1",
                "timestamp": 100.5,
                "response": {"opcode": 1, "mask": False, "payloadData": "received message"},
            },
        )
        self._cdp_session._raw_messages_ready = True

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Wait for JavaScript function to return truthy.

        Args:
            expression: JavaScript expression to evaluate.
            timeout: Maximum wait time in milliseconds.
        """
        _ = (expression, timeout)

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)
        self._closed = True

    def evaluate(self, expression: str) -> JSONValue:
        """Evaluate JavaScript expression.

        Returns:
            Script URLs list for script queries, empty list otherwise.
        """
        _ = expression
        return self._script_urls


class FakePageNoMessages:
    """Fake Playwright Page that doesn't emit WebSocket messages."""

    def __init__(self, cdp_session: FakeCDPSession | FakeCDPSessionRateLimited) -> None:
        """Initialize fake page."""
        self._cdp_session = cdp_session
        self._closed = False
        self._url = ""

    @property
    def url(self) -> str:
        """Get the current URL of the page."""
        return self._url

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Get the keyboard interface.

        Returns:
            FakeKeyboard instance.
        """
        return FakeKeyboard()

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
        return FakeResponse()

    def wait_for_timeout(self, timeout: float) -> None:
        """Wait without emitting any WebSocket events."""
        _ = timeout

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Wait for an event - returns immediately in tests."""
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Wait for JavaScript function to return truthy.

        Args:
            expression: JavaScript expression to evaluate.
            timeout: Maximum wait time in milliseconds.
        """
        _ = (expression, timeout)

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)
        self._closed = True

    def evaluate(self, expression: str) -> JSONValue:
        """Evaluate JavaScript expression - returns empty list in tests."""
        _ = expression
        return []


class FakePageGrowingMessages:
    """Fake page that appends messages to a list during wait_for_timeout.

    Used to test stabilization reset in _wait_for_game_ready, where new
    messages arrive during the stabilization loop.
    """

    def __init__(
        self,
        messages: list[CapturedMessage],
        *,
        add_on_call: int = 2,
    ) -> None:
        """Initialize fake page.

        Args:
            messages: Mutable message list (shared with the session under test).
            add_on_call: Which wait_for_timeout call number triggers a new message.
        """
        self._messages = messages
        self._add_on_call = add_on_call
        self._call_count = 0
        self._url = ""
        self._closed = False

    @property
    def url(self) -> str:
        """Get the current URL of the page."""
        return self._url

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Get the keyboard interface.

        Returns:
            FakeKeyboard instance.
        """
        return FakeKeyboard()

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
        return FakeResponse()

    def wait_for_timeout(self, timeout: float) -> None:
        """Wait and optionally add a message to trigger stabilization reset."""
        _ = timeout
        self._call_count += 1
        if self._call_count == self._add_on_call:
            self._messages.append(
                CapturedMessage(
                    timestamp_ms=self._call_count,
                    direction="received",
                    payload="growing",
                    ws_url="ws://test",
                ),
            )

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Wait for an event - returns immediately in tests."""
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Wait for JavaScript function to return truthy.

        Args:
            expression: JavaScript expression to evaluate.
            timeout: Maximum wait time in milliseconds.
        """
        _ = (expression, timeout)

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)
        self._closed = True

    def evaluate(self, expression: str) -> JSONValue:
        """Evaluate JavaScript expression - returns empty list in tests."""
        _ = expression
        return []


class FakeBrowserContext:
    """Fake Playwright BrowserContext."""

    def __init__(
        self,
        *,
        emit_messages: bool = True,
        rate_limited: bool = False,
        login_fails: bool = False,
        script_urls: list[JSONValue] | None = None,
        magic: str | None = None,
    ) -> None:
        """Initialize fake browser context.

        Args:
            emit_messages: Whether to emit WebSocket messages.
            rate_limited: Whether to simulate rate limiting.
            login_fails: Whether login should fail.
            script_urls: Script URLs to return from page.evaluate().
            magic: Magic key to embed in AUTH messages.
        """
        cdp: FakeCDPSession | FakeCDPSessionRateLimited = (
            FakeCDPSessionRateLimited(
                login_fails=login_fails,
                emit_runtime_frames=emit_messages,
            )
            if rate_limited
            else FakeCDPSession(emit_runtime_frames=emit_messages)
        )
        self._cdp_session = cdp
        if not emit_messages:
            self._cdp_session._raw_messages_ready = True
        self._pages: list[FakePage | FakePageNoMessages] = []
        self._closed = False
        self._emit_messages = emit_messages
        self._rate_limited = rate_limited
        self._script_urls = script_urls
        self._magic = magic

    def new_page(self) -> PageProtocol:
        """Create new page."""
        page: FakePage | FakePageNoMessages
        if self._emit_messages:
            page = FakePage(self._cdp_session, script_urls=self._script_urls, magic=self._magic)
        else:
            page = FakePageNoMessages(self._cdp_session)
        self._pages.append(page)
        return page

    def new_cdp_session(self, page: PageProtocol) -> CDPSessionProtocol:
        """Create CDP session for page."""
        _ = page
        return self._cdp_session

    def close(self, *, reason: str | None = None) -> None:
        """Close context."""
        _ = reason
        self._closed = True


class FakeBrowser:
    """Fake Playwright Browser."""

    def __init__(
        self,
        *,
        emit_messages: bool = True,
        rate_limited: bool = False,
        login_fails: bool = False,
        script_urls: list[JSONValue] | None = None,
        magic: str | None = None,
    ) -> None:
        """Initialize fake browser.

        Args:
            emit_messages: Whether to emit WebSocket messages.
            rate_limited: Whether to simulate rate limiting.
            login_fails: Whether login should fail.
            script_urls: Script URLs to return from page.evaluate().
            magic: Magic key to embed in AUTH messages.
        """
        self._contexts: list[FakeBrowserContext] = []
        self._closed = False
        self._emit_messages = emit_messages
        self._rate_limited = rate_limited
        self._login_fails = login_fails
        self._script_urls = script_urls
        self._magic = magic

    def new_context(self) -> BrowserContextProtocol:
        """Create new context."""
        ctx = FakeBrowserContext(
            emit_messages=self._emit_messages,
            rate_limited=self._rate_limited,
            login_fails=self._login_fails,
            script_urls=self._script_urls,
            magic=self._magic,
        )
        self._contexts.append(ctx)
        return ctx

    def close(self, *, reason: str | None = None) -> None:
        """Close browser."""
        _ = reason
        self._closed = True


class FakeBrowserType:
    """Fake Playwright BrowserType."""

    def __init__(
        self,
        *,
        emit_messages: bool = True,
        rate_limited: bool = False,
        login_fails: bool = False,
        script_urls: list[JSONValue] | None = None,
        magic: str | None = None,
    ) -> None:
        """Initialize fake browser type.

        Args:
            emit_messages: Whether to emit WebSocket messages.
            rate_limited: Whether to simulate rate limiting.
            login_fails: Whether login should fail.
            script_urls: Script URLs to return from page.evaluate().
            magic: Magic key to embed in AUTH messages.
        """
        self._browsers: list[FakeBrowser] = []
        self._emit_messages = emit_messages
        self._rate_limited = rate_limited
        self._login_fails = login_fails
        self._script_urls = script_urls
        self._magic = magic

    def launch(
        self,
        *,
        headless: bool | None = None,
        slow_mo: float | None = None,
        timeout: float | None = None,
    ) -> BrowserProtocol:
        """Launch browser."""
        _ = (headless, slow_mo, timeout)
        browser = FakeBrowser(
            emit_messages=self._emit_messages,
            rate_limited=self._rate_limited,
            login_fails=self._login_fails,
            script_urls=self._script_urls,
            magic=self._magic,
        )
        self._browsers.append(browser)
        return browser


class FakePlaywright:
    """Fake Playwright instance."""

    def __init__(
        self,
        *,
        emit_messages: bool = True,
        rate_limited: bool = False,
        login_fails: bool = False,
        script_urls: list[JSONValue] | None = None,
        magic: str | None = None,
    ) -> None:
        """Initialize fake Playwright.

        Args:
            emit_messages: Whether to emit WebSocket messages.
            rate_limited: Whether to simulate rate limiting.
            login_fails: Whether login should fail.
            script_urls: Script URLs to return from page.evaluate().
            magic: Magic key to embed in AUTH messages.
        """
        self._chromium = FakeBrowserType(
            emit_messages=emit_messages,
            rate_limited=rate_limited,
            login_fails=login_fails,
            script_urls=script_urls,
            magic=magic,
        )
        self._stopped = False

    @property
    def chromium(self) -> BrowserTypeProtocol:
        """Get chromium browser type."""
        return self._chromium

    def stop(self) -> None:
        """Stop Playwright."""
        self._stopped = True


class FakeSyncPlaywrightContextManager:
    """Fake sync_playwright() context manager."""

    def __init__(
        self,
        *,
        emit_messages: bool = True,
        rate_limited: bool = False,
        login_fails: bool = False,
        script_urls: list[JSONValue] | None = None,
        magic: str | None = None,
    ) -> None:
        """Initialize fake context manager.

        Args:
            emit_messages: Whether to emit WebSocket messages.
            rate_limited: Whether to simulate rate limiting.
            login_fails: Whether login should fail.
            script_urls: Script URLs to return from page.evaluate().
            magic: Magic key to embed in AUTH messages.
        """
        self._playwright: FakePlaywright | None = None
        self._emit_messages = emit_messages
        self._rate_limited = rate_limited
        self._login_fails = login_fails
        self._script_urls = script_urls
        self._magic = magic

    def start(self) -> PlaywrightProtocol:
        """Start Playwright."""
        self._playwright = FakePlaywright(
            emit_messages=self._emit_messages,
            rate_limited=self._rate_limited,
            login_fails=self._login_fails,
            script_urls=self._script_urls,
            magic=self._magic,
        )
        return self._playwright

    def __enter__(self) -> PlaywrightProtocol:
        """Enter context."""
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit context."""
        _ = (exc_type, exc_val, exc_tb)
        if self._playwright is not None:
            self._playwright.stop()


def fake_sync_playwright() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright that emits WebSocket messages."""
    return FakeSyncPlaywrightContextManager(emit_messages=True)


def fake_sync_playwright_no_messages() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright that doesn't emit messages."""
    return FakeSyncPlaywrightContextManager(emit_messages=False)


def fake_sync_playwright_rate_limited() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright that simulates rate-limiting with successful login."""
    return FakeSyncPlaywrightContextManager(emit_messages=True, rate_limited=True)


def fake_sync_playwright_login_fails() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright that simulates rate-limiting with failed login."""
    return FakeSyncPlaywrightContextManager(emit_messages=True, rate_limited=True, login_fails=True)


def fake_sync_playwright_with_scripts() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright that returns script URLs from page.evaluate().

    Returns:
        Context manager that produces pages returning script URLs.
    """
    return FakeSyncPlaywrightContextManager(
        emit_messages=True,
        script_urls=[
            "https://tankpit.com/js/game.js",
            "https://tankpit.com/js/protocol.js",
        ],
    )


def fake_sync_playwright_with_mixed_scripts() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright with mixed types in script_urls list.

    This tests the isinstance(url, str) check by including non-string values.

    Returns:
        Context manager that produces pages returning mixed script URL types.
    """
    return FakeSyncPlaywrightContextManager(
        emit_messages=True,
        script_urls=[
            "https://tankpit.com/js/valid.js",
            123,  # Non-string value to test isinstance check
            None,  # Another non-string value
            "https://tankpit.com/js/another.js",
        ],
    )


def fake_sync_playwright_with_magic() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright that emits AUTH messages with magic key.

    The magic key is embedded in AUTH messages emitted during WebSocket events,
    allowing tests to verify magic extraction from AUTH message payloads.

    Returns:
        Context manager that produces pages emitting AUTH messages with magic.
    """
    return FakeSyncPlaywrightContextManager(
        emit_messages=True,
        magic="test_magic_xor_key_value",
    )


__all__ = [
    "FakeBrowser",
    "FakeBrowserContext",
    "FakeBrowserType",
    "FakeCDPSession",
    "FakeCDPSessionRateLimited",
    "FakeKeyboard",
    "FakePage",
    "FakePageNoMessages",
    "FakePlaywright",
    "FakeResponse",
    "FakeSyncPlaywrightContextManager",
    "InMemoryTerrainMap",
    "_make_auth_payload",
    "fake_sync_playwright",
    "fake_sync_playwright_login_fails",
    "fake_sync_playwright_no_messages",
    "fake_sync_playwright_rate_limited",
    "fake_sync_playwright_with_magic",
    "fake_sync_playwright_with_mixed_scripts",
    "fake_sync_playwright_with_scripts",
]
