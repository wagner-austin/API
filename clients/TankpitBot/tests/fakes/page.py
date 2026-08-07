"""Fake page, keyboard, and response objects.

The page variants differ only in what the captured-message buffer
returns: full, empty, or growing between polls.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONValue,
)

from tankpit_bot._test_hooks import (
    KeyboardProtocol,
    ResponseProtocol,
)
from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot.types import CapturedMessage
from tests.fakes.cdp import (
    FakeCDPSession,
    FakeCDPSessionRateLimited,
)
from tests.fakes.payloads import _make_auth_payload


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
