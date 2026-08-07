"""Fake probe pages.

The two variants differ only in whether the captured-message buffer
returns anything.
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
from tests.fakes.page import (
    FakeKeyboard,
    FakeResponse,
)
from tests.fakes.payloads import (
    _make_auth_payload,
)
from tests.fakes.probe_cdp import FakeCDPSessionProbe


class FakePageProbe:
    """Fake Playwright Page for probe testing."""

    # Default magic value for XOR table construction
    DEFAULT_MAGIC = "test_magic_12345678"

    def __init__(
        self,
        cdp_session: FakeCDPSessionProbe,
        *,
        before_playing: bool = False,
        login_redirects_to_play: bool = False,
        emit_during_stabilization: bool = False,
    ) -> None:
        """Initialize fake page for probing.

        Args:
            cdp_session: CDP session to use.
            before_playing: Whether to simulate being on before-playing page.
            login_redirects_to_play: If True, simulates login redirecting to /play.
            emit_during_stabilization: If True, emit messages during stabilization loop.
        """
        self._cdp_session = cdp_session
        self._closed = False
        self._url = ""
        self._before_playing = before_playing
        self._login_redirects_to_play = login_redirects_to_play
        self._first_wait = True
        self._wait_count = 0
        self._emit_during_stabilization = emit_during_stabilization

    @property
    def url(self) -> str:
        """Get the current URL of the page."""
        return self._url

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Get the keyboard interface.

        Returns:
            FakeKeyboard instance (no-op for tests using CDP synthetic events).
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
        if self._before_playing:
            self._url = url.replace("/play", "/before-playing")
        else:
            self._url = url
        return FakeResponse(url=self._url)

    def wait_for_timeout(self, timeout: float) -> None:
        """Wait and emit initial WebSocket events on first call."""
        _ = timeout
        self._wait_count += 1

        # On second wait (after login click), simulate URL redirect if configured
        if self._wait_count == 2 and self._login_redirects_to_play:
            self._url = "https://tankpit.com/play"

        if self._first_wait:
            self._first_wait = False
            # Emit initial connection and auth messages
            self._cdp_session.emit_event(
                "Network.webSocketCreated",
                {"requestId": "1.1", "url": "wss://tankpit.com/ws/"},
            )
            # Emit AUTH message with magic for XOR table construction
            auth_payload = _make_auth_payload(self.DEFAULT_MAGIC)
            self._cdp_session.emit_event(
                "Network.webSocketFrameSent",
                {
                    "requestId": "1.1",
                    "timestamp": 1.0,
                    "response": {"opcode": 1, "mask": True, "payloadData": auth_payload},
                },
            )
            self._cdp_session.emit_event(
                "Network.webSocketFrameReceived",
                {
                    "requestId": "1.1",
                    "timestamp": 2.0,
                    "response": {"opcode": 1, "mask": False, "payloadData": "room_list"},
                },
            )
            self._cdp_session._raw_messages_ready = True
        elif self._emit_during_stabilization and self._wait_count == 5:
            # Emit extra message during stabilization loop (iteration 2)
            # Calls: 1=join_room, 2=join_room, 3=pre-stabilization, 4=loop iter 1, 5=loop iter 2
            self._cdp_session.emit_event(
                "Network.webSocketFrameReceived",
                {
                    "requestId": "1.1",
                    "timestamp": 3.0,
                    "response": {"opcode": 1, "mask": False, "payloadData": "extra_msg"},
                },
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
        """Evaluate JavaScript expression.

        Returns:
            Empty list for all expressions (magic comes from AUTH messages).
        """
        _ = expression
        return []


class FakePageProbeNoMessages:
    """Fake Page for probe testing that doesn't emit any messages."""

    def __init__(self, cdp_session: FakeCDPSessionProbe) -> None:
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
            FakeKeyboard instance (no-op for tests using CDP synthetic events).
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
        return FakeResponse(url=url)

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
        """Evaluate JavaScript expression.

        Returns:
            Empty list for all expressions (magic comes from AUTH messages).
        """
        _ = expression
        return []
