"""Playwright Page/CDP/Keyboard/Response protocols.

The protocols here match the Playwright ``sync_api`` interface surface
the bot consumes. Tests substitute hand-rolled fakes that satisfy these
protocols; production code receives the real Playwright objects.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from platform_core.json_utils import JSONObject, JSONValue


class ResponseProtocol(Protocol):
    """Protocol for Playwright Response object."""

    @property
    def status(self) -> int:
        """HTTP status code.

        Returns:
            Status code (e.g., 200, 404).
        """
        ...

    @property
    def url(self) -> str:
        """Response URL.

        Returns:
            The URL of the response.
        """
        ...


class CDPSessionProtocol(Protocol):
    """Protocol for Playwright CDPSession.

    Matches playwright.sync_api.CDPSession interface.
    """

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Send a CDP command and return the result.

        Args:
            method: CDP method name (e.g., "Network.enable").
            params: Optional parameters for the method.

        Returns:
            Response from CDP as a JSON object.
        """
        ...

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register an event handler for CDP events.

        Args:
            event: CDP event name (e.g., "Network.webSocketFrameReceived").
            handler: Callback that receives event params as JSONObject.
        """
        ...

    def detach(self) -> None:
        """Detach the CDP session from the target."""
        ...


class KeyboardProtocol(Protocol):
    """Protocol for Playwright Keyboard.

    Matches playwright.sync_api.Keyboard interface for methods we use.
    """

    def press(self, key: str, *, delay: float | None = None) -> None:
        """Press a keyboard key.

        Args:
            key: Key name (e.g., "f", "Enter", "ArrowUp").
            delay: Time to wait between keydown and keyup in milliseconds.
        """
        ...

    def type(self, text: str, *, delay: float | None = None) -> None:
        """Type text character by character.

        Args:
            text: Text to type.
            delay: Time to wait between key presses in milliseconds.
        """
        ...


class PageProtocol(Protocol):
    """Protocol for Playwright Page.

    Matches playwright.sync_api.Page interface for methods we use.
    """

    @property
    def url(self) -> str:
        """Get the current URL of the page."""
        ...

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Get the keyboard interface for this page.

        Returns:
            Keyboard interface for sending key events.
        """
        ...

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> ResponseProtocol | None:
        """Navigate to a URL.

        Args:
            url: URL to navigate to.
            referer: Referer header value.
            timeout: Maximum operation time in milliseconds.
            wait_until: When to consider operation succeeded ("load", "domcontentloaded",
                "networkidle", "commit").

        Returns:
            Response object or None if navigation failed.
        """
        ...

    def wait_for_timeout(self, timeout: float) -> None:
        """Wait for specified timeout in milliseconds.

        Args:
            timeout: Timeout in milliseconds.
        """
        ...

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Wait for an event to be fired.

        Args:
            event: Event name to wait for (e.g., "close").
            timeout: Maximum wait time in milliseconds.
        """
        ...

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Wait for a JavaScript function to return truthy value.

        Args:
            expression: JavaScript expression to evaluate.
            timeout: Maximum wait time in milliseconds.
        """
        ...

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close the page.

        Args:
            reason: Reason to be reported to operations interrupted by page closure.
            run_before_unload: Whether to run the before unload page handlers.
        """
        ...

    def evaluate(self, expression: str) -> JSONValue:
        """Evaluate JavaScript expression in the page context.

        Args:
            expression: JavaScript expression to evaluate.

        Returns:
            Result of the expression evaluation.
        """
        ...


__all__ = [
    "CDPSessionProtocol",
    "KeyboardProtocol",
    "PageProtocol",
    "ResponseProtocol",
]
