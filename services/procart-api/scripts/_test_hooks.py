"""Test hooks for procart-api scripts.

Production code uses the real implementations; tests replace these
module-level symbols to inject fakes without conditionals in the scripts.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from types import TracebackType
from typing import Protocol

import httpx
import pygame
from procart.types import SceneConfig
from typing_extensions import TypedDict


class RenderRequest(TypedDict):
    """Body posted to the render endpoints.

    Attributes:
        scene: Scene to render.
        output_dir: Absolute directory the service writes into.
    """

    scene: SceneConfig
    output_dir: str


class ResponseProtocol(Protocol):
    """The response surface the demo script consumes."""

    @property
    def content(self) -> bytes:
        """Raw response body."""
        ...

    def raise_for_status(self) -> ResponseProtocol:
        """Raise if the response carries an error status.

        Returns:
            The response itself; the demo ignores it.
        """
        ...


class HttpClientProtocol(Protocol):
    """The HTTP client surface the demo script consumes."""

    def get(self, url: str) -> ResponseProtocol:
        """Issue a GET.

        Args:
            url: Path relative to the client's base URL.

        Returns:
            The response.
        """
        ...

    def post(self, url: str, *, json: RenderRequest) -> ResponseProtocol:
        """Issue a POST with a JSON body.

        Args:
            url: Path relative to the client's base URL.
            json: Request body.

        Returns:
            The response.
        """
        ...

    def __enter__(self) -> HttpClientProtocol:
        """Enter the client context.

        Returns:
            The client itself.
        """
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the client on context exit.

        Args:
            exc_type: Exception type, if one is propagating.
            exc_value: Exception instance, if one is propagating.
            traceback: Traceback, if one is propagating.
        """
        ...


class HttpClientFactoryProtocol(Protocol):
    """Protocol for building the HTTP client."""

    def __call__(self, *, base_url: str, timeout_seconds: float) -> HttpClientProtocol:
        """Build a client.

        Args:
            base_url: Base URL of a running procart-api service.
            timeout_seconds: Per-request timeout.

        Returns:
            A client ready for use as a context manager.
        """
        ...


def _real_http_client(*, base_url: str, timeout_seconds: float) -> HttpClientProtocol:
    """Real implementation returning an httpx client.

    Args:
        base_url: Base URL of a running procart-api service.
        timeout_seconds: Per-request timeout.

    Returns:
        An httpx.Client bound to the base URL.
    """
    return httpx.Client(base_url=base_url, timeout=httpx.Timeout(timeout_seconds))


# The demo talks to a service over the network, which is the one thing a test
# cannot do. Construction is the seam so the request/response flow itself stays
# under test.
http_client_factory: HttpClientFactoryProtocol = _real_http_client


class DisplayProtocol(Protocol):
    """The windowing surface the preview drives."""

    def create(self, size: tuple[int, int], caption: str) -> pygame.Surface:
        """Open the preview window.

        Args:
            size: Window size in pixels.
            caption: Window title.

        Returns:
            The surface frames are drawn onto.
        """
        ...

    def present(self) -> None:
        """Publish the drawn frame."""
        ...

    def shutdown(self) -> None:
        """Tear the window down."""
        ...


class _RealDisplay:
    """Production display backed by a real pygame window."""

    def create(self, size: tuple[int, int], caption: str) -> pygame.Surface:
        """Open the preview window.

        Args:
            size: Window size in pixels.
            caption: Window title.

        Returns:
            The surface frames are drawn onto.
        """
        pygame.init()
        screen = pygame.display.set_mode(size)
        pygame.display.set_caption(caption)
        return screen

    def present(self) -> None:
        """Publish the drawn frame."""
        pygame.display.flip()

    def shutdown(self) -> None:
        """Tear the window down."""
        pygame.quit()


# Opening a window is the only part of the preview a test cannot do. Behind
# this seam a test supplies a plain pygame.Surface, so the blits, font
# rendering and surfarray conversion all still run for real.
display: DisplayProtocol = _RealDisplay()


class EventSourceProtocol(Protocol):
    """Protocol for draining the pending input events."""

    def __call__(self) -> list[pygame.event.Event]:
        """Drain the event queue.

        Returns:
            The events pending since the last call.
        """
        ...


def _real_event_source() -> list[pygame.event.Event]:
    """Real implementation reading pygame's event queue.

    Returns:
        The events pending since the last call.
    """
    return pygame.event.get()


# The preview loop runs until the user quits, so the event stream is what
# decides when it stops. Injecting it lets a test drive a fixed number of
# frames instead of waiting for a window nobody is looking at.
event_source: EventSourceProtocol = _real_event_source


def reset_hooks() -> None:
    """Reset all hooks to their default implementations."""
    global http_client_factory, event_source, display
    http_client_factory = _real_http_client
    event_source = _real_event_source
    display = _RealDisplay()


__all__ = [
    "DisplayProtocol",
    "EventSourceProtocol",
    "HttpClientFactoryProtocol",
    "HttpClientProtocol",
    "RenderRequest",
    "ResponseProtocol",
    "_RealDisplay",
    "_real_event_source",
    "_real_http_client",
    "display",
    "event_source",
    "http_client_factory",
    "reset_hooks",
]
