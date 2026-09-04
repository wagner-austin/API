"""Reading a long-lived HTTP response as it arrives.

A live-view MJPEG response never ends, so a caller cannot wait for a
body: it reads what has arrived and keeps reading. That is the whole
surface, and it sits behind a hook so the stream probe's parsing,
timing and reporting can be exercised against bytes a test wrote rather
than against a socket and a running bot.

Stdlib :mod:`http.client` rather than ``urllib.request.urlopen``, for
the reason :mod:`tankpit_bot.service.probe` already records: ``urlopen``
returns a context manager whose type collapses to ``Any`` under strict
mypy, because it is polymorphic across every URL scheme it supports.
:class:`http.client.HTTPConnection` exposes a typed
:class:`~http.client.HTTPResponse`, so ``read`` and ``getheader``
resolve concretely and no cast is needed.
"""

from __future__ import annotations

from http.client import HTTPConnection, HTTPResponse
from types import TracebackType
from typing import Protocol
from urllib.parse import urlparse


class HttpStreamProtocol(Protocol):
    """One open HTTP response being read incrementally."""

    @property
    def content_type(self) -> str:
        """The response's ``Content-Type``.

        Returns:
            The header value, which for a multipart stream carries the
            boundary the sender chose.
        """
        ...

    def read(self, size: int) -> bytes:
        """Read up to ``size`` bytes.

        Args:
            size: Maximum bytes to return.

        Returns:
            The bytes read; empty when the response has ended.
        """
        ...

    def close(self) -> None:
        """Release the connection."""
        ...


class OpenHttpStreamProtocol(Protocol):
    """Opens an HTTP stream by URL."""

    def __call__(self, url: str) -> HttpStreamProtocol:
        """Open one stream.

        Args:
            url: Absolute URL to GET.

        Returns:
            The open stream.

        Raises:
            OSError: If the host cannot be reached.
        """
        ...


class _HttpClientStream:
    """An :class:`HttpStreamProtocol` over a typed :mod:`http.client` response."""

    def __init__(self, url: str) -> None:
        """Open the response.

        Args:
            url: Absolute URL to GET.

        Raises:
            ValueError: If the URL carries no host.
            OSError: If the host cannot be reached.
        """
        parsed = urlparse(url)
        host = parsed.hostname
        if host is None:
            raise ValueError(f"stream URL missing host: {url!r}")
        port = parsed.port if parsed.port is not None else 80
        path = parsed.path if parsed.path != "" else "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        self._connection = HTTPConnection(host, port, timeout=30.0)
        self._connection.request("GET", path)
        self._response: HTTPResponse = self._connection.getresponse()

    @property
    def content_type(self) -> str:
        """The response's ``Content-Type``.

        Returns:
            The header value, or the empty string when absent.
        """
        header = self._response.getheader("Content-Type")
        return header if header is not None else ""

    def read(self, size: int) -> bytes:
        """Read up to ``size`` bytes.

        Args:
            size: Maximum bytes to return.

        Returns:
            The bytes read; empty at end of response.
        """
        return self._response.read(size)

    def close(self) -> None:
        """Release the connection."""
        self._connection.close()

    def __enter__(self) -> _HttpClientStream:
        """Enter the context.

        Returns:
            This stream.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close on the way out.

        Args:
            exc_type: Exception class, if one is propagating.
            exc: Exception instance, if one is propagating.
            tb: Traceback, if one is propagating.

        Returns:
            None; never suppresses.
        """
        _ = (exc_type, exc, tb)
        self.close()


def _real_open_http_stream(url: str) -> HttpStreamProtocol:
    """Open a real HTTP stream.

    Args:
        url: Absolute URL to GET.

    Returns:
        The open stream.

    Raises:
        ValueError: If the URL carries no host.
        OSError: If the host cannot be reached.
    """
    return _HttpClientStream(url)


#: Hookable stream opener. The stream probe reads through this name so
#: its parsing and timing run against test-authored bytes rather than a
#: socket.
open_http_stream: OpenHttpStreamProtocol = _real_open_http_stream


__all__ = [
    "HttpStreamProtocol",
    "OpenHttpStreamProtocol",
    "_HttpClientStream",
    "_real_open_http_stream",
    "open_http_stream",
]
