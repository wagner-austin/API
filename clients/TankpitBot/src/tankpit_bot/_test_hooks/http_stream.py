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

from http.client import HTTPConnection, HTTPResponse, HTTPSConnection
from types import TracebackType
from typing import Protocol
from urllib.parse import urlparse


class HttpStreamProtocol(Protocol):
    """One open HTTP response being read incrementally."""

    @property
    def status(self) -> int:
        """The response status.

        Exposed because a caller that only sees the body cannot tell a
        stream from a refusal: a 404 arrives as a perfectly valid
        text/plain response, and a reader looking for a multipart
        boundary reports "this stream has no boundary" when the truth
        is "that slot is not running".

        Returns:
            The HTTP status code.
        """
        ...

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


def resolve_target(url: str) -> tuple[bool, str, int, str]:
    """Split a stream URL into the pieces a connection needs.

    Pure, and separate from opening the socket, so the scheme and port
    decisions are testable without a network. That split is not
    cosmetic: the first version of this module folded them into the
    constructor and defaulted everything to plain HTTP on port 80, so an
    ``https://`` URL reached the public endpoint as cleartext, came back
    as a text/plain error page, and the probe blamed the STREAM for
    having no multipart boundary. The fault was the client, and nothing
    could see it because nothing could call it.

    Args:
        url: Absolute URL to GET.

    Returns:
        Whether TLS is required, the host, the port, and the
        request path with any query string reattached.

    Raises:
        ValueError: If the URL carries no host, or names a scheme other
            than http/https. Both are refused rather than defaulted.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    if host is None:
        raise ValueError(f"stream URL missing host: {url!r}")
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"stream URL scheme must be http or https: {url!r}")
    secure = parsed.scheme == "https"
    port = parsed.port if parsed.port is not None else (443 if secure else 80)
    path = parsed.path if parsed.path != "" else "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return secure, host, port, path


class _HttpClientStream:
    """An :class:`HttpStreamProtocol` over a typed :mod:`http.client` response.

    Construction takes an ALREADY-OPEN connection and response rather
    than a URL, so this object never exists half-built. It used to do
    the dialling itself, which meant a refused connection or a broken
    status line raised out of ``__init__`` with the socket already
    created and nothing left holding a reference to close it — the only
    thing that ever shut it was the garbage collector, on its own
    schedule and with a ``ResourceWarning``. Opening is
    :func:`_real_open_http_stream`'s job now, and it closes what it
    opened when the open does not complete.
    """

    def __init__(self, connection: HTTPConnection, response: HTTPResponse) -> None:
        """Bind to the connection this stream owns.

        Args:
            connection: The open connection; closed by :meth:`close`.
            response: The response already read off it.
        """
        self._connection = connection
        self._response = response

    @property
    def status(self) -> int:
        """The response status.

        Returns:
            The HTTP status code.
        """
        return self._response.status

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
    """Open a real HTTP stream, owning the socket until the stream takes it.

    The handover is the last thing that happens. Anything that fails
    before it — a refused connection, a timeout, a peer that hangs up
    mid-status-line — leaves this function still holding the socket, so
    it closes it on the way out rather than leaving it to the
    collector. Same ownership shape as
    :func:`tankpit_bot.service._test_hooks.video._real_open_child_video`,
    and for the same reason: nothing here HANDLES a failure, it just
    declines to leak on one.

    Args:
        url: Absolute URL to GET.

    Returns:
        The open stream.

    Raises:
        ValueError: If the URL carries no host or names a scheme other
            than http/https.
        OSError: If the host cannot be reached.
    """
    secure, host, port, path = resolve_target(url)
    connection: HTTPConnection = (
        HTTPSConnection(host, port, timeout=30.0)
        if secure
        else HTTPConnection(host, port, timeout=30.0)
    )
    handed_over = False
    try:
        connection.request("GET", path)
        response: HTTPResponse = connection.getresponse()
        stream = _HttpClientStream(connection, response)
        handed_over = True
        return stream
    finally:
        if not handed_over:
            connection.close()


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
    "resolve_target",
]
