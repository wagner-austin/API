"""One MCP exchange over plain http: its deadline and the opener that carries it.

Split out of :mod:`platform_core.mcp_client` (MCPs board task a1f9d16d), which
keeps the poster, :func:`platform_core.mcp_client.urllib_mcp_post`, and the
JSON-RPC envelope around it; this module holds what one exchange is opened
with.

THE OPENER IS BUILT FOR HTTP ALONE. The poster built its opener with
:func:`urllib.request.build_opener`, which adds an :class:`urllib.request.HTTPSHandler`
whenever :mod:`ssl` imports, and since Python 3.12 that handler builds a
default SSL context, the CA bundle loaded from disk, as it is constructed.
The poster refuses every scheme but http, so that context was never used,
and it was paid on every call: measured on diphtheria on 2026-10-25 in the
fleet venv (Python 3.12.3), one ``build_opener()`` took 40.7 ms of cpu and
an opener of the handlers below 0.27 ms, and py-spy over the hub's ten node
runners for 120 s put 30.6 percent of their 71.3 cpu-seconds in
``ssl.create_default_context``, each runner's queue poll every 5 s among
them. :func:`exchange_opener` is that opener: the proxy handler
``build_opener`` would add, the deadline's http handler, the pass-through
error processor, and the handler that refuses an unknown scheme by name.
Nothing else ``build_opener`` adds can be reached: the error processor hands
every status back, so no redirect or error handler is ever called, and no
file, ftp, data or https URL gets past the poster's scheme check.
"""

from __future__ import annotations

import http.client
import socket
import threading
import time
import urllib.request
from typing import Final

#: How long a single call may take before it is abandoned.
DEFAULT_MCP_TIMEOUT_SECONDS: Final = 20

#: http.client's default read size for a connection's buffered reader.
_BLOCKSIZE: Final = 8192


class _PassThroughErrorProcessor(urllib.request.HTTPErrorProcessor):
    """An error processor that hands back non-2xx responses instead of raising.

    ``urllib``'s default turns a 401 into a raised
    :class:`urllib.error.HTTPError`, which would force
    :func:`platform_core.mcp_client.call_mcp_tool` to catch an exception in
    order to read a status code it needs. The status is not an exceptional
    condition here -- a rotated API key is the ordinary way this fails, and
    the caller's job is to attach
    :attr:`McpClientErrorCode.HTTP_STATUS` to it.

    Replacing the processor removes the exception at its source rather than
    handling it downstream, so no ``except`` appears anywhere in this path.
    """

    def http_response(
        self, request: urllib.request.Request, response: http.client.HTTPResponse
    ) -> http.client.HTTPResponse:
        """Return the response unchanged, whatever its status.

        Args:
            request: The request that produced it, unused.
            response: The response as received.

        Returns:
            That same response.
        """
        return response

    https_response = http_response


class ExchangeDeadline:
    """One exchange's deadline, kept by shutting its socket when it passes.

    ``urllib``'s ``timeout`` bounds each socket operation, not the exchange:
    measured 2026-10-16 (MCPs board task e14815d3), a peer that sent one byte
    a second held a 2-second call open 12 seconds, and a hub tick held one
    connection to fleet-mcp 290 seconds across a deploy that recreated it.
    Shutting the socket at the deadline ends whatever read or write is blocked
    on it, wherever in the exchange that is.

    Attributes:
        at: The deadline on the :func:`time.monotonic` clock.
    """

    def __init__(self, seconds: float) -> None:
        """Start the clock.

        Args:
            seconds: How long the whole exchange may take from now.
        """
        self.at = time.monotonic() + seconds
        self._lock = threading.Lock()
        self._finished = False
        self._passed = False
        self._timer: threading.Timer | None = None

    def arm(self, sock: socket.socket) -> None:
        """Shut ``sock`` down when the deadline passes, unless finished first.

        Args:
            sock: The exchange's connected socket.
        """
        self._timer = threading.Timer(max(0.0, self.at - time.monotonic()), self.expire, (sock,))
        self._timer.daemon = True
        self._timer.start()

    def expire(self, sock: socket.socket) -> None:
        """Shut the socket down, unless the exchange already finished.

        Under the lock, so the exchange never closes the socket while this
        shuts it down, and a finished exchange's socket is never touched.

        Args:
            sock: The exchange's connected socket.
        """
        with self._lock:
            if self._finished:
                return
            self._passed = True
            sock.shutdown(socket.SHUT_RDWR)

    def finish(self) -> bool:
        """Stop the watchdog.

        Returns:
            Whether the deadline passed first, so what the exchange read or
            raised was cut short by it.
        """
        with self._lock:
            self._finished = True
            passed = self._passed
        if self._timer is not None:
            self._timer.cancel()
        return passed


class _DeadlineConnection(http.client.HTTPConnection):
    """An HTTP connection that arms its exchange's deadline once connected."""

    def __init__(
        self,
        host: str,
        *,
        port: int | None,
        timeout: float,
        source_address: tuple[str, int] | None,
        blocksize: int,
        deadline: ExchangeDeadline,
    ) -> None:
        """Build the connection.

        Args:
            host: The host, with or without its port.
            port: The port, when not in ``host``.
            timeout: Each socket operation's bound, the connect among them.
            source_address: The local address to bind, if any.
            blocksize: The buffered reader's read size.
            deadline: The exchange's deadline, armed on connect.
        """
        super().__init__(
            host, port=port, timeout=timeout, source_address=source_address, blocksize=blocksize
        )
        self._deadline = deadline

    def connect(self) -> None:
        """Connect, then arm the deadline on the connected socket."""
        super().connect()
        self._deadline.arm(self.sock)


class _DeadlineHandler(urllib.request.HTTPHandler):
    """Open http URLs on connections bounded by one exchange's deadline."""

    def __init__(self, deadline: ExchangeDeadline) -> None:
        """Build the handler.

        Args:
            deadline: The exchange's deadline.
        """
        super().__init__()
        self._deadline = deadline

    def http_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        """Open the request on a deadline-bounded connection.

        Args:
            req: The request.

        Returns:
            The response, whatever its status.
        """
        return self.do_open(self._connection, req)

    def _connection(
        self,
        host: str,
        /,
        *,
        port: int | None = None,
        timeout: float = DEFAULT_MCP_TIMEOUT_SECONDS,
        source_address: tuple[str, int] | None = None,
        blocksize: int = _BLOCKSIZE,
    ) -> http.client.HTTPConnection:
        """Build the connection ``do_open`` asks for, carrying the deadline.

        Args:
            host: The host, with or without its port.
            port: The port, when not in ``host``.
            timeout: Each socket operation's bound.
            source_address: The local address to bind, if any.
            blocksize: The buffered reader's read size.

        Returns:
            The connection.
        """
        return _DeadlineConnection(
            host,
            port=port,
            timeout=timeout,
            source_address=source_address,
            blocksize=blocksize,
            deadline=self._deadline,
        )


def exchange_opener(deadline: ExchangeDeadline) -> urllib.request.OpenerDirector:
    """The opener one http exchange is made with.

    Args:
        deadline: The exchange's deadline, armed on its connection.

    Returns:
        An opener of four handlers: the environment's proxies, the
        deadline's http handler, the pass-through error processor, and the
        unknown-scheme handler, which refuses any other scheme with
        :class:`urllib.error.URLError`. No SSL context is built: the module
        docstring carries the measurement.
    """
    opener = urllib.request.OpenerDirector()
    for handler in (
        urllib.request.ProxyHandler(),
        _DeadlineHandler(deadline),
        _PassThroughErrorProcessor(),
        urllib.request.UnknownHandler(),
    ):
        opener.add_handler(handler)
    return opener


def raise_if_deadline_passed(
    deadline: ExchangeDeadline, url: str, seconds: int, cause: BaseException | None
) -> None:
    """Stop the watchdog, and raise when the exchange ran past its deadline.

    Called whether the exchange raised or returned: a shutdown ends a blocked
    read with an error on Windows but as an ordinary end of stream on Linux,
    where a body read to its close returns normally, cut short. Either way an
    exchange the deadline ended never returns an answer.

    A :class:`TimeoutError` because callers already read every ``OSError``
    out of the poster as an unanswered call (fleet's ``answering`` turns one
    into ``QUEUE_UNANSWERED``), and an exchange that ran out its deadline is
    exactly that.

    Args:
        deadline: The exchange's deadline.
        url: The endpoint.
        seconds: The deadline's length.
        cause: What the exchange raised, or None when it returned.

    Raises:
        TimeoutError: When the deadline passed first, chained to ``cause``.
    """
    if deadline.finish():
        raise TimeoutError(
            f"{url} did not finish the exchange within its {seconds} s deadline"
        ) from cause


__all__ = [
    "DEFAULT_MCP_TIMEOUT_SECONDS",
    "ExchangeDeadline",
    "exchange_opener",
    "raise_if_deadline_passed",
]
