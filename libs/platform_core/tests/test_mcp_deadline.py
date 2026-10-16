"""An MCP exchange ends at its deadline, whatever the peer does (MCPs board task e14815d3).

Every case runs the real poster against a real loopback socket, because the
defect lived in what the standard library does with a socket: ``urllib``'s
timeout bounds each operation, so on 2026-10-16 a peer sending one byte a
second held a 2-second call open 12 seconds. A fake poster would answer
whatever it was told and prove nothing about that.

Each peer is a one-shot server on its own thread that reads the request and
then behaves as the case names: trickles the body, trickles the headers,
stays silent, or breaks the response off early.
"""

from __future__ import annotations

import http.client
import socket
import threading
import time
import urllib.error
from collections.abc import Callable, Generator
from typing import Final

import pytest

from platform_core.error_codes_tooling import McpClientErrorCode
from platform_core.errors import AppError
from platform_core.mcp_client import ExchangeDeadline, urllib_mcp_post

#: The deadline every case posts with.
DEADLINE_SECONDS: Final = 1

#: How long a misbehaving peer keeps going, well past the deadline.
PEER_SECONDS: Final = 6

#: Slack for thread scheduling between the deadline and the raise.
SLACK_SECONDS: Final = 1.0

#: What a peer's send meets once the client shut the connection: Windows
#: reports an abort or a reset, Linux a broken pipe or a reset.
HUNG_UP: Final = frozenset({"ConnectionAbortedError", "ConnectionResetError", "BrokenPipeError"})


def cut_short(url: str) -> str:
    """The message an exchange the deadline ended raises with.

    Args:
        url: The peer.

    Returns:
        The message.
    """
    return f"{url} did not finish the exchange within its {DEADLINE_SECONDS} s deadline"


#: A 200's head promising a 64-byte event-stream body.
SIZED_HEAD: Final = (
    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 64\r\n\r\n"
)


def trickle_body(conn: socket.socket) -> None:
    """Answer the headers, then one body byte a second.

    Args:
        conn: The accepted connection.
    """
    conn.sendall(SIZED_HEAD)
    for _ in range(PEER_SECONDS * 2):
        conn.sendall(b"x")
        time.sleep(0.5)


def trickle_headers(conn: socket.socket) -> None:
    """Answer the status line, then one header line every half second.

    Args:
        conn: The accepted connection.
    """
    conn.sendall(b"HTTP/1.1 200 OK\r\n")
    for index in range(PEER_SECONDS * 2):
        conn.sendall(f"X-Line-{index}: y\r\n".encode())
        time.sleep(0.5)


def trickle_unsized_body(conn: socket.socket) -> None:
    """Answer headers with no length, so the body runs to the close, then trickle it.

    A shutdown at the deadline ends such a body as an ordinary end of stream,
    so the read returns rather than raising.

    Args:
        conn: The accepted connection.
    """
    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
    for _ in range(PEER_SECONDS * 2):
        conn.sendall(b"x")
        time.sleep(0.5)


def silent(conn: socket.socket) -> None:
    """Answer nothing at all, then close.

    Args:
        conn: The accepted connection, held open until the peer gives up.
    """
    time.sleep(PEER_SECONDS)
    conn.close()


def broken_off(conn: socket.socket) -> None:
    """Promise a body, send a part of it, and end the stream cleanly at once.

    A half-close rather than a close, then waiting for the client to hang up,
    so the client sees an end of stream on every platform; closing with the
    request still buffered resets the connection on Windows instead.

    Args:
        conn: The accepted connection.
    """
    conn.sendall(SIZED_HEAD + b"short")
    conn.shutdown(socket.SHUT_WR)
    conn.recv(1)


class Peer:
    """One one-shot loopback server and how its connection ended.

    Attributes:
        url: The URL to post to.
        cut_off: The error the peer's next send met once the client hung up,
            or None while it has met none: the evidence that the client, not
            the peer, ended the exchange.
    """

    def __init__(self, behave: Callable[[socket.socket], None]) -> None:
        """Listen on a loopback port and serve one connection as ``behave`` says.

        Args:
            behave: What the peer does after reading the request.
        """
        self._listener = socket.create_server(("127.0.0.1", 0))
        port: int = self._listener.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/mcp"
        self.cut_off: OSError | None = None
        self._behave = behave
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        """Accept, read the request, behave, and record a send the client refused."""
        conn, _ = self._listener.accept()
        with conn:
            conn.recv(65536)
            try:
                self._behave(conn)
            except OSError as gone:
                # The peer's only way to learn the client hung up is a send
                # that fails; that failure is what the cases assert on.
                self.cut_off = gone

    def stop(self) -> None:
        """Wait for the peer to finish and close its listener."""
        self._thread.join(timeout=PEER_SECONDS + 2)
        self._listener.close()


@pytest.fixture(name="peer")
def _peer() -> Generator[Callable[[Callable[[socket.socket], None]], Peer], None, None]:
    """Start one-shot peers, stopping every one at the end.

    Yields:
        A function that starts a peer behaving as given.
    """
    peers: list[Peer] = []

    def start(behave: Callable[[socket.socket], None]) -> Peer:
        started = Peer(behave)
        peers.append(started)
        return started

    yield start
    for started in peers:
        started.stop()


@pytest.mark.parametrize("behave", [trickle_body, trickle_unsized_body, trickle_headers])
def test_a_peer_that_keeps_sending_is_cut_off_at_the_deadline(
    peer: Callable[[Callable[[socket.socket], None]], Peer],
    behave: Callable[[socket.socket], None],
) -> None:
    """A sized body breaks off when cut, an unsized one reads as ended, and
    headers break off mid-line; every one raises the deadline's error, and the
    peer's next send meets a connection the client closed."""
    started_peer = peer(behave)
    started = time.monotonic()

    with pytest.raises(TimeoutError) as raised:
        urllib_mcp_post(started_peer.url, headers={}, body=b"{}", timeout_seconds=DEADLINE_SECONDS)

    assert time.monotonic() - started < DEADLINE_SECONDS + SLACK_SECONDS
    assert str(raised.value) == cut_short(started_peer.url)
    started_peer.stop()
    assert type(started_peer.cut_off).__name__ in HUNG_UP, started_peer.cut_off


def test_a_silent_peer_ends_at_the_deadline(
    peer: Callable[[Callable[[socket.socket], None]], Peer],
) -> None:
    """Either bound may land first: the read's own timeout or the watchdog."""
    started_peer = peer(silent)
    started = time.monotonic()

    with pytest.raises(TimeoutError) as raised:
        urllib_mcp_post(started_peer.url, headers={}, body=b"{}", timeout_seconds=DEADLINE_SECONDS)

    assert time.monotonic() - started < DEADLINE_SECONDS + SLACK_SECONDS
    assert str(raised.value) in (
        cut_short(started_peer.url),
        "timed out",
    )


def test_a_response_broken_off_before_the_deadline_raises_as_itself(
    peer: Callable[[Callable[[socket.socket], None]], Peer],
) -> None:
    started_peer = peer(broken_off)

    with pytest.raises(http.client.IncompleteRead) as raised:
        urllib_mcp_post(started_peer.url, headers={}, body=b"{}", timeout_seconds=5)

    assert raised.value.partial == b"short"


def test_a_refused_connection_raises_as_itself_with_no_watchdog_armed() -> None:
    """The connect fails before the deadline is armed, so nothing is left to stop."""
    listener = socket.create_server(("127.0.0.1", 0))
    port: int = listener.getsockname()[1]
    listener.close()

    with pytest.raises(urllib.error.URLError) as raised:
        urllib_mcp_post(f"http://127.0.0.1:{port}/mcp", headers={}, body=b"{}", timeout_seconds=5)

    assert type(raised.value.reason).__name__ == "ConnectionRefusedError"


def test_a_url_that_is_not_http_is_refused_before_any_connection() -> None:
    with pytest.raises(AppError) as raised:
        urllib_mcp_post("https://127.0.0.1:1/mcp", headers={}, body=b"{}", timeout_seconds=5)

    assert raised.value.code is McpClientErrorCode.URL_NOT_HTTP
    assert raised.value.message == (
        "https://127.0.0.1:1/mcp is 'https'; an MCP exchange's deadline bounds only http"
    )


def test_a_finished_exchange_keeps_its_socket_when_the_watchdog_fires_late() -> None:
    """The race the lock closes: the deadline fires after the exchange ended."""
    left, right = socket.socketpair()
    with left, right:
        deadline = ExchangeDeadline(60)
        deadline.arm(left)

        assert deadline.finish() is False
        deadline.expire(left)

        left.sendall(b"still open")
        assert right.recv(64) == b"still open"


def test_an_exchange_that_outlived_its_deadline_is_reported_as_passed() -> None:
    left, right = socket.socketpair()
    with left, right:
        deadline = ExchangeDeadline(0)
        deadline.arm(left)
        time.sleep(0.2)

        assert deadline.finish() is True
        assert right.recv(64) == b""
