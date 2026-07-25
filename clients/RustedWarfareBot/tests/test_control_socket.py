"""The real socket implementation, over a real loopback connection.

The channel tests drive a scripted peer, which is right for asserting framing
and order shape but never executes the socket code itself. These run the
production ``connect`` against a listener in this process: no fake, no mock,
just both ends of a real TCP connection.
"""

from __future__ import annotations

import socket
import threading
from types import TracebackType

from rw_bot.control import _test_hooks


class _Listener:
    """A loopback server that echoes and records one connection.

    Attributes:
        port: The ephemeral port the OS assigned.
        received: Every line the client sent, in order.
        to_send: Lines served to the client on connect.
    """

    def __init__(self, to_send: list[str]) -> None:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        # getsockname() is Any in typeshed because the address shape varies by
        # family. Annotating at the assignment is the workspace's pattern for
        # pinning a value the stubs cannot narrow.
        bound: tuple[str, int] = self._server.getsockname()
        self.port = bound[1]
        self.received: list[str] = []
        self.to_send = to_send
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        """Accept one client, send the prepared lines, then read until closed."""
        accepted: tuple[socket.socket, tuple[str, int]] = self._server.accept()
        with accepted[0] as connection:
            for line in self.to_send:
                connection.sendall(f"{line}\n".encode())
            reader = connection.makefile("r", encoding="utf-8", newline="\n")
            while True:
                line = reader.readline()
                if line == "":
                    return
                self.received.append(line.rstrip("\n"))

    def __enter__(self) -> _Listener:
        """Start serving.

        Returns:
            This listener.
        """
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop serving and release the listening socket.

        Args:
            exc_type: Exception class raised in the block, if any.
            exc: Exception raised in the block, if any.
            traceback: Traceback of the raised exception, if any.
        """
        self._thread.join(timeout=5.0)
        self._server.close()


def test_a_real_connection_reads_the_lines_the_peer_sent() -> None:
    with _Listener(["first", "second"]) as listener:
        connection = _test_hooks.connect("127.0.0.1", listener.port, 5.0)
        assert connection.read_line() == "first"
        assert connection.read_line() == "second"
        connection.close()


def test_a_real_connection_delivers_the_lines_it_sends() -> None:
    with _Listener([]) as listener:
        connection = _test_hooks.connect("127.0.0.1", listener.port, 5.0)
        connection.send_line('{"kind":"move","unit_id":1,"x":0.0,"y":0.0}')
        connection.send_line('{"kind":"move","unit_id":2,"x":1.0,"y":2.0}')
        connection.close()
    assert listener.received == [
        '{"kind":"move","unit_id":1,"x":0.0,"y":0.0}',
        '{"kind":"move","unit_id":2,"x":1.0,"y":2.0}',
    ]


def test_a_peer_that_hangs_up_reads_as_end_of_stream() -> None:
    """An empty line is how the channel learns the agent went away."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    bound: tuple[str, int] = server.getsockname()
    port = bound[1]

    def hang_up() -> None:
        accepted: tuple[socket.socket, tuple[str, int]] = server.accept()
        accepted[0].close()

    closer = threading.Thread(target=hang_up, daemon=True)
    closer.start()
    connection = _test_hooks.connect("127.0.0.1", port, 5.0)
    closer.join(timeout=5.0)
    try:
        assert connection.read_line() == ""
    finally:
        connection.close()
        server.close()
