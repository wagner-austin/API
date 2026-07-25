"""Dependency-injection hooks for the control package.

Same discipline as the harness hooks: every non-pure operation is a
module-level symbol bound to its real implementation at import time and called
unconditionally, so the production and test paths are identical in shape.

The socket is behind a Protocol rather than used directly. A planner talking to
a real game needs a real socket; a test asserting the planner's decisions needs
neither a socket nor a game, and the seam between them is the only place that
distinction should exist.
"""

from __future__ import annotations

import socket
from typing import Protocol


class Connection(Protocol):
    """A duplex line channel to the agent."""

    def send_line(self, line: str) -> None:
        """Write one line, terminated.

        Args:
            line: Line content, without a newline.

        Raises:
            OSError: When the write fails.
        """
        ...

    def read_line(self) -> str:
        """Read one line.

        Returns:
            The line without its terminator, or an empty string at end of
            stream.

        Raises:
            OSError: When the read fails.
        """
        ...

    def close(self) -> None:
        """Release the channel."""
        ...


class ConnectProto(Protocol):
    """Open a connection to the agent."""

    def __call__(self, host: str, port: int, timeout_s: float) -> Connection:
        """Connect.

        Args:
            host: Host to reach the agent on.
            port: Port the agent listens on.
            timeout_s: Socket timeout in seconds.

        Returns:
            The open connection.

        Raises:
            OSError: When the connection cannot be established.
        """
        ...


class _SocketConnection:
    """A :class:`Connection` over a real TCP socket.

    Attributes:
        _socket: The connected socket.
        _reader: Buffered text reader over the socket.
    """

    def __init__(self, sock: socket.socket) -> None:
        self._socket = sock
        self._reader = sock.makefile("r", encoding="utf-8", newline="\n")

    def send_line(self, line: str) -> None:
        """Write one line, terminated.

        Args:
            line: Line content, without a newline.

        Raises:
            OSError: When the write fails.
        """
        self._socket.sendall(f"{line}\n".encode())

    def read_line(self) -> str:
        """Read one line.

        Returns:
            The line without its terminator, or an empty string at end of
            stream.

        Raises:
            OSError: When the read fails.
        """
        return self._reader.readline().rstrip("\n")

    def close(self) -> None:
        """Close the reader and the socket."""
        self._reader.close()
        self._socket.close()


def _connect_impl(host: str, port: int, timeout_s: float) -> Connection:
    """Production implementation of :class:`ConnectProto`.

    Args:
        host: Host to reach the agent on.
        port: Port the agent listens on.
        timeout_s: Socket timeout in seconds.

    Returns:
        The open connection.

    Raises:
        OSError: When the connection cannot be established.
    """
    sock = socket.create_connection((host, port), timeout=timeout_s)
    return _SocketConnection(sock)


connect: ConnectProto = _connect_impl


__all__ = ["ConnectProto", "Connection", "connect"]
