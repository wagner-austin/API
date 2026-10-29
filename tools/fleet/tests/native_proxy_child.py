"""Real byte-copying subprocess twin of Codex's WebSocket stdio proxy."""

from __future__ import annotations

import select
import socket
import sys


def main() -> None:
    """Forward the inherited duplex standard stream to the test server."""
    with (
        socket.socket(fileno=0) as source,
        socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=5) as destination,
    ):
        while True:
            unused: list[socket.socket] = []
            readiness: tuple[list[socket.socket], list[socket.socket], list[socket.socket]] = (
                select.select([source, destination], unused, unused, 5)
            )
            for stream in readiness[0]:
                payload = stream.recv(65536)
                if not payload:
                    return
                target = destination if stream is source else source
                target.sendall(payload)


if __name__ == "__main__":
    main()


__all__ = ["main"]
