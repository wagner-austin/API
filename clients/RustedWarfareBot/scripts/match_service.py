"""Serve the match service's HTTP door until interrupted.

The phase-one control plane of [[harness-match-service]], in the fleet
server's own shape: a stdlib HTTP server whose every request routes through
one pure function, with the queue connection carried on the server object
the way the fleet carries its manager.

Usage::

    poetry run python -m scripts.match_service <dsn> [port]

Workers do not talk to this; they claim from the database directly. This
door exists so a session without the repo -- another machine, another AI --
can submit a batch and watch its counts with nothing but HTTP.
"""

from __future__ import annotations

from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, HTTPServer

from rw_bot.harness import _test_hooks as host_hooks
from rw_bot.service import _test_hooks
from rw_bot.service._test_hooks import Connection
from rw_bot.service.http import route_service_request
from rw_bot.service.queue import MatchServiceError, bootstrap

EXIT_OK = 0
EXIT_BAD_USAGE = 2

#: Default listen port, one above the fleet page's 27500.
SERVICE_PORT_DEFAULT = 27501


class MatchServiceServer(HTTPServer):
    """The service's HTTP server, carrying its connection to the handlers."""

    def __init__(self, address: tuple[str, int], conn: Connection) -> None:
        """Bind the server.

        Args:
            address: ``(host, port)`` to bind.
            conn: The open queue connection every request operates on.
        """
        super().__init__(address, MatchServiceRequestHandler)
        self.queue_connection = conn


class MatchServiceRequestHandler(BaseHTTPRequestHandler):
    """Thin socket shell around :func:`route_service_request`."""

    def _dispatch(self, method: str) -> None:
        """Read one request, route it, write the response.

        Args:
            method: HTTP method, upper-case.

        Raises:
            MatchServiceError: If the handler is serving for something
                other than a :class:`MatchServiceServer` -- a wiring bug,
                not a request fault.
        """
        server = self.server
        if not isinstance(server, MatchServiceServer):
            raise MatchServiceError("RW-SERVICE-002", "handler bound to a non-service server")
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        status, content_type, payload = route_service_request(
            server.queue_connection, method, self.path, body
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        """Serve one GET."""
        self._dispatch("GET")

    def do_POST(self) -> None:
        """Serve one POST."""
        self._dispatch("POST")

    def log_message(self, format: str, *args: str | int) -> None:
        """Route the per-request access line through the output hook.

        Args:
            format: printf-style format string.
            *args: Format arguments.
        """
        host_hooks.write_line(f"[service] {self.address_string()} {format % args}")


def main(argv: Sequence[str] | None = None) -> int:
    """Serve until interrupted.

    Args:
        argv: ``<dsn> [port]``. ``None`` reads the process arguments.

    Returns:
        ``EXIT_OK`` on a clean interrupt, ``EXIT_BAD_USAGE`` on a bad
        argument count.

    Raises:
        Exception: Whatever the database driver raises when the queue is
            unreachable at startup.
    """
    args = list(argv) if argv is not None else host_hooks.read_argv()
    if len(args) not in (1, 2):
        host_hooks.write_line("usage: match_service <dsn> [port]")
        return EXIT_BAD_USAGE
    port = int(args[1]) if len(args) == 2 else SERVICE_PORT_DEFAULT
    conn = _test_hooks.connect(args[0])
    # The door migrates at startup like every worker does, so a read route
    # never meets a table an older writer shaped.
    bootstrap(conn)
    server = MatchServiceServer(("127.0.0.1", port), conn)
    host_hooks.write_line(f"[service] listening on http://127.0.0.1:{port}/")
    host_hooks.serve_forever(server)
    server.server_close()
    conn.close()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))
