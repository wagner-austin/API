"""Test hooks for scripts modules.

Production code uses real implementations; tests override these module-level
symbols to inject fakes without conditionals in core logic.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import http.server
from pathlib import Path
from typing import Protocol


class ServeForeverProtocol(Protocol):
    """Protocol for server serve function."""

    def __call__(self, server: http.server.HTTPServer) -> None:
        """Serve HTTP requests.

        Args:
            server: The HTTP server to run.
        """
        ...


def _real_serve_forever(server: http.server.HTTPServer) -> None:
    """Real implementation calling server.serve_forever().

    Args:
        server: The HTTP server to run.
    """
    server.serve_forever()


serve_forever: ServeForeverProtocol = _real_serve_forever


class ServerFactoryProtocol(Protocol):
    """Protocol for constructing the HTTP server."""

    def __call__(
        self,
        address: tuple[str, int],
        handler: type[http.server.BaseHTTPRequestHandler],
    ) -> http.server.HTTPServer:
        """Construct a server bound to an address.

        Args:
            address: Host and port to bind.
            handler: Request handler class.

        Returns:
            A bound HTTP server.
        """
        ...


def _real_server_factory(
    address: tuple[str, int],
    handler: type[http.server.BaseHTTPRequestHandler],
) -> http.server.HTTPServer:
    """Real implementation binding the requested address.

    Args:
        address: Host and port to bind.
        handler: Request handler class.

    Returns:
        A bound HTTP server.
    """
    return http.server.HTTPServer(address, handler)


# Construction is the seam because HTTPServer binds its socket in __init__:
# without it, any test that reaches main() has to bind the real default port,
# which fails outright on hosts where that port is reserved.
server_factory: ServerFactoryProtocol = _real_server_factory


class IsDirProtocol(Protocol):
    """Protocol for checking if a path is a directory."""

    def __call__(self, path: Path) -> bool:
        """Check if path is a directory.

        Args:
            path: Path to check.

        Returns:
            True if path is a directory, False otherwise.
        """
        ...


def _real_is_dir(path: Path) -> bool:
    """Real implementation using Path.is_dir().

    Args:
        path: Path to check.

    Returns:
        True if path is a directory, False otherwise.
    """
    return path.is_dir()


is_dir: IsDirProtocol = _real_is_dir


def reset_hooks() -> None:
    """Reset all hooks to their default implementations."""
    global serve_forever, is_dir, server_factory
    serve_forever = _real_serve_forever
    is_dir = _real_is_dir
    server_factory = _real_server_factory


__all__ = [
    "IsDirProtocol",
    "ServeForeverProtocol",
    "ServerFactoryProtocol",
    "_real_is_dir",
    "_real_serve_forever",
    "_real_server_factory",
    "is_dir",
    "reset_hooks",
    "serve_forever",
    "server_factory",
]
