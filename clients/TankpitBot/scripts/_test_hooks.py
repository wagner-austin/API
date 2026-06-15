"""Test hooks for scripts.

Production code uses real implementations; tests override these module-level
symbols to inject fakes without conditionals in core logic.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

from tankpit_bot.decoder import DecodedCommand, DecodedLobbyMessage

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class PathExistsProtocol(Protocol):
    """Protocol for checking if a path exists."""

    def __call__(self, path: Path) -> bool:
        """Check if path exists.

        Args:
            path: Path to check.

        Returns:
            True if path exists, False otherwise.
        """
        ...


def _real_path_exists(path: Path) -> bool:
    """Real implementation using Path.exists().

    Args:
        path: Path to check.

    Returns:
        True if path exists, False otherwise.
    """
    return path.exists()


path_exists: PathExistsProtocol = _real_path_exists


class ReadTextProtocol(Protocol):
    """Protocol for reading text from a file."""

    def __call__(self, path: Path) -> str:
        """Read text from a file.

        Args:
            path: Path to the file.

        Returns:
            File contents as string.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        ...


def _real_read_text(path: Path) -> str:
    """Real implementation using Path.read_text().

    Args:
        path: Path to the file.

    Returns:
        File contents as string.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    return path.read_text(encoding="utf-8")


read_text: ReadTextProtocol = _real_read_text


class SessionDecoderProtocol(Protocol):
    """Protocol for session decoder interface."""

    @property
    def commands(self) -> list[DecodedCommand]:
        """Get decoded commands."""
        ...

    @property
    def lobby_messages(self) -> list[DecodedLobbyMessage]:
        """Get decoded lobby messages."""
        ...


# Type alias for load_and_decode_session hook
LoadAndDecodeSessionFunc = Callable[[Path], SessionDecoderProtocol]


def _real_load_and_decode_session(session_path: Path) -> SessionDecoderProtocol:
    """Real implementation using decoder module.

    Args:
        session_path: Path to session JSON file.

    Returns:
        SessionDecoder with decoded messages.

    Raises:
        MissingMagicError: If session has no magic key.
        FileNotFoundError: If file doesn't exist.
    """
    from tankpit_bot.decoder import load_and_decode_session as real_load

    return real_load(session_path)


load_and_decode_session: LoadAndDecodeSessionFunc = _real_load_and_decode_session


class SetupRichLoggingProtocol(Protocol):
    """Protocol for setting up rich logging."""

    def __call__(self, level: LogLevel) -> None:
        """Set up rich logging.

        Args:
            level: Log level string.
        """
        ...


def _real_setup_rich_logging(level: LogLevel) -> None:
    """Real implementation using platform_core.

    Args:
        level: Log level string.
    """
    from platform_core.logging import setup_rich_logging as real_setup

    real_setup(level=level)


setup_rich_logging: SetupRichLoggingProtocol = _real_setup_rich_logging


class HttpGetResponseProtocol(Protocol):
    """Protocol for an HTTP GET response."""

    @property
    def status_code(self) -> int:
        """HTTP status code."""
        ...

    @property
    def content(self) -> bytes:
        """Response body bytes."""
        ...


class HttpGetProtocol(Protocol):
    """Protocol for performing HTTP GET requests."""

    def __call__(self, url: str) -> HttpGetResponseProtocol:
        """Perform an HTTP GET request.

        Args:
            url: URL to fetch.

        Returns:
            Response with status_code and content.
        """
        ...


def _real_http_get(url: str) -> HttpGetResponseProtocol:
    """Real implementation using httpx.

    Args:
        url: URL to fetch.

    Returns:
        httpx Response.
    """
    httpx_mod = __import__("httpx")
    response: HttpGetResponseProtocol = httpx_mod.get(url, timeout=30.0, follow_redirects=True)
    return response


http_get: HttpGetProtocol = _real_http_get


__all__ = [
    "HttpGetProtocol",
    "HttpGetResponseProtocol",
    "LoadAndDecodeSessionFunc",
    "LogLevel",
    "PathExistsProtocol",
    "ReadTextProtocol",
    "SessionDecoderProtocol",
    "SetupRichLoggingProtocol",
    "http_get",
    "load_and_decode_session",
    "path_exists",
    "read_text",
    "setup_rich_logging",
]
