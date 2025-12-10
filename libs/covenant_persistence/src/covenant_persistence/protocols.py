"""Database protocol definitions for psycopg with strict typing."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Protocol


class CursorProtocol(Protocol):
    """Protocol for psycopg cursor."""

    def execute(self, query: str, params: tuple[str | int | bool | None, ...] = ()) -> None:
        """Execute a query with parameters."""
        ...

    def fetchone(self) -> tuple[str | int | bool | None, ...] | None:
        """Fetch one row or None if no more rows."""
        ...

    def fetchall(self) -> Sequence[tuple[str | int | bool | None, ...]]:
        """Fetch all remaining rows."""
        ...

    @property
    def rowcount(self) -> int:
        """Number of rows affected by the last execute."""
        ...


class ConnectionProtocol(Protocol):
    """Protocol for psycopg connection."""

    def cursor(self) -> CursorProtocol:
        """Create a new cursor."""
        ...

    def commit(self) -> None:
        """Commit the current transaction."""
        ...

    def rollback(self) -> None:
        """Rollback the current transaction."""
        ...

    def close(self) -> None:
        """Close the connection."""
        ...


class ConnectCallable(Protocol):
    """Protocol for psycopg.connect function."""

    def __call__(self, conninfo: str) -> ConnectionProtocol:
        """Connect to database and return connection."""
        ...


def _get_psycopg_connect() -> ConnectCallable:
    """Get psycopg.connect function with typed interface."""
    psycopg = __import__("psycopg")
    connect_fn: ConnectCallable = psycopg.connect
    return connect_fn


@contextmanager
def connect(
    dsn: str,
    connect_fn: ConnectCallable | None = None,
) -> Generator[ConnectionProtocol, None, None]:
    """Context manager for database connection with typed interface.

    Args:
        dsn: Database connection string
        connect_fn: Optional connection callable (uses psycopg.connect if not provided)
    """
    if connect_fn is None:
        connect_fn = _get_psycopg_connect()
    conn = connect_fn(dsn)
    try:
        yield conn
    finally:
        conn.close()


__all__ = [
    "ConnectCallable",
    "ConnectionProtocol",
    "CursorProtocol",
    "connect",
]
