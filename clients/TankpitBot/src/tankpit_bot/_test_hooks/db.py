"""Database hooks: the slice of a Postgres connection the sim server uses.

The networked sim keeps its accounts and session results in Postgres
(:mod:`tankpit_bot.sim.net_store`). The connection sits behind these
Protocols rather than psycopg's own classes, the pattern the
RustedWarfareBot match service set: production binds
:data:`connect_database` to psycopg at import, a test binds a scripted
fake, and the code between them never asks which it has.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class DbCursorProtocol(Protocol):
    """The slice of a database cursor the sim server uses."""

    def execute(self, sql: str, params: Sequence[str | int] = ()) -> None:
        """Run one statement.

        Args:
            sql: The statement, with ``%s`` placeholders.
            params: Values for the placeholders, in order.

        Raises:
            Exception: Whatever the driver raises; a failed statement is
                a failed operation, and nothing here catches it.
        """
        ...

    def fetchone(self) -> Sequence[str | int] | None:
        """The next row of the last query, or None past the end."""
        ...

    def fetchall(self) -> Sequence[Sequence[str | int]]:
        """Every remaining row of the last query."""
        ...


class DbConnectionProtocol(Protocol):
    """The slice of a database connection the sim server uses."""

    def cursor(self) -> DbCursorProtocol:
        """Open a cursor."""
        ...

    def commit(self) -> None:
        """Commit the open transaction."""
        ...

    def close(self) -> None:
        """Close the connection."""
        ...


class ConnectDatabaseProtocol(Protocol):
    """Open a connection from a libpq connection string."""

    def __call__(self, dsn: str) -> DbConnectionProtocol:
        """Open the connection.

        Args:
            dsn: A libpq connection string.

        Returns:
            The live connection.
        """
        ...


def _real_connect_database(dsn: str) -> DbConnectionProtocol:
    """Open a real Postgres connection through psycopg.

    Args:
        dsn: A libpq connection string.

    Returns:
        The live connection.

    Raises:
        Exception: Whatever psycopg raises when the server is unreachable
            or refuses the credentials.
    """
    psycopg = __import__("psycopg")
    connector: ConnectDatabaseProtocol = psycopg.connect
    return connector(dsn)


connect_database: ConnectDatabaseProtocol = _real_connect_database
"""Opens a database connection. Tests bind a scripted fake."""


__all__ = [
    "ConnectDatabaseProtocol",
    "DbConnectionProtocol",
    "DbCursorProtocol",
    "_real_connect_database",
    "connect_database",
]
