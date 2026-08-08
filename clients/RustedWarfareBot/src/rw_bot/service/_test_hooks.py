"""Dependency-injection hooks for the match service.

Same discipline as the control and harness hooks: every non-pure operation is
a module-level symbol bound to its real implementation at import time and
called unconditionally, so the production and test paths are identical in
shape.

The database sits behind Protocols rather than psycopg's own classes. A
worker claiming real jobs needs a real connection; a test asserting the claim
transaction's shape needs neither a server nor a socket, and the seam between
them is the only place that distinction should exist.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Protocol

from rw_bot.harness.runner import SweepConfig
from rw_bot.harness.sweep import SweepJob


class Cursor(Protocol):
    """The slice of a database cursor the service uses."""

    def execute(self, sql: str, params: Sequence[str | int | bool] = ()) -> None:
        """Run one statement.

        Args:
            sql: The statement, with ``%s`` placeholders.
            params: Values for the placeholders, in order.

        Raises:
            Exception: Whatever the driver raises; the service does not
                catch it, because a failed statement is a failed operation.
        """
        ...

    def fetchone(self) -> Sequence[str | int] | None:
        """Return the next row of the last query, or None past the end."""
        ...

    def fetchall(self) -> Sequence[Sequence[str | int]]:
        """Return every remaining row of the last query."""
        ...


class Connection(Protocol):
    """The slice of a database connection the service uses."""

    def cursor(self) -> Cursor:
        """Open a cursor."""
        ...

    def commit(self) -> None:
        """Commit the open transaction."""
        ...

    def rollback(self) -> None:
        """Abandon the open transaction."""
        ...

    def close(self) -> None:
        """Close the connection."""
        ...


def _connect_impl(dsn: str) -> Connection:
    """Open a real Postgres connection.

    Args:
        dsn: A libpq connection string.

    Returns:
        The live connection, JSON adaptation active.

    Raises:
        Exception: Whatever psycopg raises when the server is unreachable;
            the worker's caller decides whether to retry, not this seam.
    """
    psycopg = __import__("psycopg")
    connector: Callable[[str], Connection] = psycopg.connect
    return connector(dsn)


#: Opens a database connection. Tests bind a scripted fake.
connect: Callable[[str], Connection] = _connect_impl

#: Sleeps between polls. Tests bind a recorder so a loop runs in microseconds.
sleep: Callable[[float], None] = time.sleep


def _prepare_tree_impl(config: SweepConfig) -> None:
    """Freeze the batch tree through the harness, exactly as sweeps do."""
    from rw_bot.harness.runner import prepare_tree

    prepare_tree(config)


def _prepare_clone_impl(index: int, config: SweepConfig) -> str:
    """Ready one leased clone through the harness, exactly as sweeps do."""
    from rw_bot.harness.runner import prepare_clone

    return prepare_clone(index, config)


def _play_job_impl(job: SweepJob, game_dir: str, config: SweepConfig) -> bool:
    """Play one match through the harness, exactly as sweeps do."""
    from rw_bot.harness.runner import play_job

    return play_job(job, game_dir, config)


#: Freezes the batch tree. Tests bind a recorder.
prepare_tree: Callable[[SweepConfig], None] = _prepare_tree_impl

#: Readies a leased clone. Tests bind a recorder returning a fake dir.
prepare_clone: Callable[[int, SweepConfig], str] = _prepare_clone_impl

#: Plays one match. Tests bind a scripted outcome.
play_job: Callable[[SweepJob, str, SweepConfig], bool] = _play_job_impl
