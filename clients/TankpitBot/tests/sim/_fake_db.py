"""A scripted database for the sim's account store, bound through ``_test_hooks``.

The store's statements are asserted for their shape (which statement,
which values, committed or not), and every query answers the rows the
test scripted for it, in order. Nothing here parses SQL: a test that
needs a row says so.
"""

from __future__ import annotations

from collections.abc import Sequence

DSN_VARIABLE = "TANKPIT_SIM_TEST_DSN"
DSN = "host=sim-db dbname=tankpit_sim user=sim"


class FakeDatabase:
    """Every statement run against it, the rows it will answer, and its connections."""

    def __init__(self) -> None:
        """Start with nothing run and nothing to answer."""
        self.answers: list[list[Sequence[str | int]]] = []
        self.executed: list[tuple[str, tuple[str | int, ...]]] = []
        self.commits = 0
        self.dsns: list[str] = []
        self.closed = 0

    def connect(self, dsn: str) -> FakeConnection:
        """Open a connection, recording the string it was given.

        Args:
            dsn: The connection string.

        Returns:
            A connection to this database.
        """
        self.dsns.append(dsn)
        return FakeConnection(self)

    def statements(self) -> list[str]:
        """Each statement run, its whitespace collapsed, in order."""
        return [" ".join(sql.split()) for sql, _ in self.executed]


class FakeConnection:
    """One connection to a :class:`FakeDatabase`."""

    def __init__(self, database: FakeDatabase) -> None:
        """Bind to the database.

        Args:
            database: The database.
        """
        self._database = database

    def cursor(self) -> FakeCursor:
        """A cursor over the database."""
        return FakeCursor(self._database)

    def commit(self) -> None:
        """Count a commit."""
        self._database.commits += 1

    def close(self) -> None:
        """Count a close."""
        self._database.closed += 1


class FakeCursor:
    """Runs statements by recording them; a query takes the next scripted answer."""

    def __init__(self, database: FakeDatabase) -> None:
        """Bind to the database.

        Args:
            database: The database.
        """
        self._database = database
        self._rows: list[Sequence[str | int]] = []

    def execute(self, sql: str, params: Sequence[str | int] = ()) -> None:
        """Record a statement; a SELECT takes the next scripted answer.

        Args:
            sql: The statement.
            params: Its values.

        Raises:
            AssertionError: If a query runs with no answer scripted.
        """
        self._database.executed.append((sql, tuple(params)))
        if sql.lstrip().startswith("SELECT"):
            if not self._database.answers:
                raise AssertionError(f"no answer scripted for {sql!r}")
            self._rows = self._database.answers.pop(0)

    def fetchone(self) -> Sequence[str | int] | None:
        """The first row of the last answer, or None when it had none."""
        return self._rows[0] if self._rows else None

    def fetchall(self) -> Sequence[Sequence[str | int]]:
        """Every row of the last answer."""
        return self._rows


def scripted_env(key: str) -> str | None:
    """The environment a test sees: the test string in the test variable, nothing else.

    Args:
        key: The variable asked for.

    Returns:
        :data:`DSN` for :data:`DSN_VARIABLE`, else None.
    """
    return DSN if key == DSN_VARIABLE else None
