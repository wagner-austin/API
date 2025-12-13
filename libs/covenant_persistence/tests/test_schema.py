"""Tests for ensure_schema in covenant_persistence.postgres.

Covers reading the schema file and executing it via a typed
ConnectionProtocol implementation while committing the transaction.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import covenant_persistence.postgres as pg
from covenant_persistence.protocols import ConnectionProtocol, CursorProtocol


class _FakeCursor(CursorProtocol):
    """Minimal cursor capturing the last executed SQL for assertions."""

    def __init__(self) -> None:
        self.last_query: str | None = None
        self.last_params: tuple[str | int | bool | None, ...] | None = None

    @property
    def rowcount(self) -> int:
        return 0

    def execute(self, query: str, params: tuple[str | int | bool | None, ...] = ()) -> None:
        self.last_query = query
        self.last_params = params

    def fetchone(self) -> tuple[str | int | bool | None, ...] | None:
        return None

    def fetchall(self) -> Sequence[tuple[str | int | bool | None, ...]]:
        return []


class _FakeConnection(ConnectionProtocol):
    """Minimal connection that returns a fake cursor and records commits."""

    def __init__(self) -> None:
        self._cursor = _FakeCursor()
        self.commits: int = 0
        self.rollbacks: int = 0
        self.closed: bool = False

    def cursor(self) -> CursorProtocol:
        return self._cursor

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


class TestEnsureSchema:
    def test_reads_and_executes_schema_and_commits(self) -> None:
        conn = _FakeConnection()

        # Act
        pg.ensure_schema(conn)

        # Assert: executed SQL matches the schema.sql contents and commit called once
        schema_path = Path(pg.__file__).parent / "schema.sql"
        expected_sql = schema_path.read_text(encoding="utf-8")

        assert conn._cursor.last_query == expected_sql
        assert conn.commits == 1
