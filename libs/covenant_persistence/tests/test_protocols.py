"""Tests for protocols module."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from covenant_persistence.protocols import (
    ConnectCallable,
    ConnectionProtocol,
    CursorProtocol,
    _get_psycopg_connect,
    connect,
)


class _TestCursorBase:
    """Base test cursor implementation."""

    def __init__(self) -> None:
        self._rowcount = 0

    def execute(self, query: str, params: tuple[str | int | bool | None, ...] = ()) -> None:
        pass

    def fetchone(self) -> tuple[str | int | bool | None, ...] | None:
        return None

    def fetchall(self) -> Sequence[tuple[str | int | bool | None, ...]]:
        return []

    @property
    def rowcount(self) -> int:
        return self._rowcount


class _TestCursorWithRow(_TestCursorBase):
    """Test cursor that returns a single row."""

    def __init__(self) -> None:
        super().__init__()
        self._rowcount = 1

    def fetchone(self) -> tuple[str | int | bool | None, ...] | None:
        return ("value", 123, True, None)


class _TestCursorWithRows(_TestCursorBase):
    """Test cursor that returns multiple rows."""

    def __init__(self) -> None:
        super().__init__()
        self._rowcount = 2

    def fetchall(self) -> Sequence[tuple[str | int | bool | None, ...]]:
        return [("row1", 1), ("row2", 2)]


class _TestConnection:
    """Test connection implementation."""

    def __init__(self) -> None:
        self._cursor = _TestCursorBase()
        self._committed = False
        self._rolled_back = False
        self._closed = False

    def cursor(self) -> CursorProtocol:
        return self._cursor

    def commit(self) -> None:
        self._committed = True

    def rollback(self) -> None:
        self._rolled_back = True

    def close(self) -> None:
        self._closed = True


class _TestConnectionWithConninfo:
    """Test connection that stores conninfo."""

    def __init__(self, conninfo: str) -> None:
        self._cursor = _TestCursorBase()
        self._conninfo = conninfo

    def cursor(self) -> CursorProtocol:
        return self._cursor

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass


def _test_connect(conninfo: str) -> ConnectionProtocol:
    """Test connect function."""
    return _TestConnectionWithConninfo(conninfo)


class TestProtocolStructuralTyping:
    """Tests that verify protocol structural typing works correctly."""

    def test_cursor_implementation_satisfies_protocol(self) -> None:
        """A class implementing cursor methods satisfies CursorProtocol."""
        cursor: CursorProtocol = _TestCursorBase()
        cursor.execute("SELECT 1")
        result_one = cursor.fetchone()
        result_all = cursor.fetchall()
        count = cursor.rowcount
        assert result_one is None
        assert result_all == []
        assert count == 0

    def test_cursor_fetchone_returns_tuple(self) -> None:
        """Cursor fetchone can return a tuple row."""
        cursor: CursorProtocol = _TestCursorWithRow()
        row = cursor.fetchone()
        assert row == ("value", 123, True, None)

    def test_cursor_fetchall_returns_rows(self) -> None:
        """Cursor fetchall returns sequence of tuple rows."""
        cursor: CursorProtocol = _TestCursorWithRows()
        rows = cursor.fetchall()
        assert len(rows) == 2
        assert rows[0] == ("row1", 1)
        assert rows[1] == ("row2", 2)

    def test_connection_implementation_satisfies_protocol(self) -> None:
        """A class implementing connection methods satisfies ConnectionProtocol."""
        test_conn = _TestConnection()
        conn: ConnectionProtocol = test_conn
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        conn.commit()
        conn.rollback()
        conn.close()
        assert test_conn._committed
        assert test_conn._rolled_back
        assert test_conn._closed

    def test_connect_callable_implementation_satisfies_protocol(self) -> None:
        """A callable returning ConnectionProtocol satisfies ConnectCallable."""
        connect_fn: ConnectCallable = _test_connect
        conn = connect_fn("postgresql://localhost/test")
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        assert cursor.rowcount == 0


class TestConnectContextManager:
    """Tests for the connect context manager."""

    def test_get_psycopg_connect_returns_callable(self) -> None:
        """_get_psycopg_connect returns psycopg.connect function."""
        connect_fn = _get_psycopg_connect()
        assert callable(connect_fn)

    def test_connect_with_custom_connect_fn(self) -> None:
        """Connect uses provided connect_fn instead of psycopg."""
        test_conn = _TestConnection()

        def custom_connect(conninfo: str) -> ConnectionProtocol:
            return test_conn

        with connect("test_dsn", connect_fn=custom_connect) as conn:
            typed_conn: ConnectionProtocol = conn
            cursor = typed_conn.cursor()
            cursor.execute("SELECT 1")

        assert test_conn._closed

    def test_connect_closes_connection_on_exception(self) -> None:
        """Connect closes connection in finally block even when exception raised."""
        test_conn = _TestConnection()

        def custom_connect(conninfo: str) -> ConnectionProtocol:
            return test_conn

        with pytest.raises(ValueError), connect("test_dsn", connect_fn=custom_connect):
            raise ValueError("Test error")

        assert test_conn._closed

    def test_connect_uses_psycopg_when_no_connect_fn_provided(self) -> None:
        """Connect uses psycopg.connect when connect_fn is None."""
        # Verify that _get_psycopg_connect returns a callable that matches the protocol
        connect_fn: ConnectCallable = _get_psycopg_connect()
        # The returned function should be callable and assignable to our protocol
        assert callable(connect_fn)

    def test_connect_default_uses_psycopg_and_raises_on_invalid_dsn(self) -> None:
        """Connect without connect_fn uses psycopg.connect and raises on invalid DSN."""
        # This test covers the branch where connect_fn is None
        # psycopg.connect will fail with OperationalError on invalid connection
        psycopg = __import__("psycopg")
        operational_error: type[Exception] = psycopg.OperationalError

        # Use connection string that fails fast (no host resolution needed)
        with pytest.raises(operational_error), connect("host= dbname=x"):
            pass
