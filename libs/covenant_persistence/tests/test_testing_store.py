"""Tests for testing: InMemoryStore."""

from __future__ import annotations

from covenant_persistence.testing import InMemoryConnection, InMemoryStore


class TestInMemoryStore:
    """Tests for InMemoryStore."""

    def test_init_creates_empty_collections(self) -> None:
        store = InMemoryStore()
        assert store.deals == {}
        assert store.covenants == {}
        assert store.measurements == []
        assert store.covenant_results == []
        assert store._deal_order == []
        assert store._covenant_order == []


class TestInMemoryConnection:
    """Tests for InMemoryConnection."""

    def test_cursor_returns_cursor_protocol(self) -> None:
        store = InMemoryStore()
        conn = InMemoryConnection(store)
        cursor = conn.cursor()
        # Verify cursor is an InMemoryCursor by testing actual methods
        cursor.execute("SELECT * FROM deals")
        assert cursor.fetchone() is None
        assert cursor.fetchall() == []
        assert cursor.rowcount == 0

    def test_commit_is_noop(self) -> None:
        store = InMemoryStore()
        conn = InMemoryConnection(store)
        conn.commit()  # Should not raise

    def test_rollback_is_noop(self) -> None:
        store = InMemoryStore()
        conn = InMemoryConnection(store)
        conn.rollback()  # Should not raise

    def test_close_sets_closed_flag(self) -> None:
        store = InMemoryStore()
        conn = InMemoryConnection(store)
        assert conn.closed is False
        conn.close()
        assert conn.closed is True
