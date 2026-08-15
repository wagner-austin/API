"""Tests for testing: InMemoryCursorDeals."""

from __future__ import annotations

import pytest
from covenant_domain import Deal, DealId

from covenant_persistence.testing import InMemoryCursor, InMemoryStore


class TestInMemoryCursorDeals:
    """Tests for InMemoryCursor deal operations."""

    def test_insert_deal(self) -> None:
        store = InMemoryStore()
        cursor = InMemoryCursor(store)
        sql = (
            "INSERT INTO deals "
            "(id, name, borrower, sector, region, commitment, currency, maturity) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        )
        params = ("d1", "Deal One", "Borrower A", "Tech", "NA", 1000000, "USD", "2025-12-31")
        cursor.execute(sql, params)
        assert cursor.rowcount == 1
        assert "d1" in store.deals
        assert store.deals["d1"]["name"] == "Deal One"

    def test_insert_duplicate_deal_raises(self) -> None:
        store = InMemoryStore()
        cursor = InMemoryCursor(store)
        cursor.execute(
            "INSERT INTO deals VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            ("d1", "Deal", "B", "S", "R", 100, "USD", "2025-01-01"),
        )
        with pytest.raises(ValueError, match="Duplicate deal ID"):
            cursor.execute(
                "INSERT INTO deals VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                ("d1", "Dup", "B", "S", "R", 100, "USD", "2025-01-01"),
            )

    def test_select_deal_by_id(self) -> None:
        store = InMemoryStore()
        store.deals["d1"] = Deal(
            id=DealId(value="d1"),
            name="Test",
            borrower="B",
            sector="S",
            region="R",
            commitment_amount_cents=100,
            currency="USD",
            maturity_date_iso="2025-01-01",
        )
        cursor = InMemoryCursor(store)
        cursor.execute("SELECT * FROM deals WHERE id = %s", ("d1",))
        row = cursor.fetchone()
        assert row == ("d1", "Test", "B", "S", "R", 100, "USD", "2025-01-01")

    def test_select_deal_not_found(self) -> None:
        store = InMemoryStore()
        cursor = InMemoryCursor(store)
        cursor.execute("SELECT * FROM deals WHERE id = %s", ("missing",))
        assert cursor.fetchone() is None

    def test_select_all_deals(self) -> None:
        store = InMemoryStore()
        store._deal_order = ["d1", "d2"]
        store.deals["d1"] = Deal(
            id=DealId(value="d1"),
            name="A",
            borrower="B",
            sector="S",
            region="R",
            commitment_amount_cents=100,
            currency="USD",
            maturity_date_iso="2025-01-01",
        )
        store.deals["d2"] = Deal(
            id=DealId(value="d2"),
            name="B",
            borrower="B",
            sector="S",
            region="R",
            commitment_amount_cents=200,
            currency="USD",
            maturity_date_iso="2025-01-01",
        )
        cursor = InMemoryCursor(store)
        cursor.execute("SELECT * FROM deals")
        rows = cursor.fetchall()
        assert len(rows) == 2
        # Reversed order (created_at DESC)
        assert rows[0][0] == "d2"
        assert rows[1][0] == "d1"

    def test_update_deal(self) -> None:
        store = InMemoryStore()
        store.deals["d1"] = Deal(
            id=DealId(value="d1"),
            name="Old",
            borrower="B",
            sector="S",
            region="R",
            commitment_amount_cents=100,
            currency="USD",
            maturity_date_iso="2025-01-01",
        )
        cursor = InMemoryCursor(store)
        sql = (
            "UPDATE deals SET name=%s, borrower=%s, sector=%s, region=%s, "
            "commitment=%s, currency=%s, maturity=%s WHERE id=%s"
        )
        cursor.execute(sql, ("New", "B", "S", "R", 200, "EUR", "2026-01-01", "d1"))
        assert cursor.rowcount == 1
        assert store.deals["d1"]["name"] == "New"

    def test_update_deal_not_found(self) -> None:
        store = InMemoryStore()
        cursor = InMemoryCursor(store)
        sql = (
            "UPDATE deals SET name=%s, borrower=%s, sector=%s, region=%s, "
            "commitment=%s, currency=%s, maturity=%s WHERE id=%s"
        )
        cursor.execute(sql, ("New", "B", "S", "R", 200, "EUR", "2026-01-01", "missing"))
        assert cursor.rowcount == 0

    def test_delete_deal(self) -> None:
        store = InMemoryStore()
        store.deals["d1"] = Deal(
            id=DealId(value="d1"),
            name="A",
            borrower="B",
            sector="S",
            region="R",
            commitment_amount_cents=100,
            currency="USD",
            maturity_date_iso="2025-01-01",
        )
        cursor = InMemoryCursor(store)
        cursor.execute("DELETE FROM deals WHERE id = %s", ("d1",))
        assert cursor.rowcount == 1
        assert "d1" not in store.deals

    def test_delete_deal_not_found(self) -> None:
        store = InMemoryStore()
        cursor = InMemoryCursor(store)
        cursor.execute("DELETE FROM deals WHERE id = %s", ("missing",))
        assert cursor.rowcount == 0
