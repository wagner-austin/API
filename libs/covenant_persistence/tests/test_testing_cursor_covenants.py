"""Tests for testing: InMemoryCursorCovenants."""

from __future__ import annotations

import pytest
from covenant_domain import Covenant, CovenantId, DealId

from covenant_persistence.testing import InMemoryCursor, InMemoryStore


class TestInMemoryCursorCovenants:
    """Tests for InMemoryCursor covenant operations."""

    def test_insert_covenant(self) -> None:
        store = InMemoryStore()
        cursor = InMemoryCursor(store)
        cursor.execute(
            "INSERT INTO covenants VALUES (%s, %s, %s, %s, %s, %s, %s)",
            ("c1", "d1", "Coverage", "ebitda / interest", 1500000, ">=", "QUARTERLY"),
        )
        assert cursor.rowcount == 1
        assert "c1" in store.covenants

    def test_insert_duplicate_covenant_raises(self) -> None:
        store = InMemoryStore()
        cursor = InMemoryCursor(store)
        cursor.execute(
            "INSERT INTO covenants VALUES (%s, %s, %s, %s, %s, %s, %s)",
            ("c1", "d1", "Cov", "x", 100, ">=", "QUARTERLY"),
        )
        with pytest.raises(ValueError, match="Duplicate covenant ID"):
            cursor.execute(
                "INSERT INTO covenants VALUES (%s, %s, %s, %s, %s, %s, %s)",
                ("c1", "d1", "Dup", "y", 200, "<=", "ANNUAL"),
            )

    def test_select_covenant_by_id(self) -> None:
        store = InMemoryStore()
        store.covenants["c1"] = Covenant(
            id=CovenantId(value="c1"),
            deal_id=DealId(value="d1"),
            name="Cov",
            formula="x",
            threshold_value_scaled=100,
            threshold_direction=">=",
            frequency="QUARTERLY",
        )
        cursor = InMemoryCursor(store)
        cursor.execute("SELECT * FROM covenants WHERE id = %s", ("c1",))
        row = cursor.fetchone()
        assert row == ("c1", "d1", "Cov", "x", 100, ">=", "QUARTERLY")

    def test_select_covenants_by_deal_id(self) -> None:
        store = InMemoryStore()
        store._covenant_order = ["c1", "c2"]
        store.covenants["c1"] = Covenant(
            id=CovenantId(value="c1"),
            deal_id=DealId(value="d1"),
            name="A",
            formula="x",
            threshold_value_scaled=100,
            threshold_direction=">=",
            frequency="QUARTERLY",
        )
        store.covenants["c2"] = Covenant(
            id=CovenantId(value="c2"),
            deal_id=DealId(value="d1"),
            name="B",
            formula="y",
            threshold_value_scaled=200,
            threshold_direction="<=",
            frequency="ANNUAL",
        )
        cursor = InMemoryCursor(store)
        cursor.execute("SELECT * FROM covenants WHERE deal_id = %s", ("d1",))
        rows = cursor.fetchall()
        assert len(rows) == 2

    def test_delete_covenant(self) -> None:
        store = InMemoryStore()
        store.covenants["c1"] = Covenant(
            id=CovenantId(value="c1"),
            deal_id=DealId(value="d1"),
            name="A",
            formula="x",
            threshold_value_scaled=100,
            threshold_direction=">=",
            frequency="QUARTERLY",
        )
        cursor = InMemoryCursor(store)
        cursor.execute("DELETE FROM covenants WHERE id = %s", ("c1",))
        assert cursor.rowcount == 1
        assert "c1" not in store.covenants

    def test_delete_covenant_not_found(self) -> None:
        store = InMemoryStore()
        cursor = InMemoryCursor(store)
        cursor.execute("DELETE FROM covenants WHERE id = %s", ("missing",))
        assert cursor.rowcount == 0

    def test_select_covenant_by_id_not_found(self) -> None:
        store = InMemoryStore()
        cursor = InMemoryCursor(store)
        cursor.execute("SELECT * FROM covenants WHERE id = %s", ("missing",))
        assert cursor.fetchone() is None

    def test_select_covenants_by_deal_with_stale_order(self) -> None:
        store = InMemoryStore()
        store._covenant_order = ["c1", "c2"]
        store.covenants["c1"] = Covenant(
            id=CovenantId(value="c1"),
            deal_id=DealId(value="d1"),
            name="A",
            formula="x",
            threshold_value_scaled=100,
            threshold_direction=">=",
            frequency="QUARTERLY",
        )
        cursor = InMemoryCursor(store)
        cursor.execute("SELECT * FROM covenants WHERE deal_id = %s", ("d1",))
        rows = cursor.fetchall()
        assert len(rows) == 1

    def test_select_covenants_by_deal_filters_other_deals(self) -> None:
        store = InMemoryStore()
        store._covenant_order = ["c1", "c2"]
        store.covenants["c1"] = Covenant(
            id=CovenantId(value="c1"),
            deal_id=DealId(value="d1"),
            name="A",
            formula="x",
            threshold_value_scaled=100,
            threshold_direction=">=",
            frequency="QUARTERLY",
        )
        store.covenants["c2"] = Covenant(
            id=CovenantId(value="c2"),
            deal_id=DealId(value="d2"),
            name="B",
            formula="y",
            threshold_value_scaled=200,
            threshold_direction="<=",
            frequency="ANNUAL",
        )
        cursor = InMemoryCursor(store)
        cursor.execute("SELECT * FROM covenants WHERE deal_id = %s", ("d1",))
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "c1"
