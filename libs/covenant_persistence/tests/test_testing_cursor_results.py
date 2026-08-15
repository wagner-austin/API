"""Tests for testing: InMemoryCursorCovenantResults."""

from __future__ import annotations

import pytest
from covenant_domain import Covenant, CovenantId, CovenantResult, DealId

from covenant_persistence.testing import InMemoryCursor, InMemoryStore


class TestInMemoryCursorCovenantResults:
    """Tests for InMemoryCursor covenant result operations."""

    def test_insert_covenant_result(self) -> None:
        store = InMemoryStore()
        cursor = InMemoryCursor(store)
        cursor.execute(
            "INSERT INTO covenant_results VALUES (%s, %s, %s, %s, %s)",
            ("c1", "2025-01-01", "2025-03-31", 1200000, "OK"),
        )
        assert cursor.rowcount == 1
        assert len(store.covenant_results) == 1

    def test_insert_covenant_result_upsert(self) -> None:
        store = InMemoryStore()
        store.covenant_results.append(
            CovenantResult(
                covenant_id=CovenantId(value="c1"),
                period_start_iso="2025-01-01",
                period_end_iso="2025-03-31",
                calculated_value_scaled=100,
                status="OK",
            )
        )
        cursor = InMemoryCursor(store)
        cursor.execute(
            "INSERT INTO covenant_results VALUES (%s, %s, %s, %s, %s)",
            ("c1", "2025-01-01", "2025-03-31", 200, "BREACH"),
        )
        assert len(store.covenant_results) == 1
        assert store.covenant_results[0]["status"] == "BREACH"

    def test_select_covenant_results_by_deal(self) -> None:
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
        store.covenant_results.append(
            CovenantResult(
                covenant_id=CovenantId(value="c1"),
                period_start_iso="2025-01-01",
                period_end_iso="2025-03-31",
                calculated_value_scaled=100,
                status="OK",
            )
        )
        cursor = InMemoryCursor(store)
        sql = (
            "SELECT r.* FROM covenant_results r "
            "JOIN covenants c ON r.covenant_id = c.id WHERE c.deal_id = %s"
        )
        cursor.execute(sql, ("d1",))
        rows = cursor.fetchall()
        assert len(rows) == 1

    def test_select_covenant_results_by_covenant(self) -> None:
        store = InMemoryStore()
        store.covenant_results = [
            CovenantResult(
                covenant_id=CovenantId(value="c1"),
                period_start_iso="2025-01-01",
                period_end_iso="2025-03-31",
                calculated_value_scaled=100,
                status="OK",
            ),
            CovenantResult(
                covenant_id=CovenantId(value="c1"),
                period_start_iso="2025-04-01",
                period_end_iso="2025-06-30",
                calculated_value_scaled=200,
                status="NEAR_BREACH",
            ),
        ]
        cursor = InMemoryCursor(store)
        cursor.execute(
            "SELECT * FROM covenant_results WHERE covenant_id = %s",
            ("c1",),
        )
        rows = cursor.fetchall()
        assert len(rows) == 2

    def test_select_covenant_results_by_deal_filters_other_covenants(self) -> None:
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
        store.covenant_results = [
            CovenantResult(
                covenant_id=CovenantId(value="c1"),
                period_start_iso="2025-01-01",
                period_end_iso="2025-03-31",
                calculated_value_scaled=100,
                status="OK",
            ),
            CovenantResult(
                covenant_id=CovenantId(value="c2"),
                period_start_iso="2025-01-01",
                period_end_iso="2025-03-31",
                calculated_value_scaled=200,
                status="BREACH",
            ),
        ]
        cursor = InMemoryCursor(store)
        sql = (
            "SELECT r.* FROM covenant_results r "
            "JOIN covenants c ON r.covenant_id = c.id WHERE c.deal_id = %s"
        )
        cursor.execute(sql, ("d1",))
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "c1"

    def test_select_covenant_results_by_covenant_filters_other(self) -> None:
        store = InMemoryStore()
        store.covenant_results = [
            CovenantResult(
                covenant_id=CovenantId(value="c1"),
                period_start_iso="2025-01-01",
                period_end_iso="2025-03-31",
                calculated_value_scaled=100,
                status="OK",
            ),
            CovenantResult(
                covenant_id=CovenantId(value="c2"),
                period_start_iso="2025-01-01",
                period_end_iso="2025-03-31",
                calculated_value_scaled=200,
                status="BREACH",
            ),
        ]
        cursor = InMemoryCursor(store)
        cursor.execute(
            "SELECT * FROM covenant_results WHERE covenant_id = %s",
            ("c1",),
        )
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "c1"

    def test_select_covenant_results_unknown_query_returns_empty(self) -> None:
        store = InMemoryStore()
        store.covenant_results = [
            CovenantResult(
                covenant_id=CovenantId(value="c1"),
                period_start_iso="2025-01-01",
                period_end_iso="2025-03-31",
                calculated_value_scaled=100,
                status="OK",
            ),
        ]
        cursor = InMemoryCursor(store)
        cursor.execute("SELECT * FROM covenant_results", ())
        rows = cursor.fetchall()
        assert len(rows) == 0


class TestInMemoryCursorUnknownQuery:
    """Tests for unknown query handling."""

    def test_unknown_query_does_nothing(self) -> None:
        store = InMemoryStore()
        cursor = InMemoryCursor(store)
        cursor.execute("DROP TABLE foo")
        assert cursor.rowcount == 0
        assert cursor.fetchone() is None
        assert cursor.fetchall() == []


class TestValidationHelpers:
    """Tests for validation helper coverage."""

    def test_invalid_direction_raises(self) -> None:
        store = InMemoryStore()
        cursor = InMemoryCursor(store)
        with pytest.raises(ValueError, match="Invalid direction"):
            cursor.execute(
                "INSERT INTO covenants VALUES (%s, %s, %s, %s, %s, %s, %s)",
                ("c1", "d1", "Cov", "x", 100, "INVALID", "QUARTERLY"),
            )

    def test_invalid_frequency_raises(self) -> None:
        store = InMemoryStore()
        cursor = InMemoryCursor(store)
        with pytest.raises(ValueError, match="Invalid frequency"):
            cursor.execute(
                "INSERT INTO covenants VALUES (%s, %s, %s, %s, %s, %s, %s)",
                ("c1", "d1", "Cov", "x", 100, ">=", "INVALID"),
            )

    def test_invalid_status_raises(self) -> None:
        store = InMemoryStore()
        cursor = InMemoryCursor(store)
        with pytest.raises(ValueError, match="Invalid status"):
            cursor.execute(
                "INSERT INTO covenant_results VALUES (%s, %s, %s, %s, %s)",
                ("c1", "2025-01-01", "2025-03-31", 100, "INVALID"),
            )

    def test_direction_lte(self) -> None:
        store = InMemoryStore()
        cursor = InMemoryCursor(store)
        cursor.execute(
            "INSERT INTO covenants VALUES (%s, %s, %s, %s, %s, %s, %s)",
            ("c1", "d1", "Cov", "x", 100, "<=", "ANNUAL"),
        )
        assert store.covenants["c1"]["threshold_direction"] == "<="
        assert store.covenants["c1"]["frequency"] == "ANNUAL"

    def test_status_near_breach(self) -> None:
        store = InMemoryStore()
        cursor = InMemoryCursor(store)
        cursor.execute(
            "INSERT INTO covenant_results VALUES (%s, %s, %s, %s, %s)",
            ("c1", "2025-01-01", "2025-03-31", 100, "NEAR_BREACH"),
        )
        assert store.covenant_results[0]["status"] == "NEAR_BREACH"

    def test_status_breach(self) -> None:
        store = InMemoryStore()
        cursor = InMemoryCursor(store)
        cursor.execute(
            "INSERT INTO covenant_results VALUES (%s, %s, %s, %s, %s)",
            ("c1", "2025-01-01", "2025-03-31", 100, "BREACH"),
        )
        assert store.covenant_results[0]["status"] == "BREACH"
