"""Tests for covenant_persistence.testing in-memory implementations."""

from __future__ import annotations

import pytest
from covenant_domain import Covenant, CovenantId, CovenantResult, Deal, DealId, Measurement

from covenant_persistence.testing import InMemoryConnection, InMemoryCursor, InMemoryStore


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
        conn.close()
        assert conn._closed is True


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


class TestInMemoryCursorMeasurements:
    """Tests for InMemoryCursor measurement operations."""

    def test_insert_measurement(self) -> None:
        store = InMemoryStore()
        cursor = InMemoryCursor(store)
        cursor.execute(
            "INSERT INTO measurements VALUES (%s, %s, %s, %s, %s)",
            ("d1", "2025-01-01", "2025-03-31", "ebitda", 5000000),
        )
        assert cursor.rowcount == 1
        assert len(store.measurements) == 1

    def test_insert_duplicate_measurement_raises(self) -> None:
        store = InMemoryStore()
        store.measurements.append(
            Measurement(
                deal_id=DealId(value="d1"),
                period_start_iso="2025-01-01",
                period_end_iso="2025-03-31",
                metric_name="ebitda",
                metric_value_scaled=100,
            )
        )
        cursor = InMemoryCursor(store)
        with pytest.raises(ValueError, match="Duplicate measurement"):
            cursor.execute(
                "INSERT INTO measurements VALUES (%s, %s, %s, %s, %s)",
                ("d1", "2025-01-01", "2025-03-31", "ebitda", 200),
            )

    def test_select_measurements_by_deal_and_period(self) -> None:
        store = InMemoryStore()
        store.measurements = [
            Measurement(
                deal_id=DealId(value="d1"),
                period_start_iso="2025-01-01",
                period_end_iso="2025-03-31",
                metric_name="ebitda",
                metric_value_scaled=100,
            ),
            Measurement(
                deal_id=DealId(value="d1"),
                period_start_iso="2025-01-01",
                period_end_iso="2025-03-31",
                metric_name="interest",
                metric_value_scaled=50,
            ),
        ]
        cursor = InMemoryCursor(store)
        sql = (
            "SELECT * FROM measurements WHERE deal_id = %s "
            "AND period_start = %s AND period_end = %s"
        )
        cursor.execute(sql, ("d1", "2025-01-01", "2025-03-31"))
        rows = cursor.fetchall()
        assert len(rows) == 2
        # Sorted by metric_name
        assert rows[0][3] == "ebitda"
        assert rows[1][3] == "interest"

    def test_select_measurements_by_deal_only(self) -> None:
        store = InMemoryStore()
        store.measurements = [
            Measurement(
                deal_id=DealId(value="d1"),
                period_start_iso="2025-01-01",
                period_end_iso="2025-03-31",
                metric_name="ebitda",
                metric_value_scaled=100,
            ),
            Measurement(
                deal_id=DealId(value="d1"),
                period_start_iso="2025-04-01",
                period_end_iso="2025-06-30",
                metric_name="ebitda",
                metric_value_scaled=200,
            ),
        ]
        cursor = InMemoryCursor(store)
        cursor.execute("SELECT * FROM measurements WHERE deal_id = %s", ("d1",))
        rows = cursor.fetchall()
        assert len(rows) == 2

    def test_select_measurements_filters_other_deals(self) -> None:
        store = InMemoryStore()
        store.measurements = [
            Measurement(
                deal_id=DealId(value="d1"),
                period_start_iso="2025-01-01",
                period_end_iso="2025-03-31",
                metric_name="ebitda",
                metric_value_scaled=100,
            ),
            Measurement(
                deal_id=DealId(value="d2"),
                period_start_iso="2025-01-01",
                period_end_iso="2025-03-31",
                metric_name="ebitda",
                metric_value_scaled=200,
            ),
        ]
        cursor = InMemoryCursor(store)
        cursor.execute("SELECT * FROM measurements WHERE deal_id = %s", ("d1",))
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][4] == 100

    def test_select_measurements_by_period_filters_non_matching(self) -> None:
        store = InMemoryStore()
        store.measurements = [
            Measurement(
                deal_id=DealId(value="d1"),
                period_start_iso="2025-01-01",
                period_end_iso="2025-03-31",
                metric_name="ebitda",
                metric_value_scaled=100,
            ),
            Measurement(
                deal_id=DealId(value="d1"),
                period_start_iso="2025-04-01",
                period_end_iso="2025-06-30",
                metric_name="ebitda",
                metric_value_scaled=200,
            ),
        ]
        cursor = InMemoryCursor(store)
        sql = (
            "SELECT * FROM measurements WHERE deal_id = %s "
            "AND period_start = %s AND period_end = %s"
        )
        cursor.execute(sql, ("d1", "2025-01-01", "2025-03-31"))
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][4] == 100

    def test_insert_measurement_empty_store(self) -> None:
        store = InMemoryStore()
        cursor = InMemoryCursor(store)
        cursor.execute(
            "INSERT INTO measurements VALUES (%s, %s, %s, %s, %s)",
            ("d1", "2025-01-01", "2025-03-31", "ebitda", 5000000),
        )
        assert len(store.measurements) == 1

    def test_insert_measurement_different_metric(self) -> None:
        store = InMemoryStore()
        store.measurements.append(
            Measurement(
                deal_id=DealId(value="d1"),
                period_start_iso="2025-01-01",
                period_end_iso="2025-03-31",
                metric_name="ebitda",
                metric_value_scaled=100,
            )
        )
        cursor = InMemoryCursor(store)
        cursor.execute(
            "INSERT INTO measurements VALUES (%s, %s, %s, %s, %s)",
            ("d1", "2025-01-01", "2025-03-31", "interest", 50),
        )
        assert len(store.measurements) == 2


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
