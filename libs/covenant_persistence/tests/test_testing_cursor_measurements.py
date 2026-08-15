"""Tests for testing: InMemoryCursorMeasurements."""

from __future__ import annotations

import pytest
from covenant_domain import DealId, Measurement

from covenant_persistence.testing import InMemoryCursor, InMemoryStore


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
