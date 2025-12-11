"""Tests for PostgreSQL repository implementations."""

from __future__ import annotations

import pytest
from covenant_domain.models import (
    Covenant,
    CovenantResult,
    Deal,
    Measurement,
)

from covenant_persistence.postgres import (
    PostgresCovenantRepository,
    PostgresCovenantResultRepository,
    PostgresDealRepository,
    PostgresMeasurementRepository,
)
from covenant_persistence.protocols import ConnectionProtocol
from covenant_persistence.testing import InMemoryConnection, InMemoryStore


def _create_connection() -> tuple[InMemoryConnection, InMemoryStore]:
    """Create in-memory connection with store."""
    store = InMemoryStore()
    conn = InMemoryConnection(store)
    return conn, store


class TestPostgresDealRepository:
    """Tests for PostgresDealRepository."""

    def test_create_and_get(self) -> None:
        """Create and retrieve a deal."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn
        repo = PostgresDealRepository(typed_conn)

        deal: Deal = {
            "id": {"value": "deal-123"},
            "name": "Test Deal",
            "borrower": "Acme Corp",
            "sector": "Technology",
            "region": "North America",
            "commitment_amount_cents": 100_000_000,
            "currency": "USD",
            "maturity_date_iso": "2025-12-31",
        }

        repo.create(deal)
        retrieved = repo.get({"value": "deal-123"})

        assert retrieved["id"]["value"] == "deal-123"
        assert retrieved["name"] == "Test Deal"
        assert retrieved["borrower"] == "Acme Corp"

    def test_get_not_found_raises(self) -> None:
        """Get non-existent deal raises KeyError."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn
        repo = PostgresDealRepository(typed_conn)

        with pytest.raises(KeyError) as exc_info:
            repo.get({"value": "missing"})
        assert "Deal not found" in str(exc_info.value)

    def test_list_all(self) -> None:
        """List all deals."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn
        repo = PostgresDealRepository(typed_conn)

        deal1: Deal = {
            "id": {"value": "deal-1"},
            "name": "Deal 1",
            "borrower": "Corp A",
            "sector": "Tech",
            "region": "NA",
            "commitment_amount_cents": 1000,
            "currency": "USD",
            "maturity_date_iso": "2025-01-01",
        }
        deal2: Deal = {
            "id": {"value": "deal-2"},
            "name": "Deal 2",
            "borrower": "Corp B",
            "sector": "Finance",
            "region": "EU",
            "commitment_amount_cents": 2000,
            "currency": "EUR",
            "maturity_date_iso": "2026-01-01",
        }

        repo.create(deal1)
        repo.create(deal2)
        all_deals = repo.list_all()

        assert len(all_deals) == 2

    def test_update(self) -> None:
        """Update an existing deal."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn
        repo = PostgresDealRepository(typed_conn)

        deal: Deal = {
            "id": {"value": "deal-123"},
            "name": "Original",
            "borrower": "Acme",
            "sector": "Tech",
            "region": "NA",
            "commitment_amount_cents": 1000,
            "currency": "USD",
            "maturity_date_iso": "2025-01-01",
        }
        repo.create(deal)

        updated: Deal = {
            "id": {"value": "deal-123"},
            "name": "Updated",
            "borrower": "Acme Corp",
            "sector": "Technology",
            "region": "North America",
            "commitment_amount_cents": 2000,
            "currency": "USD",
            "maturity_date_iso": "2026-01-01",
        }
        repo.update(updated)
        retrieved = repo.get({"value": "deal-123"})

        assert retrieved["name"] == "Updated"

    def test_update_not_found_raises(self) -> None:
        """Update non-existent deal raises KeyError."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn
        repo = PostgresDealRepository(typed_conn)

        deal: Deal = {
            "id": {"value": "missing"},
            "name": "Test",
            "borrower": "Corp",
            "sector": "Tech",
            "region": "NA",
            "commitment_amount_cents": 1000,
            "currency": "USD",
            "maturity_date_iso": "2025-01-01",
        }
        with pytest.raises(KeyError) as exc_info:
            repo.update(deal)
        assert "Deal not found" in str(exc_info.value)

    def test_delete(self) -> None:
        """Delete a deal."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn
        repo = PostgresDealRepository(typed_conn)

        deal: Deal = {
            "id": {"value": "deal-123"},
            "name": "Test",
            "borrower": "Corp",
            "sector": "Tech",
            "region": "NA",
            "commitment_amount_cents": 1000,
            "currency": "USD",
            "maturity_date_iso": "2025-01-01",
        }
        repo.create(deal)
        repo.delete({"value": "deal-123"})

        with pytest.raises(KeyError):
            repo.get({"value": "deal-123"})

    def test_delete_not_found_raises(self) -> None:
        """Delete non-existent deal raises KeyError."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn
        repo = PostgresDealRepository(typed_conn)

        with pytest.raises(KeyError) as exc_info:
            repo.delete({"value": "missing"})
        assert "Deal not found" in str(exc_info.value)


class TestPostgresCovenantRepository:
    """Tests for PostgresCovenantRepository."""

    def test_create_and_get(self) -> None:
        """Create and retrieve a covenant."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn
        repo = PostgresCovenantRepository(typed_conn)

        covenant: Covenant = {
            "id": {"value": "cov-123"},
            "deal_id": {"value": "deal-1"},
            "name": "Debt to EBITDA",
            "formula": "total_debt / ebitda",
            "threshold_value_scaled": 3_500_000,
            "threshold_direction": "<=",
            "frequency": "QUARTERLY",
        }

        repo.create(covenant)
        retrieved = repo.get({"value": "cov-123"})

        assert retrieved["id"]["value"] == "cov-123"
        assert retrieved["name"] == "Debt to EBITDA"
        assert retrieved["threshold_direction"] == "<="
        assert retrieved["frequency"] == "QUARTERLY"

    def test_get_not_found_raises(self) -> None:
        """Get non-existent covenant raises KeyError."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn
        repo = PostgresCovenantRepository(typed_conn)

        with pytest.raises(KeyError) as exc_info:
            repo.get({"value": "missing"})
        assert "Covenant not found" in str(exc_info.value)

    def test_list_for_deal(self) -> None:
        """List covenants for a deal."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn
        repo = PostgresCovenantRepository(typed_conn)

        cov1: Covenant = {
            "id": {"value": "cov-1"},
            "deal_id": {"value": "deal-1"},
            "name": "Cov 1",
            "formula": "a / b",
            "threshold_value_scaled": 1000,
            "threshold_direction": "<=",
            "frequency": "QUARTERLY",
        }
        cov2: Covenant = {
            "id": {"value": "cov-2"},
            "deal_id": {"value": "deal-1"},
            "name": "Cov 2",
            "formula": "c / d",
            "threshold_value_scaled": 2000,
            "threshold_direction": ">=",
            "frequency": "ANNUAL",
        }

        repo.create(cov1)
        repo.create(cov2)
        covenants = repo.list_for_deal({"value": "deal-1"})

        assert len(covenants) == 2

    def test_delete(self) -> None:
        """Delete a covenant."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn
        repo = PostgresCovenantRepository(typed_conn)

        covenant: Covenant = {
            "id": {"value": "cov-123"},
            "deal_id": {"value": "deal-1"},
            "name": "Test",
            "formula": "a / b",
            "threshold_value_scaled": 1000,
            "threshold_direction": "<=",
            "frequency": "QUARTERLY",
        }
        repo.create(covenant)
        repo.delete({"value": "cov-123"})

        with pytest.raises(KeyError):
            repo.get({"value": "cov-123"})

    def test_delete_not_found_raises(self) -> None:
        """Delete non-existent covenant raises KeyError."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn
        repo = PostgresCovenantRepository(typed_conn)

        with pytest.raises(KeyError) as exc_info:
            repo.delete({"value": "missing"})
        assert "Covenant not found" in str(exc_info.value)


class TestPostgresMeasurementRepository:
    """Tests for PostgresMeasurementRepository."""

    def test_add_many(self) -> None:
        """Add multiple measurements."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn
        repo = PostgresMeasurementRepository(typed_conn)

        measurements: list[Measurement] = [
            {
                "deal_id": {"value": "deal-1"},
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "metric_name": "total_debt",
                "metric_value_scaled": 100_000_000,
            },
            {
                "deal_id": {"value": "deal-1"},
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "metric_name": "ebitda",
                "metric_value_scaled": 50_000_000,
            },
        ]

        count = repo.add_many(measurements)
        assert count == 2

    def test_add_many_empty(self) -> None:
        """Add empty list returns zero."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn
        repo = PostgresMeasurementRepository(typed_conn)

        count = repo.add_many([])
        assert count == 0

    def test_list_for_deal_and_period(self) -> None:
        """List measurements for a specific deal and period."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn
        repo = PostgresMeasurementRepository(typed_conn)

        m1: Measurement = {
            "deal_id": {"value": "deal-1"},
            "period_start_iso": "2024-01-01",
            "period_end_iso": "2024-03-31",
            "metric_name": "total_debt",
            "metric_value_scaled": 100_000_000,
        }
        m2: Measurement = {
            "deal_id": {"value": "deal-1"},
            "period_start_iso": "2024-04-01",
            "period_end_iso": "2024-06-30",
            "metric_name": "total_debt",
            "metric_value_scaled": 110_000_000,
        }

        repo.add_many([m1, m2])
        results = repo.list_for_deal_and_period({"value": "deal-1"}, "2024-01-01", "2024-03-31")

        assert len(results) == 1
        assert results[0]["metric_value_scaled"] == 100_000_000

    def test_list_for_deal(self) -> None:
        """List all measurements for a deal."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn
        repo = PostgresMeasurementRepository(typed_conn)

        m1: Measurement = {
            "deal_id": {"value": "deal-1"},
            "period_start_iso": "2024-01-01",
            "period_end_iso": "2024-03-31",
            "metric_name": "total_debt",
            "metric_value_scaled": 100_000_000,
        }
        m2: Measurement = {
            "deal_id": {"value": "deal-1"},
            "period_start_iso": "2024-04-01",
            "period_end_iso": "2024-06-30",
            "metric_name": "total_debt",
            "metric_value_scaled": 110_000_000,
        }

        repo.add_many([m1, m2])
        results = repo.list_for_deal({"value": "deal-1"})

        assert len(results) == 2


class TestPostgresCovenantResultRepository:
    """Tests for PostgresCovenantResultRepository."""

    def test_save(self) -> None:
        """Save a covenant result."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn
        repo = PostgresCovenantResultRepository(typed_conn)

        result: CovenantResult = {
            "covenant_id": {"value": "cov-1"},
            "period_start_iso": "2024-01-01",
            "period_end_iso": "2024-03-31",
            "calculated_value_scaled": 2_500_000,
            "status": "OK",
        }

        repo.save(result)
        results = repo.list_for_covenant({"value": "cov-1"})

        assert len(results) == 1
        assert results[0]["status"] == "OK"

    def test_save_upsert(self) -> None:
        """Save updates existing result for same covenant/period."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn
        repo = PostgresCovenantResultRepository(typed_conn)

        result1: CovenantResult = {
            "covenant_id": {"value": "cov-1"},
            "period_start_iso": "2024-01-01",
            "period_end_iso": "2024-03-31",
            "calculated_value_scaled": 2_500_000,
            "status": "OK",
        }
        result2: CovenantResult = {
            "covenant_id": {"value": "cov-1"},
            "period_start_iso": "2024-01-01",
            "period_end_iso": "2024-03-31",
            "calculated_value_scaled": 3_500_000,
            "status": "BREACH",
        }

        repo.save(result1)
        repo.save(result2)
        results = repo.list_for_covenant({"value": "cov-1"})

        assert len(results) == 1
        assert results[0]["status"] == "BREACH"

    def test_save_many(self) -> None:
        """Save multiple results."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn
        repo = PostgresCovenantResultRepository(typed_conn)

        results_to_save: list[CovenantResult] = [
            {
                "covenant_id": {"value": "cov-1"},
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "calculated_value_scaled": 2_500_000,
                "status": "OK",
            },
            {
                "covenant_id": {"value": "cov-1"},
                "period_start_iso": "2024-04-01",
                "period_end_iso": "2024-06-30",
                "calculated_value_scaled": 3_000_000,
                "status": "NEAR_BREACH",
            },
        ]

        count = repo.save_many(results_to_save)
        assert count == 2

    def test_save_many_empty(self) -> None:
        """Save empty list returns zero."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn
        repo = PostgresCovenantResultRepository(typed_conn)

        count = repo.save_many([])
        assert count == 0

    def test_list_for_covenant(self) -> None:
        """List results for a specific covenant."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn
        repo = PostgresCovenantResultRepository(typed_conn)

        r1: CovenantResult = {
            "covenant_id": {"value": "cov-1"},
            "period_start_iso": "2024-01-01",
            "period_end_iso": "2024-03-31",
            "calculated_value_scaled": 2_500_000,
            "status": "OK",
        }
        r2: CovenantResult = {
            "covenant_id": {"value": "cov-2"},
            "period_start_iso": "2024-01-01",
            "period_end_iso": "2024-03-31",
            "calculated_value_scaled": 1_500_000,
            "status": "BREACH",
        }

        repo.save(r1)
        repo.save(r2)
        results = repo.list_for_covenant({"value": "cov-1"})

        assert len(results) == 1
        assert results[0]["covenant_id"]["value"] == "cov-1"

    def test_list_for_deal(self) -> None:
        """List results for a deal's covenants."""
        conn, _ = _create_connection()
        typed_conn: ConnectionProtocol = conn

        # First create covenants
        cov_repo = PostgresCovenantRepository(typed_conn)
        cov: Covenant = {
            "id": {"value": "cov-1"},
            "deal_id": {"value": "deal-1"},
            "name": "Test",
            "formula": "a / b",
            "threshold_value_scaled": 1000,
            "threshold_direction": "<=",
            "frequency": "QUARTERLY",
        }
        cov_repo.create(cov)

        result_repo = PostgresCovenantResultRepository(typed_conn)
        result: CovenantResult = {
            "covenant_id": {"value": "cov-1"},
            "period_start_iso": "2024-01-01",
            "period_end_iso": "2024-03-31",
            "calculated_value_scaled": 2_500_000,
            "status": "OK",
        }
        result_repo.save(result)

        results = result_repo.list_for_deal({"value": "deal-1"})
        assert len(results) == 1
