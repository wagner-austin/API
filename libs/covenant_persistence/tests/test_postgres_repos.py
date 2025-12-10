"""Tests for PostgreSQL repository implementations."""

from __future__ import annotations

from collections.abc import Sequence

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
from covenant_persistence.protocols import ConnectionProtocol, CursorProtocol


class InMemoryCursor:
    """In-memory cursor that simulates database behavior."""

    def __init__(self, store: dict[str, list[tuple[str | int | bool | None, ...]]]) -> None:
        """Initialize cursor with shared store."""
        self._store = store
        self._rows: list[tuple[str | int | bool | None, ...]] = []
        self._rowcount = 0
        self._last_query = ""
        self._last_params: tuple[str | int | bool | None, ...] = ()

    def execute(self, query: str, params: tuple[str | int | bool | None, ...] = ()) -> None:
        """Execute query and simulate database behavior."""
        self._last_query = query
        self._last_params = params
        self._rows = []
        self._rowcount = 0
        query_lower = query.strip().lower()
        self._dispatch_query(query_lower, params)

    def _dispatch_query(
        self, query_lower: str, params: tuple[str | int | bool | None, ...]
    ) -> None:
        """Dispatch query to appropriate handler."""
        if self._dispatch_deals(query_lower, params):
            return
        if self._dispatch_covenants(query_lower, params):
            return
        if self._dispatch_measurements(query_lower, params):
            return
        self._dispatch_results(query_lower, params)

    def _dispatch_deals(
        self, query_lower: str, params: tuple[str | int | bool | None, ...]
    ) -> bool:
        """Dispatch deal-related queries. Returns True if handled."""
        if query_lower.startswith("insert into deals"):
            self._handle_deal_insert(params)
            return True
        if query_lower.startswith("select") and "from deals" in query_lower:
            self._handle_deal_select(params, query_lower)
            return True
        if query_lower.startswith("update deals"):
            self._handle_deal_update(params)
            return True
        if query_lower.startswith("delete from deals"):
            self._handle_deal_delete(params)
            return True
        return False

    def _dispatch_covenants(
        self, query_lower: str, params: tuple[str | int | bool | None, ...]
    ) -> bool:
        """Dispatch covenant-related queries. Returns True if handled."""
        if query_lower.startswith("insert into covenants"):
            self._handle_covenant_insert(params)
            return True
        if query_lower.startswith("select") and "from covenants" in query_lower:
            self._handle_covenant_select(params, query_lower)
            return True
        if query_lower.startswith("delete from covenants"):
            self._handle_covenant_delete(params)
            return True
        return False

    def _dispatch_measurements(
        self, query_lower: str, params: tuple[str | int | bool | None, ...]
    ) -> bool:
        """Dispatch measurement-related queries. Returns True if handled."""
        if query_lower.startswith("insert into measurements"):
            self._handle_measurement_insert(params)
            return True
        if query_lower.startswith("select") and "from measurements" in query_lower:
            self._handle_measurement_select(params, query_lower)
            return True
        return False

    def _dispatch_results(
        self, query_lower: str, params: tuple[str | int | bool | None, ...]
    ) -> None:
        """Dispatch result-related queries."""
        if query_lower.startswith("insert into covenant_results"):
            self._handle_result_insert(params)
        elif query_lower.startswith("select") and "from covenant_results" in query_lower:
            self._handle_result_select(params, query_lower)

    def _handle_deal_insert(self, params: tuple[str | int | bool | None, ...]) -> None:
        if "deals" not in self._store:
            self._store["deals"] = []
        self._store["deals"].append(params)
        self._rowcount = 1

    def _handle_deal_select(self, params: tuple[str | int | bool | None, ...], query: str) -> None:
        if "deals" not in self._store:
            self._rows = []
            return
        if "where id" in query and params:
            deal_id = params[0]
            self._rows = [d for d in self._store["deals"] if d[0] == deal_id]
        else:
            self._rows = list(self._store["deals"])

    def _handle_deal_update(self, params: tuple[str | int | bool | None, ...]) -> None:
        if "deals" not in self._store:
            self._rowcount = 0
            return
        deal_id = params[-1]  # ID is last param in UPDATE
        for i, d in enumerate(self._store["deals"]):
            if d[0] == deal_id:
                # Update: (name, borrower, sector, region, commitment, currency, maturity, id)
                self._store["deals"][i] = (
                    deal_id,
                    params[0],
                    params[1],
                    params[2],
                    params[3],
                    params[4],
                    params[5],
                    params[6],
                )
                self._rowcount = 1
                return
        self._rowcount = 0

    def _handle_deal_delete(self, params: tuple[str | int | bool | None, ...]) -> None:
        if "deals" not in self._store:
            self._rowcount = 0
            return
        deal_id = params[0]
        original_len = len(self._store["deals"])
        self._store["deals"] = [d for d in self._store["deals"] if d[0] != deal_id]
        self._rowcount = original_len - len(self._store["deals"])

    def _handle_covenant_insert(self, params: tuple[str | int | bool | None, ...]) -> None:
        if "covenants" not in self._store:
            self._store["covenants"] = []
        self._store["covenants"].append(params)
        self._rowcount = 1

    def _handle_covenant_select(
        self, params: tuple[str | int | bool | None, ...], query: str
    ) -> None:
        if "covenants" not in self._store:
            self._rows = []
            return
        if "where id" in query and params:
            cov_id = params[0]
            self._rows = [c for c in self._store["covenants"] if c[0] == cov_id]
        elif "where deal_id" in query and params:
            deal_id = params[0]
            self._rows = [c for c in self._store["covenants"] if c[1] == deal_id]
        else:
            self._rows = list(self._store["covenants"])

    def _handle_covenant_delete(self, params: tuple[str | int | bool | None, ...]) -> None:
        if "covenants" not in self._store:
            self._rowcount = 0
            return
        cov_id = params[0]
        original_len = len(self._store["covenants"])
        self._store["covenants"] = [c for c in self._store["covenants"] if c[0] != cov_id]
        self._rowcount = original_len - len(self._store["covenants"])

    def _handle_measurement_insert(self, params: tuple[str | int | bool | None, ...]) -> None:
        if "measurements" not in self._store:
            self._store["measurements"] = []
        self._store["measurements"].append(params)
        self._rowcount = 1

    def _handle_measurement_select(
        self, params: tuple[str | int | bool | None, ...], query: str
    ) -> None:
        if "measurements" not in self._store:
            self._rows = []
            return
        if "period_start" in query and len(params) >= 3:
            deal_id, period_start, period_end = params[0], params[1], params[2]
            self._rows = [
                m
                for m in self._store["measurements"]
                if m[0] == deal_id and m[1] == period_start and m[2] == period_end
            ]
        elif "where deal_id" in query and params:
            deal_id = params[0]
            self._rows = [m for m in self._store["measurements"] if m[0] == deal_id]
        else:
            self._rows = list(self._store["measurements"])

    def _handle_result_insert(self, params: tuple[str | int | bool | None, ...]) -> None:
        if "results" not in self._store:
            self._store["results"] = []
        # Check for upsert (ON CONFLICT)
        cov_id, period_start, period_end = params[0], params[1], params[2]
        for i, r in enumerate(self._store["results"]):
            if r[0] == cov_id and r[1] == period_start and r[2] == period_end:
                self._store["results"][i] = params
                self._rowcount = 1
                return
        self._store["results"].append(params)
        self._rowcount = 1

    def _handle_result_select(
        self, params: tuple[str | int | bool | None, ...], query: str
    ) -> None:
        if "results" not in self._store:
            self._rows = []
            return
        if "where covenant_id" in query and params:
            cov_id = params[0]
            self._rows = [r for r in self._store["results"] if r[0] == cov_id]
        elif "join covenants" in query and params:
            # list_for_deal joins with covenants table
            deal_id = params[0]
            if "covenants" in self._store:
                cov_ids = {c[0] for c in self._store["covenants"] if c[1] == deal_id}
                self._rows = [r for r in self._store["results"] if r[0] in cov_ids]
            else:
                self._rows = []
        else:
            self._rows = list(self._store["results"])

    def fetchone(self) -> tuple[str | int | bool | None, ...] | None:
        """Fetch one row or None."""
        if self._rows:
            return self._rows[0]
        return None

    def fetchall(self) -> Sequence[tuple[str | int | bool | None, ...]]:
        """Fetch all rows."""
        return self._rows

    @property
    def rowcount(self) -> int:
        """Return number of affected rows."""
        return self._rowcount


class InMemoryConnection:
    """In-memory connection that simulates database behavior."""

    def __init__(self) -> None:
        """Initialize with empty store."""
        self._store: dict[str, list[tuple[str | int | bool | None, ...]]] = {}
        self._cursor = InMemoryCursor(self._store)
        self._committed = False
        self._rolled_back = False
        self._closed = False

    def cursor(self) -> CursorProtocol:
        """Return cursor."""
        return self._cursor

    def commit(self) -> None:
        """Mark as committed."""
        self._committed = True

    def rollback(self) -> None:
        """Mark as rolled back."""
        self._rolled_back = True

    def close(self) -> None:
        """Mark as closed."""
        self._closed = True


def _create_connection() -> InMemoryConnection:
    """Create in-memory connection."""
    return InMemoryConnection()


class TestPostgresDealRepository:
    """Tests for PostgresDealRepository."""

    def test_create_and_get(self) -> None:
        """Create and retrieve a deal."""
        typed_conn: ConnectionProtocol = _create_connection()
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
        typed_conn: ConnectionProtocol = _create_connection()
        repo = PostgresDealRepository(typed_conn)

        with pytest.raises(KeyError) as exc_info:
            repo.get({"value": "missing"})
        assert "Deal not found" in str(exc_info.value)

    def test_list_all(self) -> None:
        """List all deals."""
        typed_conn: ConnectionProtocol = _create_connection()
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
        typed_conn: ConnectionProtocol = _create_connection()
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
        typed_conn: ConnectionProtocol = _create_connection()
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
        typed_conn: ConnectionProtocol = _create_connection()
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
        typed_conn: ConnectionProtocol = _create_connection()
        repo = PostgresDealRepository(typed_conn)

        with pytest.raises(KeyError) as exc_info:
            repo.delete({"value": "missing"})
        assert "Deal not found" in str(exc_info.value)


class TestPostgresCovenantRepository:
    """Tests for PostgresCovenantRepository."""

    def test_create_and_get(self) -> None:
        """Create and retrieve a covenant."""
        typed_conn: ConnectionProtocol = _create_connection()
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
        typed_conn: ConnectionProtocol = _create_connection()
        repo = PostgresCovenantRepository(typed_conn)

        with pytest.raises(KeyError) as exc_info:
            repo.get({"value": "missing"})
        assert "Covenant not found" in str(exc_info.value)

    def test_list_for_deal(self) -> None:
        """List covenants for a deal."""
        typed_conn: ConnectionProtocol = _create_connection()
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
        typed_conn: ConnectionProtocol = _create_connection()
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
        typed_conn: ConnectionProtocol = _create_connection()
        repo = PostgresCovenantRepository(typed_conn)

        with pytest.raises(KeyError) as exc_info:
            repo.delete({"value": "missing"})
        assert "Covenant not found" in str(exc_info.value)


class TestPostgresMeasurementRepository:
    """Tests for PostgresMeasurementRepository."""

    def test_add_many(self) -> None:
        """Add multiple measurements."""
        typed_conn: ConnectionProtocol = _create_connection()
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
        typed_conn: ConnectionProtocol = _create_connection()
        repo = PostgresMeasurementRepository(typed_conn)

        count = repo.add_many([])
        assert count == 0

    def test_list_for_deal_and_period(self) -> None:
        """List measurements for a specific deal and period."""
        typed_conn: ConnectionProtocol = _create_connection()
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
        typed_conn: ConnectionProtocol = _create_connection()
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
        typed_conn: ConnectionProtocol = _create_connection()
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
        typed_conn: ConnectionProtocol = _create_connection()
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
        typed_conn: ConnectionProtocol = _create_connection()
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
        typed_conn: ConnectionProtocol = _create_connection()
        repo = PostgresCovenantResultRepository(typed_conn)

        count = repo.save_many([])
        assert count == 0

    def test_list_for_covenant(self) -> None:
        """List results for a specific covenant."""
        typed_conn: ConnectionProtocol = _create_connection()
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
        typed_conn: ConnectionProtocol = _create_connection()

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
