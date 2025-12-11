"""Integration tests for batch evaluation job with real implementations."""

from __future__ import annotations

from typing import Literal

import pytest
from covenant_domain import (
    Covenant,
    CovenantId,
    DealId,
    Measurement,
)
from covenant_persistence import (
    CovenantRepository,
    CovenantResultRepository,
    MeasurementRepository,
    PostgresCovenantRepository,
    PostgresCovenantResultRepository,
    PostgresMeasurementRepository,
)
from covenant_persistence.testing import InMemoryConnection, InMemoryStore
from platform_core.json_utils import (
    JSONTypeError,
    dump_json_str,
    narrow_json_to_dict,
    narrow_json_to_list,
    require_int,
    require_str,
)

from covenant_radar_api.worker.evaluate_job import run_batch_evaluation


class _RepoProvider:
    """Repository provider using InMemoryConnection for batch evaluation tests."""

    def __init__(self, store: InMemoryStore) -> None:
        self._conn = InMemoryConnection(store)

    def covenant_repo(self) -> CovenantRepository:
        """Return covenant repository."""
        repo: CovenantRepository = PostgresCovenantRepository(self._conn)
        return repo

    def measurement_repo(self) -> MeasurementRepository:
        """Return measurement repository."""
        repo: MeasurementRepository = PostgresMeasurementRepository(self._conn)
        return repo

    def covenant_result_repo(self) -> CovenantResultRepository:
        """Return result repository."""
        repo: CovenantResultRepository = PostgresCovenantResultRepository(self._conn)
        return repo


def _add_covenant(
    store: InMemoryStore,
    cov_id: str,
    deal_id: str,
    formula: str,
    threshold: int,
    direction: Literal["<=", ">="],
) -> None:
    """Add a covenant to store."""
    store.covenants[cov_id] = Covenant(
        id=CovenantId(value=cov_id),
        deal_id=DealId(value=deal_id),
        name="Test Covenant",
        formula=formula,
        threshold_value_scaled=threshold,
        threshold_direction=direction,
        frequency="QUARTERLY",
    )
    store._covenant_order.append(cov_id)


def _add_measurement(
    store: InMemoryStore,
    deal_id: str,
    period_start: str,
    period_end: str,
    metric_name: str,
    value: int,
) -> None:
    """Add a measurement to store."""
    store.measurements.append(
        Measurement(
            deal_id=DealId(value=deal_id),
            period_start_iso=period_start,
            period_end_iso=period_end,
            metric_name=metric_name,
            metric_value_scaled=value,
        )
    )


class TestRunBatchEvaluation:
    """Tests for run_batch_evaluation job function."""

    def test_evaluate_single_deal_single_covenant(self) -> None:
        """Test batch evaluation with one deal and one covenant."""
        store = InMemoryStore()
        _add_covenant(store, "c1", "d1", "debt / ebitda", 4_000_000, "<=")
        _add_measurement(store, "d1", "2024-01-01", "2024-03-31", "debt", 10_000_000)
        _add_measurement(store, "d1", "2024-01-01", "2024-03-31", "ebitda", 5_000_000)

        provider = _RepoProvider(store)
        deal_ids_json = dump_json_str(["d1"])

        result = run_batch_evaluation(
            deal_ids_json=deal_ids_json,
            period_start_iso="2024-01-01",
            period_end_iso="2024-03-31",
            tolerance_ratio_scaled=100_000,
            repo_provider=provider,
        )

        assert result["status"] == "complete"
        assert result["deals_evaluated"] == 1
        assert result["results_count"] == 1
        results_list = narrow_json_to_list(result["results"])
        assert len(results_list) == 1
        first_result = narrow_json_to_dict(results_list[0])
        covenant_id = narrow_json_to_dict(first_result["covenant_id"])
        assert require_str(covenant_id, "value") == "c1"

        # Verify result was saved
        assert len(store.covenant_results) == 1
        assert store.covenant_results[0]["status"] == "OK"

    def test_evaluate_multiple_deals(self) -> None:
        """Test batch evaluation with multiple deals."""
        store = InMemoryStore()

        # Deal 1 - will pass
        _add_covenant(store, "c1", "d1", "debt / ebitda", 4_000_000, "<=")
        _add_measurement(store, "d1", "2024-01-01", "2024-03-31", "debt", 10_000_000)
        _add_measurement(store, "d1", "2024-01-01", "2024-03-31", "ebitda", 5_000_000)

        # Deal 2 - will breach
        _add_covenant(store, "c2", "d2", "debt / ebitda", 3_000_000, "<=")
        _add_measurement(store, "d2", "2024-01-01", "2024-03-31", "debt", 20_000_000)
        _add_measurement(store, "d2", "2024-01-01", "2024-03-31", "ebitda", 5_000_000)

        provider = _RepoProvider(store)
        deal_ids_json = dump_json_str(["d1", "d2"])

        result = run_batch_evaluation(
            deal_ids_json=deal_ids_json,
            period_start_iso="2024-01-01",
            period_end_iso="2024-03-31",
            tolerance_ratio_scaled=100_000,
            repo_provider=provider,
        )

        assert result["status"] == "complete"
        assert result["deals_evaluated"] == 2
        assert result["results_count"] == 2
        assert len(store.covenant_results) == 2

        # Check both results were saved with correct statuses
        statuses = {r["status"] for r in store.covenant_results}
        assert "OK" in statuses
        assert "BREACH" in statuses

    def test_evaluate_deal_with_multiple_covenants(self) -> None:
        """Test batch evaluation with one deal having multiple covenants."""
        store = InMemoryStore()
        _add_covenant(store, "c1", "d1", "debt / ebitda", 4_000_000, "<=")
        _add_covenant(store, "c2", "d1", "ebitda / interest", 2_000_000, ">=")
        _add_measurement(store, "d1", "2024-01-01", "2024-03-31", "debt", 10_000_000)
        _add_measurement(store, "d1", "2024-01-01", "2024-03-31", "ebitda", 5_000_000)
        _add_measurement(store, "d1", "2024-01-01", "2024-03-31", "interest", 1_000_000)

        provider = _RepoProvider(store)
        deal_ids_json = dump_json_str(["d1"])

        result = run_batch_evaluation(
            deal_ids_json=deal_ids_json,
            period_start_iso="2024-01-01",
            period_end_iso="2024-03-31",
            tolerance_ratio_scaled=100_000,
            repo_provider=provider,
        )

        assert result["results_count"] == 2
        assert len(store.covenant_results) == 2

    def test_evaluate_empty_deal_list(self) -> None:
        """Test batch evaluation with empty deal list."""
        store = InMemoryStore()
        provider = _RepoProvider(store)
        deal_ids_json = dump_json_str([])

        result = run_batch_evaluation(
            deal_ids_json=deal_ids_json,
            period_start_iso="2024-01-01",
            period_end_iso="2024-03-31",
            tolerance_ratio_scaled=100_000,
            repo_provider=provider,
        )

        assert result["status"] == "complete"
        assert result["deals_evaluated"] == 0
        assert result["results_count"] == 0
        assert len(store.covenant_results) == 0

    def test_evaluate_deal_with_no_covenants(self) -> None:
        """Test batch evaluation with deal that has no covenants."""
        store = InMemoryStore()
        # No covenants for d1
        _add_measurement(store, "d1", "2024-01-01", "2024-03-31", "debt", 10_000_000)

        provider = _RepoProvider(store)
        deal_ids_json = dump_json_str(["d1"])

        result = run_batch_evaluation(
            deal_ids_json=deal_ids_json,
            period_start_iso="2024-01-01",
            period_end_iso="2024-03-31",
            tolerance_ratio_scaled=100_000,
            repo_provider=provider,
        )

        assert result["deals_evaluated"] == 1
        assert result["results_count"] == 0

    def test_evaluate_invalid_deal_id_type_raises(self) -> None:
        """Test that non-string deal IDs raise JSONTypeError."""
        store = InMemoryStore()
        provider = _RepoProvider(store)
        # JSON with integer instead of string
        deal_ids_json = "[123]"

        with pytest.raises(JSONTypeError, match="must be a string"):
            run_batch_evaluation(
                deal_ids_json=deal_ids_json,
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                tolerance_ratio_scaled=100_000,
                repo_provider=provider,
            )

    def test_evaluate_results_contain_encoded_data(self) -> None:
        """Test that results contain properly encoded covenant result data."""
        store = InMemoryStore()
        _add_covenant(store, "c1", "d1", "debt / ebitda", 4_000_000, "<=")
        _add_measurement(store, "d1", "2024-01-01", "2024-03-31", "debt", 10_000_000)
        _add_measurement(store, "d1", "2024-01-01", "2024-03-31", "ebitda", 5_000_000)

        provider = _RepoProvider(store)
        deal_ids_json = dump_json_str(["d1"])

        result = run_batch_evaluation(
            deal_ids_json=deal_ids_json,
            period_start_iso="2024-01-01",
            period_end_iso="2024-03-31",
            tolerance_ratio_scaled=100_000,
            repo_provider=provider,
        )

        results_list = narrow_json_to_list(result["results"])
        assert len(results_list) == 1
        encoded_result = narrow_json_to_dict(results_list[0])
        # Check encoded structure has expected values
        covenant_id = narrow_json_to_dict(encoded_result["covenant_id"])
        assert require_str(covenant_id, "value") == "c1"
        assert require_str(encoded_result, "period_start_iso") == "2024-01-01"
        assert require_str(encoded_result, "status") == "OK"
        assert require_int(encoded_result, "calculated_value_scaled") == 2_000_000
