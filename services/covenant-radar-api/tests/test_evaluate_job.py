"""Integration tests for batch evaluation job with real implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pytest
from covenant_domain import (
    Covenant,
    CovenantId,
    CovenantResult,
    DealId,
    Measurement,
)
from covenant_persistence import (
    CovenantRepository,
    CovenantResultRepository,
    MeasurementRepository,
)
from platform_core.json_utils import JSONTypeError, dump_json_str

from covenant_radar_api.worker.evaluate_job import run_batch_evaluation


class _InMemoryStore:
    """In-memory storage for evaluation test data."""

    def __init__(self) -> None:
        self.covenants: dict[str, Covenant] = {}
        self.measurements: list[Measurement] = []
        self.results: list[CovenantResult] = []


class _InMemoryCovenantRepository:
    """In-memory implementation of CovenantRepository."""

    def __init__(self, store: _InMemoryStore) -> None:
        self._store = store

    def create(self, covenant: Covenant) -> None:
        """Insert new covenant."""
        cov_id = covenant["id"]["value"]
        self._store.covenants[cov_id] = covenant

    def get(self, covenant_id: CovenantId) -> Covenant:
        """Get covenant by ID."""
        key = covenant_id["value"]
        if key not in self._store.covenants:
            raise KeyError(f"Covenant not found: {key}")
        return self._store.covenants[key]

    def list_for_deal(self, deal_id: DealId) -> Sequence[Covenant]:
        """List covenants for a deal."""
        result: list[Covenant] = []
        for cov in self._store.covenants.values():
            if cov["deal_id"]["value"] == deal_id["value"]:
                result.append(cov)
        return result

    def delete(self, covenant_id: CovenantId) -> None:
        """Delete covenant."""
        key = covenant_id["value"]
        del self._store.covenants[key]


class _InMemoryMeasurementRepository:
    """In-memory implementation of MeasurementRepository."""

    def __init__(self, store: _InMemoryStore) -> None:
        self._store = store

    def add_many(self, measurements: Sequence[Measurement]) -> int:
        """Insert measurements."""
        for m in measurements:
            self._store.measurements.append(m)
        return len(measurements)

    def list_for_deal(self, deal_id: DealId) -> Sequence[Measurement]:
        """List all measurements for a deal."""
        result: list[Measurement] = []
        for m in self._store.measurements:
            if m["deal_id"]["value"] == deal_id["value"]:
                result.append(m)
        return result

    def list_for_deal_and_period(
        self,
        deal_id: DealId,
        period_start_iso: str,
        period_end_iso: str,
    ) -> Sequence[Measurement]:
        """List measurements for deal and period."""
        result: list[Measurement] = []
        for m in self._store.measurements:
            if (
                m["deal_id"]["value"] == deal_id["value"]
                and m["period_start_iso"] == period_start_iso
                and m["period_end_iso"] == period_end_iso
            ):
                result.append(m)
        return result


class _InMemoryCovenantResultRepository:
    """In-memory implementation of CovenantResultRepository."""

    def __init__(self, store: _InMemoryStore) -> None:
        self._store = store

    def save(self, result: CovenantResult) -> None:
        """Save result."""
        self._store.results.append(result)

    def save_many(self, results: Sequence[CovenantResult]) -> int:
        """Save multiple results."""
        for r in results:
            self._store.results.append(r)
        return len(results)

    def list_for_deal(self, deal_id: DealId) -> Sequence[CovenantResult]:
        """List results for a deal's covenants."""
        cov_ids: set[str] = set()
        for cov in self._store.covenants.values():
            if cov["deal_id"]["value"] == deal_id["value"]:
                cov_ids.add(cov["id"]["value"])
        result: list[CovenantResult] = []
        for r in self._store.results:
            if r["covenant_id"]["value"] in cov_ids:
                result.append(r)
        return result

    def list_for_covenant(self, covenant_id: CovenantId) -> Sequence[CovenantResult]:
        """List results for a covenant."""
        result: list[CovenantResult] = []
        for r in self._store.results:
            if r["covenant_id"]["value"] == covenant_id["value"]:
                result.append(r)
        return result


class _TestRepoProvider:
    """Test repository provider for batch evaluation."""

    def __init__(self, store: _InMemoryStore) -> None:
        self._store = store

    def covenant_repo(self) -> CovenantRepository:
        """Return in-memory covenant repository."""
        repo: CovenantRepository = _InMemoryCovenantRepository(self._store)
        return repo

    def measurement_repo(self) -> MeasurementRepository:
        """Return in-memory measurement repository."""
        repo: MeasurementRepository = _InMemoryMeasurementRepository(self._store)
        return repo

    def covenant_result_repo(self) -> CovenantResultRepository:
        """Return in-memory result repository."""
        repo: CovenantResultRepository = _InMemoryCovenantResultRepository(self._store)
        return repo


def _add_covenant(
    store: _InMemoryStore,
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


def _add_measurement(
    store: _InMemoryStore,
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
        store = _InMemoryStore()
        _add_covenant(store, "c1", "d1", "debt / ebitda", 4_000_000, "<=")
        _add_measurement(store, "d1", "2024-01-01", "2024-03-31", "debt", 10_000_000)
        _add_measurement(store, "d1", "2024-01-01", "2024-03-31", "ebitda", 5_000_000)

        provider = _TestRepoProvider(store)
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
        assert isinstance(result["results"], list)
        assert len(result["results"]) == 1

        # Verify result was saved
        assert len(store.results) == 1
        assert store.results[0]["status"] == "OK"

    def test_evaluate_multiple_deals(self) -> None:
        """Test batch evaluation with multiple deals."""
        store = _InMemoryStore()

        # Deal 1 - will pass
        _add_covenant(store, "c1", "d1", "debt / ebitda", 4_000_000, "<=")
        _add_measurement(store, "d1", "2024-01-01", "2024-03-31", "debt", 10_000_000)
        _add_measurement(store, "d1", "2024-01-01", "2024-03-31", "ebitda", 5_000_000)

        # Deal 2 - will breach
        _add_covenant(store, "c2", "d2", "debt / ebitda", 3_000_000, "<=")
        _add_measurement(store, "d2", "2024-01-01", "2024-03-31", "debt", 20_000_000)
        _add_measurement(store, "d2", "2024-01-01", "2024-03-31", "ebitda", 5_000_000)

        provider = _TestRepoProvider(store)
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
        assert len(store.results) == 2

        # Check both results were saved with correct statuses
        statuses = {r["status"] for r in store.results}
        assert "OK" in statuses
        assert "BREACH" in statuses

    def test_evaluate_deal_with_multiple_covenants(self) -> None:
        """Test batch evaluation with one deal having multiple covenants."""
        store = _InMemoryStore()
        _add_covenant(store, "c1", "d1", "debt / ebitda", 4_000_000, "<=")
        _add_covenant(store, "c2", "d1", "ebitda / interest", 2_000_000, ">=")
        _add_measurement(store, "d1", "2024-01-01", "2024-03-31", "debt", 10_000_000)
        _add_measurement(store, "d1", "2024-01-01", "2024-03-31", "ebitda", 5_000_000)
        _add_measurement(store, "d1", "2024-01-01", "2024-03-31", "interest", 1_000_000)

        provider = _TestRepoProvider(store)
        deal_ids_json = dump_json_str(["d1"])

        result = run_batch_evaluation(
            deal_ids_json=deal_ids_json,
            period_start_iso="2024-01-01",
            period_end_iso="2024-03-31",
            tolerance_ratio_scaled=100_000,
            repo_provider=provider,
        )

        assert result["results_count"] == 2
        assert len(store.results) == 2

    def test_evaluate_empty_deal_list(self) -> None:
        """Test batch evaluation with empty deal list."""
        store = _InMemoryStore()
        provider = _TestRepoProvider(store)
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
        assert len(store.results) == 0

    def test_evaluate_deal_with_no_covenants(self) -> None:
        """Test batch evaluation with deal that has no covenants."""
        store = _InMemoryStore()
        # No covenants for d1
        _add_measurement(store, "d1", "2024-01-01", "2024-03-31", "debt", 10_000_000)

        provider = _TestRepoProvider(store)
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
        store = _InMemoryStore()
        provider = _TestRepoProvider(store)
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
        store = _InMemoryStore()
        _add_covenant(store, "c1", "d1", "debt / ebitda", 4_000_000, "<=")
        _add_measurement(store, "d1", "2024-01-01", "2024-03-31", "debt", 10_000_000)
        _add_measurement(store, "d1", "2024-01-01", "2024-03-31", "ebitda", 5_000_000)

        provider = _TestRepoProvider(store)
        deal_ids_json = dump_json_str(["d1"])

        result = run_batch_evaluation(
            deal_ids_json=deal_ids_json,
            period_start_iso="2024-01-01",
            period_end_iso="2024-03-31",
            tolerance_ratio_scaled=100_000,
            repo_provider=provider,
        )

        assert isinstance(result["results"], list)
        encoded_result = result["results"][0]
        assert isinstance(encoded_result, dict)
        # Check encoded structure has expected keys
        assert "covenant_id" in encoded_result
        assert "period_start_iso" in encoded_result
        assert "status" in encoded_result
        assert "calculated_value_scaled" in encoded_result
