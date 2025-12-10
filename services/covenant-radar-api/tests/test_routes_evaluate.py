"""Integration tests for evaluate routes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from covenant_domain import Covenant, CovenantId, CovenantResult, DealId, Measurement
from covenant_persistence import CovenantRepository, CovenantResultRepository, MeasurementRepository
from fastapi import FastAPI
from fastapi.testclient import TestClient
from platform_core.json_utils import load_json_str, narrow_json_to_list

from covenant_radar_api.api.routes.evaluate import build_router


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
        if cov_id in self._store.covenants:
            raise ValueError(f"Covenant already exists: {cov_id}")
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
        if key not in self._store.covenants:
            raise KeyError(f"Covenant not found: {key}")
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

    def list_for_deal(self, deal_id: DealId) -> Sequence[Measurement]:
        """List all measurements for a deal."""
        result: list[Measurement] = []
        for m in self._store.measurements:
            if m["deal_id"]["value"] == deal_id["value"]:
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
        # Get covenant IDs for this deal
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


class _TestContainer:
    """Test container for evaluate routes."""

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


def _create_test_client(store: _InMemoryStore) -> TestClient:
    """Create test client with in-memory repositories."""
    app = FastAPI()
    container = _TestContainer(store)
    router = build_router(container)
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


def _add_test_covenant(
    store: _InMemoryStore,
    cov_id: str,
    deal_id: str,
    formula: str,
    threshold: int,
    direction: Literal["<=", ">="],
) -> None:
    """Add a test covenant to store."""
    store.covenants[cov_id] = Covenant(
        id=CovenantId(value=cov_id),
        deal_id=DealId(value=deal_id),
        name="Test Covenant",
        formula=formula,
        threshold_value_scaled=threshold,
        threshold_direction=direction,
        frequency="QUARTERLY",
    )


def _add_test_measurement(
    store: _InMemoryStore,
    deal_id: str,
    period_start: str,
    period_end: str,
    metric_name: str,
    value: int,
) -> None:
    """Add a test measurement to store."""
    store.measurements.append(
        Measurement(
            deal_id=DealId(value=deal_id),
            period_start_iso=period_start,
            period_end_iso=period_end,
            metric_name=metric_name,
            metric_value_scaled=value,
        )
    )


class TestEvaluateEndpoint:
    """Tests for POST /evaluate."""

    def test_evaluate_single_covenant_ok(self) -> None:
        """Test evaluation with single covenant that passes."""
        store = _InMemoryStore()
        _add_test_covenant(store, "c1", "d1", "debt / ebitda", 4_000_000, "<=")
        _add_test_measurement(store, "d1", "2024-01-01", "2024-03-31", "debt", 10_000_000)
        _add_test_measurement(store, "d1", "2024-01-01", "2024-03-31", "ebitda", 5_000_000)

        client = _create_test_client(store)
        response = client.post(
            "/evaluate",
            content=b"""{
                "deal_id": "d1",
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "tolerance_ratio_scaled": 100000
            }""",
        )

        assert response.status_code == 200
        data = narrow_json_to_list(load_json_str(response.text))
        assert len(data) == 1
        result = data[0]
        assert isinstance(result, dict)
        assert result["status"] == "OK"
        # 10M / 5M = 2.0, threshold is 4.0, so OK
        assert result["calculated_value_scaled"] == 2_000_000

    def test_evaluate_single_covenant_breach(self) -> None:
        """Test evaluation with single covenant that breaches."""
        store = _InMemoryStore()
        _add_test_covenant(store, "c1", "d1", "debt / ebitda", 3_000_000, "<=")
        _add_test_measurement(store, "d1", "2024-01-01", "2024-03-31", "debt", 20_000_000)
        _add_test_measurement(store, "d1", "2024-01-01", "2024-03-31", "ebitda", 5_000_000)

        client = _create_test_client(store)
        response = client.post(
            "/evaluate",
            content=b"""{
                "deal_id": "d1",
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "tolerance_ratio_scaled": 100000
            }""",
        )

        assert response.status_code == 200
        data = narrow_json_to_list(load_json_str(response.text))
        assert len(data) == 1
        result = data[0]
        assert isinstance(result, dict)
        assert result["status"] == "BREACH"
        # 20M / 5M = 4.0 > 3.0 threshold, so BREACH

    def test_evaluate_near_breach(self) -> None:
        """Test evaluation with near-breach status."""
        store = _InMemoryStore()
        # Threshold is 3.0, tolerance is 10%, so near-breach band is 2.7-3.0
        _add_test_covenant(store, "c1", "d1", "debt / ebitda", 3_000_000, "<=")
        # 14M / 5M = 2.8, which is in near-breach band
        _add_test_measurement(store, "d1", "2024-01-01", "2024-03-31", "debt", 14_000_000)
        _add_test_measurement(store, "d1", "2024-01-01", "2024-03-31", "ebitda", 5_000_000)

        client = _create_test_client(store)
        response = client.post(
            "/evaluate",
            content=b"""{
                "deal_id": "d1",
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "tolerance_ratio_scaled": 100000
            }""",
        )

        assert response.status_code == 200
        data = narrow_json_to_list(load_json_str(response.text))
        assert len(data) == 1
        result = data[0]
        assert isinstance(result, dict)
        assert result["status"] == "NEAR_BREACH"

    def test_evaluate_multiple_covenants(self) -> None:
        """Test evaluation with multiple covenants."""
        store = _InMemoryStore()
        _add_test_covenant(store, "c1", "d1", "debt / ebitda", 4_000_000, "<=")
        _add_test_covenant(store, "c2", "d1", "ebitda / interest", 2_000_000, ">=")
        _add_test_measurement(store, "d1", "2024-01-01", "2024-03-31", "debt", 10_000_000)
        _add_test_measurement(store, "d1", "2024-01-01", "2024-03-31", "ebitda", 5_000_000)
        _add_test_measurement(store, "d1", "2024-01-01", "2024-03-31", "interest", 1_000_000)

        client = _create_test_client(store)
        response = client.post(
            "/evaluate",
            content=b"""{
                "deal_id": "d1",
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "tolerance_ratio_scaled": 100000
            }""",
        )

        assert response.status_code == 200
        data = narrow_json_to_list(load_json_str(response.text))
        assert len(data) == 2

    def test_evaluate_no_covenants(self) -> None:
        """Test evaluation when deal has no covenants."""
        store = _InMemoryStore()
        client = _create_test_client(store)

        response = client.post(
            "/evaluate",
            content=b"""{
                "deal_id": "d1",
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "tolerance_ratio_scaled": 100000
            }""",
        )

        assert response.status_code == 200
        data = narrow_json_to_list(load_json_str(response.text))
        assert len(data) == 0

    def test_evaluate_missing_metric(self) -> None:
        """Test evaluation when required metric is missing."""
        store = _InMemoryStore()
        _add_test_covenant(store, "c1", "d1", "debt / ebitda", 4_000_000, "<=")
        _add_test_measurement(store, "d1", "2024-01-01", "2024-03-31", "debt", 10_000_000)
        # Missing ebitda measurement

        client = _create_test_client(store)
        response = client.post(
            "/evaluate",
            content=b"""{
                "deal_id": "d1",
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "tolerance_ratio_scaled": 100000
            }""",
        )

        # Should fail with KeyError (metric not found)
        assert response.status_code == 500

    def test_evaluate_invalid_json(self) -> None:
        """Test evaluation with invalid JSON request."""
        store = _InMemoryStore()
        client = _create_test_client(store)

        response = client.post("/evaluate", content=b"not valid json")

        assert response.status_code == 500

    def test_evaluate_missing_field(self) -> None:
        """Test evaluation with missing required field."""
        store = _InMemoryStore()
        client = _create_test_client(store)

        response = client.post(
            "/evaluate",
            content=b"""{
                "deal_id": "d1",
                "period_start_iso": "2024-01-01"
            }""",
        )

        # Missing period_end_iso and tolerance_ratio_scaled
        assert response.status_code == 500

    def test_evaluate_results_saved(self) -> None:
        """Test that evaluation results are saved to repository."""
        store = _InMemoryStore()
        _add_test_covenant(store, "c1", "d1", "debt / ebitda", 4_000_000, "<=")
        _add_test_measurement(store, "d1", "2024-01-01", "2024-03-31", "debt", 10_000_000)
        _add_test_measurement(store, "d1", "2024-01-01", "2024-03-31", "ebitda", 5_000_000)

        client = _create_test_client(store)
        client.post(
            "/evaluate",
            content=b"""{
                "deal_id": "d1",
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "tolerance_ratio_scaled": 100000
            }""",
        )

        # Verify results were saved
        assert len(store.results) == 1
        assert store.results[0]["covenant_id"]["value"] == "c1"
        assert store.results[0]["status"] == "OK"

    def test_evaluate_greater_than_direction(self) -> None:
        """Test evaluation with >= threshold direction."""
        store = _InMemoryStore()
        # Interest coverage must be >= 2.0
        _add_test_covenant(store, "c1", "d1", "ebitda / interest", 2_000_000, ">=")
        _add_test_measurement(store, "d1", "2024-01-01", "2024-03-31", "ebitda", 6_000_000)
        _add_test_measurement(store, "d1", "2024-01-01", "2024-03-31", "interest", 2_000_000)

        client = _create_test_client(store)
        response = client.post(
            "/evaluate",
            content=b"""{
                "deal_id": "d1",
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "tolerance_ratio_scaled": 100000
            }""",
        )

        assert response.status_code == 200
        data = narrow_json_to_list(load_json_str(response.text))
        assert len(data) == 1
        result = data[0]
        assert isinstance(result, dict)
        # 6M / 2M = 3.0 >= 2.0 threshold, so OK
        assert result["status"] == "OK"
        assert result["calculated_value_scaled"] == 3_000_000

    def test_evaluate_greater_than_breach(self) -> None:
        """Test evaluation with >= threshold direction breach."""
        store = _InMemoryStore()
        # Interest coverage must be >= 2.0
        _add_test_covenant(store, "c1", "d1", "ebitda / interest", 2_000_000, ">=")
        _add_test_measurement(store, "d1", "2024-01-01", "2024-03-31", "ebitda", 3_000_000)
        _add_test_measurement(store, "d1", "2024-01-01", "2024-03-31", "interest", 2_000_000)

        client = _create_test_client(store)
        response = client.post(
            "/evaluate",
            content=b"""{
                "deal_id": "d1",
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "tolerance_ratio_scaled": 100000
            }""",
        )

        assert response.status_code == 200
        data = narrow_json_to_list(load_json_str(response.text))
        result = data[0]
        assert isinstance(result, dict)
        # 3M / 2M = 1.5 < 2.0 threshold, so BREACH
        assert result["status"] == "BREACH"
