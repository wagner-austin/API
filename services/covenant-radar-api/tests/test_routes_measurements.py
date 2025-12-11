"""Integration tests for measurements CRUD routes."""

from __future__ import annotations

from covenant_domain import DealId, Measurement
from fastapi import FastAPI
from fastapi.testclient import TestClient
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_list,
)

from covenant_radar_api.api.routes.measurements import build_router

from .conftest import ContainerAndStore


def _create_test_client(cas: ContainerAndStore) -> TestClient:
    """Create test client with real container."""
    app = FastAPI()
    router = build_router(cas.container)
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


def _make_measurement(
    deal_id: str,
    metric_name: str,
    metric_value: int,
    period_start: str = "2024-01-01",
    period_end: str = "2024-03-31",
) -> Measurement:
    """Create a test measurement."""
    return Measurement(
        deal_id=DealId(value=deal_id),
        period_start_iso=period_start,
        period_end_iso=period_end,
        metric_name=metric_name,
        metric_value_scaled=metric_value,
    )


class TestListMeasurementsForDeal:
    """Tests for GET /measurements/by-deal/{deal_id}."""

    def test_empty_list(self, container_with_store: ContainerAndStore) -> None:
        """Test listing measurements when none exist for deal."""
        client = _create_test_client(container_with_store)

        response = client.get("/measurements/by-deal/deal-123")

        assert response.status_code == 200
        data = load_json_str(response.text)
        assert data == []

    def test_list_with_measurements(self, container_with_store: ContainerAndStore) -> None:
        """Test listing measurements when some exist for deal."""
        store = container_with_store.store
        store.measurements.append(_make_measurement("deal-123", "total_debt", 5000000000))
        store.measurements.append(_make_measurement("deal-123", "ebitda", 1500000000))
        store.measurements.append(_make_measurement("other-deal", "revenue", 2000000000))
        client = _create_test_client(container_with_store)

        response = client.get("/measurements/by-deal/deal-123")

        assert response.status_code == 200
        data = narrow_json_to_list(load_json_str(response.text))
        assert len(data) == 2


class TestListMeasurementsForDealAndPeriod:
    """Tests for GET /measurements/by-deal/{deal_id}/period."""

    def test_empty_list(self, container_with_store: ContainerAndStore) -> None:
        """Test listing measurements when none exist for period."""
        client = _create_test_client(container_with_store)

        response = client.get(
            "/measurements/by-deal/deal-123/period",
            params={"period_start": "2024-01-01", "period_end": "2024-03-31"},
        )

        assert response.status_code == 200
        data = load_json_str(response.text)
        assert data == []

    def test_list_with_matching_period(self, container_with_store: ContainerAndStore) -> None:
        """Test listing measurements for matching period."""
        store = container_with_store.store
        store.measurements.append(
            _make_measurement("deal-123", "total_debt", 5000000000, "2024-01-01", "2024-03-31")
        )
        store.measurements.append(
            _make_measurement("deal-123", "ebitda", 1500000000, "2024-01-01", "2024-03-31")
        )
        store.measurements.append(
            _make_measurement("deal-123", "revenue", 2000000000, "2024-04-01", "2024-06-30")
        )
        client = _create_test_client(container_with_store)

        response = client.get(
            "/measurements/by-deal/deal-123/period",
            params={"period_start": "2024-01-01", "period_end": "2024-03-31"},
        )

        assert response.status_code == 200
        data = narrow_json_to_list(load_json_str(response.text))
        assert len(data) == 2


class TestAddMeasurements:
    """Tests for POST /measurements."""

    def test_add_measurements_success(self, container_with_store: ContainerAndStore) -> None:
        """Test adding new measurements."""
        client = _create_test_client(container_with_store)

        response = client.post(
            "/measurements",
            content=b"""{
                "measurements": [
                    {
                        "deal_id": {"value": "deal-123"},
                        "period_start_iso": "2024-01-01",
                        "period_end_iso": "2024-03-31",
                        "metric_name": "total_debt",
                        "metric_value_scaled": 5000000000
                    },
                    {
                        "deal_id": {"value": "deal-123"},
                        "period_start_iso": "2024-01-01",
                        "period_end_iso": "2024-03-31",
                        "metric_name": "ebitda",
                        "metric_value_scaled": 1500000000
                    }
                ]
            }""",
        )

        assert response.status_code == 201
        assert len(container_with_store.store.measurements) == 2
        data = narrow_json_to_dict(load_json_str(response.text))
        assert data["count"] == 2

    def test_add_empty_measurements(self, container_with_store: ContainerAndStore) -> None:
        """Test adding empty measurements list."""
        client = _create_test_client(container_with_store)

        response = client.post("/measurements", content=b"""{"measurements": []}""")

        assert response.status_code == 201
        assert len(container_with_store.store.measurements) == 0
        data = narrow_json_to_dict(load_json_str(response.text))
        assert data["count"] == 0

    def test_add_measurements_invalid_json(self, container_with_store: ContainerAndStore) -> None:
        """Test adding measurements with invalid JSON."""
        client = _create_test_client(container_with_store)

        response = client.post("/measurements", content=b"not valid json")

        assert response.status_code == 500
