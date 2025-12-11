"""Integration tests for covenants CRUD routes."""

from __future__ import annotations

from typing import Literal

from covenant_domain import Covenant, CovenantId, DealId
from fastapi import FastAPI
from fastapi.testclient import TestClient
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_list,
)

from covenant_radar_api.api.routes.covenants import build_router

from .conftest import ContainerAndStore


def _create_test_client(cas: ContainerAndStore) -> TestClient:
    """Create test client with real container."""
    app = FastAPI()
    router = build_router(cas.container)
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


def _make_covenant(
    cov_id: str,
    deal_id: str,
    name: str = "Test Covenant",
    direction: Literal["<=", ">="] = "<=",
    frequency: Literal["QUARTERLY", "ANNUAL"] = "QUARTERLY",
) -> Covenant:
    """Create a test covenant."""
    return Covenant(
        id=CovenantId(value=cov_id),
        deal_id=DealId(value=deal_id),
        name=name,
        formula="total_debt / ebitda",
        threshold_value_scaled=3500000,
        threshold_direction=direction,
        frequency=frequency,
    )


class TestListCovenantsForDeal:
    """Tests for GET /covenants/by-deal/{deal_id}."""

    def test_empty_list(self, container_with_store: ContainerAndStore) -> None:
        """Test listing covenants when none exist for deal."""
        client = _create_test_client(container_with_store)

        response = client.get("/covenants/by-deal/deal-123")

        assert response.status_code == 200
        data = load_json_str(response.text)
        assert data == []

    def test_list_with_covenants(self, container_with_store: ContainerAndStore) -> None:
        """Test listing covenants when some exist for deal."""
        store = container_with_store.store
        store.covenants["c1"] = _make_covenant("c1", "deal-123", "Covenant One")
        store.covenants["c2"] = _make_covenant("c2", "deal-123", "Covenant Two")
        store.covenants["c3"] = _make_covenant("c3", "other-deal", "Other Covenant")
        store._covenant_order.extend(["c1", "c2", "c3"])
        client = _create_test_client(container_with_store)

        response = client.get("/covenants/by-deal/deal-123")

        assert response.status_code == 200
        data = narrow_json_to_list(load_json_str(response.text))
        assert len(data) == 2


class TestCreateCovenant:
    """Tests for POST /covenants."""

    def test_create_covenant_success(self, container_with_store: ContainerAndStore) -> None:
        """Test creating a new covenant."""
        client = _create_test_client(container_with_store)

        response = client.post(
            "/covenants",
            content=b"""{
                "id": {"value": "new-cov"},
                "deal_id": {"value": "deal-123"},
                "name": "Debt to EBITDA",
                "formula": "total_debt / ebitda",
                "threshold_value_scaled": 3500000,
                "threshold_direction": "<=",
                "frequency": "QUARTERLY"
            }""",
        )

        assert response.status_code == 201
        assert "new-cov" in container_with_store.store.covenants
        data = narrow_json_to_dict(load_json_str(response.text))
        assert data["name"] == "Debt to EBITDA"

    def test_create_covenant_gte_direction(self, container_with_store: ContainerAndStore) -> None:
        """Test creating a covenant with >= direction."""
        client = _create_test_client(container_with_store)

        response = client.post(
            "/covenants",
            content=b"""{
                "id": {"value": "cov-gte"},
                "deal_id": {"value": "deal-123"},
                "name": "Interest Coverage",
                "formula": "ebitda / interest",
                "threshold_value_scaled": 2000000,
                "threshold_direction": ">=",
                "frequency": "ANNUAL"
            }""",
        )

        assert response.status_code == 201
        store = container_with_store.store
        assert store.covenants["cov-gte"]["threshold_direction"] == ">="
        assert store.covenants["cov-gte"]["frequency"] == "ANNUAL"


class TestGetCovenant:
    """Tests for GET /covenants/{covenant_id}."""

    def test_get_existing_covenant(self, container_with_store: ContainerAndStore) -> None:
        """Test getting an existing covenant."""
        store = container_with_store.store
        store.covenants["c1"] = _make_covenant("c1", "deal-123", "Test Covenant")
        client = _create_test_client(container_with_store)

        response = client.get("/covenants/c1")

        assert response.status_code == 200
        data = narrow_json_to_dict(load_json_str(response.text))
        assert data["name"] == "Test Covenant"

    def test_get_nonexistent_covenant(self, container_with_store: ContainerAndStore) -> None:
        """Test getting a covenant that doesn't exist."""
        client = _create_test_client(container_with_store)

        response = client.get("/covenants/nonexistent")

        assert response.status_code == 500


class TestDeleteCovenant:
    """Tests for DELETE /covenants/{covenant_id}."""

    def test_delete_existing_covenant(self, container_with_store: ContainerAndStore) -> None:
        """Test deleting an existing covenant."""
        store = container_with_store.store
        store.covenants["c1"] = _make_covenant("c1", "deal-123")
        client = _create_test_client(container_with_store)

        response = client.delete("/covenants/c1")

        assert response.status_code == 204
        assert "c1" not in store.covenants

    def test_delete_nonexistent_covenant(self, container_with_store: ContainerAndStore) -> None:
        """Test deleting a covenant that doesn't exist."""
        client = _create_test_client(container_with_store)

        response = client.delete("/covenants/nonexistent")

        assert response.status_code == 500
