"""Integration tests for deals CRUD routes."""

from __future__ import annotations

from covenant_domain import Deal, DealId
from fastapi import FastAPI
from fastapi.testclient import TestClient
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_list,
)

from covenant_radar_api.api.routes.deals import build_router

from .conftest import ContainerAndStore


def _create_test_client(cas: ContainerAndStore) -> TestClient:
    """Create test client with real container."""
    app = FastAPI()
    router = build_router(cas.container)
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


class TestListDeals:
    """Tests for GET /deals."""

    def test_empty_list(self, container_with_store: ContainerAndStore) -> None:
        """Test listing deals when none exist."""
        client = _create_test_client(container_with_store)

        response = client.get("/deals")

        assert response.status_code == 200
        data = load_json_str(response.text)
        assert data == []

    def test_list_with_deals(self, container_with_store: ContainerAndStore) -> None:
        """Test listing deals when some exist."""
        container_with_store.store.deals["d1"] = Deal(
            id=DealId(value="d1"),
            name="Deal One",
            borrower="Borrower A",
            sector="Tech",
            region="NA",
            commitment_amount_cents=1000000,
            currency="USD",
            maturity_date_iso="2025-12-31",
        )
        container_with_store.store._deal_order.append("d1")
        client = _create_test_client(container_with_store)

        response = client.get("/deals")

        assert response.status_code == 200
        data = narrow_json_to_list(load_json_str(response.text))
        assert len(data) == 1
        first_deal = narrow_json_to_dict(data[0])
        assert first_deal["name"] == "Deal One"


class TestCreateDeal:
    """Tests for POST /deals."""

    def test_create_deal_success(self, container_with_store: ContainerAndStore) -> None:
        """Test creating a new deal."""
        client = _create_test_client(container_with_store)

        response = client.post(
            "/deals",
            content=b"""{
                "id": {"value": "new-deal"},
                "name": "New Deal",
                "borrower": "Corp",
                "sector": "Finance",
                "region": "EU",
                "commitment_amount_cents": 500000,
                "currency": "EUR",
                "maturity_date_iso": "2026-01-01"
            }""",
        )

        assert response.status_code == 201
        assert "new-deal" in container_with_store.store.deals
        data = narrow_json_to_dict(load_json_str(response.text))
        assert data["name"] == "New Deal"

    def test_create_deal_invalid_json(self, container_with_store: ContainerAndStore) -> None:
        """Test creating deal with invalid JSON."""
        client = _create_test_client(container_with_store)

        response = client.post("/deals", content=b"not valid json")

        assert response.status_code == 500


class TestGetDeal:
    """Tests for GET /deals/{deal_id}."""

    def test_get_existing_deal(self, container_with_store: ContainerAndStore) -> None:
        """Test getting an existing deal."""
        container_with_store.store.deals["d1"] = Deal(
            id=DealId(value="d1"),
            name="Test Deal",
            borrower="Test Corp",
            sector="Tech",
            region="NA",
            commitment_amount_cents=1000000,
            currency="USD",
            maturity_date_iso="2025-12-31",
        )
        client = _create_test_client(container_with_store)

        response = client.get("/deals/d1")

        assert response.status_code == 200
        data = narrow_json_to_dict(load_json_str(response.text))
        assert data["name"] == "Test Deal"

    def test_get_nonexistent_deal(self, container_with_store: ContainerAndStore) -> None:
        """Test getting a deal that doesn't exist."""
        client = _create_test_client(container_with_store)

        response = client.get("/deals/nonexistent")

        assert response.status_code == 500


class TestUpdateDeal:
    """Tests for PUT /deals/{deal_id}."""

    def test_update_existing_deal(self, container_with_store: ContainerAndStore) -> None:
        """Test updating an existing deal."""
        container_with_store.store.deals["d1"] = Deal(
            id=DealId(value="d1"),
            name="Old Name",
            borrower="Old Corp",
            sector="Tech",
            region="NA",
            commitment_amount_cents=1000000,
            currency="USD",
            maturity_date_iso="2025-12-31",
        )
        client = _create_test_client(container_with_store)

        response = client.put(
            "/deals/d1",
            content=b"""{
                "name": "Updated Name",
                "borrower": "New Corp",
                "sector": "Finance",
                "region": "EU",
                "commitment_amount_cents": 2000000,
                "currency": "EUR",
                "maturity_date_iso": "2026-06-30"
            }""",
        )

        assert response.status_code == 200
        assert container_with_store.store.deals["d1"]["name"] == "Updated Name"
        data = narrow_json_to_dict(load_json_str(response.text))
        assert data["name"] == "Updated Name"

    def test_update_nonexistent_deal(self, container_with_store: ContainerAndStore) -> None:
        """Test updating a deal that doesn't exist."""
        client = _create_test_client(container_with_store)

        response = client.put(
            "/deals/nonexistent",
            content=b"""{
                "name": "Name",
                "borrower": "Corp",
                "sector": "Tech",
                "region": "NA",
                "commitment_amount_cents": 1000,
                "currency": "USD",
                "maturity_date_iso": "2025-01-01"
            }""",
        )

        assert response.status_code == 500


class TestDeleteDeal:
    """Tests for DELETE /deals/{deal_id}."""

    def test_delete_existing_deal(self, container_with_store: ContainerAndStore) -> None:
        """Test deleting an existing deal."""
        container_with_store.store.deals["d1"] = Deal(
            id=DealId(value="d1"),
            name="To Delete",
            borrower="Corp",
            sector="Tech",
            region="NA",
            commitment_amount_cents=1000000,
            currency="USD",
            maturity_date_iso="2025-12-31",
        )
        client = _create_test_client(container_with_store)

        response = client.delete("/deals/d1")

        assert response.status_code == 204
        assert "d1" not in container_with_store.store.deals

    def test_delete_nonexistent_deal(self, container_with_store: ContainerAndStore) -> None:
        """Test deleting a deal that doesn't exist."""
        client = _create_test_client(container_with_store)

        response = client.delete("/deals/nonexistent")

        assert response.status_code == 500
