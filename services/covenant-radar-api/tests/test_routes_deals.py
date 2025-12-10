"""Integration tests for deals CRUD routes."""

from __future__ import annotations

from collections.abc import Sequence

from covenant_domain import Deal, DealId
from covenant_persistence import DealRepository
from fastapi.testclient import TestClient
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_list,
)

from covenant_radar_api.api.routes.deals import build_router


class _InMemoryDealStore:
    """In-memory storage for deals."""

    def __init__(self) -> None:
        self.deals: dict[str, Deal] = {}


class _InMemoryDealRepository:
    """In-memory implementation of DealRepository for testing."""

    def __init__(self, store: _InMemoryDealStore) -> None:
        self._store = store

    def create(self, deal: Deal) -> None:
        """Insert new deal. Raises on duplicate ID."""
        deal_id = deal["id"]["value"]
        if deal_id in self._store.deals:
            raise ValueError(f"Deal already exists: {deal_id}")
        self._store.deals[deal_id] = deal

    def get(self, deal_id: DealId) -> Deal:
        """Get deal by ID. Raises KeyError if not found."""
        key = deal_id["value"]
        if key not in self._store.deals:
            raise KeyError(f"Deal not found: {key}")
        return self._store.deals[key]

    def list_all(self) -> Sequence[Deal]:
        """List all deals."""
        return list(self._store.deals.values())

    def update(self, deal: Deal) -> None:
        """Update existing deal. Raises KeyError if not found."""
        key = deal["id"]["value"]
        if key not in self._store.deals:
            raise KeyError(f"Deal not found: {key}")
        self._store.deals[key] = deal

    def delete(self, deal_id: DealId) -> None:
        """Delete deal. Raises KeyError if not found."""
        key = deal_id["value"]
        if key not in self._store.deals:
            raise KeyError(f"Deal not found: {key}")
        del self._store.deals[key]


class _TestContainer:
    """Test container with in-memory deal repository."""

    def __init__(self, store: _InMemoryDealStore) -> None:
        self._store = store

    def deal_repo(self) -> DealRepository:
        """Return in-memory deal repository."""
        repo: DealRepository = _InMemoryDealRepository(self._store)
        return repo


def _create_test_client(store: _InMemoryDealStore) -> TestClient:
    """Create test client with in-memory repository."""
    from fastapi import FastAPI

    app = FastAPI()
    container = _TestContainer(store)
    router = build_router(container)
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


class TestListDeals:
    """Tests for GET /deals."""

    def test_empty_list(self) -> None:
        """Test listing deals when none exist."""
        store = _InMemoryDealStore()
        client = _create_test_client(store)

        response = client.get("/deals")

        assert response.status_code == 200
        data = load_json_str(response.text)
        assert data == []

    def test_list_with_deals(self) -> None:
        """Test listing deals when some exist."""
        store = _InMemoryDealStore()
        store.deals["d1"] = Deal(
            id=DealId(value="d1"),
            name="Deal One",
            borrower="Borrower A",
            sector="Tech",
            region="NA",
            commitment_amount_cents=1000000,
            currency="USD",
            maturity_date_iso="2025-12-31",
        )
        client = _create_test_client(store)

        response = client.get("/deals")

        assert response.status_code == 200
        data = narrow_json_to_list(load_json_str(response.text))
        assert len(data) == 1
        first_deal = narrow_json_to_dict(data[0])
        assert first_deal["name"] == "Deal One"


class TestCreateDeal:
    """Tests for POST /deals."""

    def test_create_deal_success(self) -> None:
        """Test creating a new deal."""
        store = _InMemoryDealStore()
        client = _create_test_client(store)

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
        assert "new-deal" in store.deals
        data = narrow_json_to_dict(load_json_str(response.text))
        assert data["name"] == "New Deal"

    def test_create_deal_invalid_json(self) -> None:
        """Test creating deal with invalid JSON."""
        store = _InMemoryDealStore()
        client = _create_test_client(store)

        response = client.post("/deals", content=b"not valid json")

        assert response.status_code == 500


class TestGetDeal:
    """Tests for GET /deals/{deal_id}."""

    def test_get_existing_deal(self) -> None:
        """Test getting an existing deal."""
        store = _InMemoryDealStore()
        store.deals["d1"] = Deal(
            id=DealId(value="d1"),
            name="Test Deal",
            borrower="Test Corp",
            sector="Tech",
            region="NA",
            commitment_amount_cents=1000000,
            currency="USD",
            maturity_date_iso="2025-12-31",
        )
        client = _create_test_client(store)

        response = client.get("/deals/d1")

        assert response.status_code == 200
        data = narrow_json_to_dict(load_json_str(response.text))
        assert data["name"] == "Test Deal"

    def test_get_nonexistent_deal(self) -> None:
        """Test getting a deal that doesn't exist."""
        store = _InMemoryDealStore()
        client = _create_test_client(store)

        response = client.get("/deals/nonexistent")

        assert response.status_code == 500


class TestUpdateDeal:
    """Tests for PUT /deals/{deal_id}."""

    def test_update_existing_deal(self) -> None:
        """Test updating an existing deal."""
        store = _InMemoryDealStore()
        store.deals["d1"] = Deal(
            id=DealId(value="d1"),
            name="Old Name",
            borrower="Old Corp",
            sector="Tech",
            region="NA",
            commitment_amount_cents=1000000,
            currency="USD",
            maturity_date_iso="2025-12-31",
        )
        client = _create_test_client(store)

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
        assert store.deals["d1"]["name"] == "Updated Name"
        data = narrow_json_to_dict(load_json_str(response.text))
        assert data["name"] == "Updated Name"

    def test_update_nonexistent_deal(self) -> None:
        """Test updating a deal that doesn't exist."""
        store = _InMemoryDealStore()
        client = _create_test_client(store)

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

    def test_delete_existing_deal(self) -> None:
        """Test deleting an existing deal."""
        store = _InMemoryDealStore()
        store.deals["d1"] = Deal(
            id=DealId(value="d1"),
            name="To Delete",
            borrower="Corp",
            sector="Tech",
            region="NA",
            commitment_amount_cents=1000000,
            currency="USD",
            maturity_date_iso="2025-12-31",
        )
        client = _create_test_client(store)

        response = client.delete("/deals/d1")

        assert response.status_code == 204
        assert "d1" not in store.deals

    def test_delete_nonexistent_deal(self) -> None:
        """Test deleting a deal that doesn't exist."""
        store = _InMemoryDealStore()
        client = _create_test_client(store)

        response = client.delete("/deals/nonexistent")

        assert response.status_code == 500
