"""Integration tests for covenants CRUD routes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from covenant_domain import Covenant, CovenantId, DealId
from covenant_persistence import CovenantRepository
from fastapi.testclient import TestClient
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_list,
)

from covenant_radar_api.api.routes.covenants import build_router


class _InMemoryCovenantStore:
    """In-memory storage for covenants."""

    def __init__(self) -> None:
        self.covenants: dict[str, Covenant] = {}


class _InMemoryCovenantRepository:
    """In-memory implementation of CovenantRepository for testing."""

    def __init__(self, store: _InMemoryCovenantStore) -> None:
        self._store = store

    def create(self, covenant: Covenant) -> None:
        """Insert new covenant. Raises on duplicate ID."""
        cov_id = covenant["id"]["value"]
        if cov_id in self._store.covenants:
            raise ValueError(f"Covenant already exists: {cov_id}")
        self._store.covenants[cov_id] = covenant

    def get(self, covenant_id: CovenantId) -> Covenant:
        """Get covenant by ID. Raises KeyError if not found."""
        key = covenant_id["value"]
        if key not in self._store.covenants:
            raise KeyError(f"Covenant not found: {key}")
        return self._store.covenants[key]

    def list_for_deal(self, deal_id: DealId) -> Sequence[Covenant]:
        """List all covenants for a deal."""
        deal_key = deal_id["value"]
        return [c for c in self._store.covenants.values() if c["deal_id"]["value"] == deal_key]

    def delete(self, covenant_id: CovenantId) -> None:
        """Delete covenant. Raises KeyError if not found."""
        key = covenant_id["value"]
        if key not in self._store.covenants:
            raise KeyError(f"Covenant not found: {key}")
        del self._store.covenants[key]


class _TestContainer:
    """Test container with in-memory covenant repository."""

    def __init__(self, store: _InMemoryCovenantStore) -> None:
        self._store = store

    def covenant_repo(self) -> CovenantRepository:
        """Return in-memory covenant repository."""
        repo: CovenantRepository = _InMemoryCovenantRepository(self._store)
        return repo


def _create_test_client(store: _InMemoryCovenantStore) -> TestClient:
    """Create test client with in-memory repository."""
    from fastapi import FastAPI

    app = FastAPI()
    container = _TestContainer(store)
    router = build_router(container)
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

    def test_empty_list(self) -> None:
        """Test listing covenants when none exist for deal."""
        store = _InMemoryCovenantStore()
        client = _create_test_client(store)

        response = client.get("/covenants/by-deal/deal-123")

        assert response.status_code == 200
        data = load_json_str(response.text)
        assert data == []

    def test_list_with_covenants(self) -> None:
        """Test listing covenants when some exist for deal."""
        store = _InMemoryCovenantStore()
        store.covenants["c1"] = _make_covenant("c1", "deal-123", "Covenant One")
        store.covenants["c2"] = _make_covenant("c2", "deal-123", "Covenant Two")
        store.covenants["c3"] = _make_covenant("c3", "other-deal", "Other Covenant")
        client = _create_test_client(store)

        response = client.get("/covenants/by-deal/deal-123")

        assert response.status_code == 200
        data = narrow_json_to_list(load_json_str(response.text))
        assert len(data) == 2


class TestCreateCovenant:
    """Tests for POST /covenants."""

    def test_create_covenant_success(self) -> None:
        """Test creating a new covenant."""
        store = _InMemoryCovenantStore()
        client = _create_test_client(store)

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
        assert "new-cov" in store.covenants
        data = narrow_json_to_dict(load_json_str(response.text))
        assert data["name"] == "Debt to EBITDA"

    def test_create_covenant_gte_direction(self) -> None:
        """Test creating a covenant with >= direction."""
        store = _InMemoryCovenantStore()
        client = _create_test_client(store)

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
        assert store.covenants["cov-gte"]["threshold_direction"] == ">="
        assert store.covenants["cov-gte"]["frequency"] == "ANNUAL"


class TestGetCovenant:
    """Tests for GET /covenants/{covenant_id}."""

    def test_get_existing_covenant(self) -> None:
        """Test getting an existing covenant."""
        store = _InMemoryCovenantStore()
        store.covenants["c1"] = _make_covenant("c1", "deal-123", "Test Covenant")
        client = _create_test_client(store)

        response = client.get("/covenants/c1")

        assert response.status_code == 200
        data = narrow_json_to_dict(load_json_str(response.text))
        assert data["name"] == "Test Covenant"

    def test_get_nonexistent_covenant(self) -> None:
        """Test getting a covenant that doesn't exist."""
        store = _InMemoryCovenantStore()
        client = _create_test_client(store)

        response = client.get("/covenants/nonexistent")

        assert response.status_code == 500


class TestDeleteCovenant:
    """Tests for DELETE /covenants/{covenant_id}."""

    def test_delete_existing_covenant(self) -> None:
        """Test deleting an existing covenant."""
        store = _InMemoryCovenantStore()
        store.covenants["c1"] = _make_covenant("c1", "deal-123")
        client = _create_test_client(store)

        response = client.delete("/covenants/c1")

        assert response.status_code == 204
        assert "c1" not in store.covenants

    def test_delete_nonexistent_covenant(self) -> None:
        """Test deleting a covenant that doesn't exist."""
        store = _InMemoryCovenantStore()
        client = _create_test_client(store)

        response = client.delete("/covenants/nonexistent")

        assert response.status_code == 500
