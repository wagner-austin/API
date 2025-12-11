"""Tests for status route."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_list,
    require_bool,
    require_str,
)

from covenant_radar_api.api.routes.status import build_router

from .conftest import ContainerAndStore


def _create_test_client(cas: ContainerAndStore) -> TestClient:
    """Create test client with real container."""
    app = FastAPI()
    router = build_router(cas.container)
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


class TestStatusEndpoint:
    """Tests for GET /status."""

    def test_status_returns_service_info(self, container_with_store: ContainerAndStore) -> None:
        """Test status returns service name and version."""
        client = _create_test_client(container_with_store)
        response = client.get("/status")

        assert response.status_code == 200
        data = narrow_json_to_dict(load_json_str(response.text))
        assert require_str(data, "service") == "covenant-radar-api"
        assert require_str(data, "version") == "0.1.0"

    def test_status_returns_dependencies(self, container_with_store: ContainerAndStore) -> None:
        """Test status returns dependency health checks."""
        client = _create_test_client(container_with_store)
        response = client.get("/status")

        assert response.status_code == 200
        data = narrow_json_to_dict(load_json_str(response.text))
        deps = narrow_json_to_list(data["dependencies"])
        assert len(deps) == 2

        redis_dep = narrow_json_to_dict(deps[0])
        assert require_str(redis_dep, "name") == "redis"
        assert require_str(redis_dep, "status") == "ok"

        postgres_dep = narrow_json_to_dict(deps[1])
        assert require_str(postgres_dep, "name") == "postgres"
        assert require_str(postgres_dep, "status") == "ok"

    def test_status_returns_model_info(self, container_with_store: ContainerAndStore) -> None:
        """Test status returns model information."""
        client = _create_test_client(container_with_store)
        response = client.get("/status")

        assert response.status_code == 200
        data = narrow_json_to_dict(load_json_str(response.text))
        model = narrow_json_to_dict(data["model"])
        assert require_str(model, "model_id") == "default"
        assert require_str(model, "model_path").endswith("test_model.ubj")
        assert require_bool(model, "is_loaded") is False

    def test_status_returns_data_counts(self, container_with_store: ContainerAndStore) -> None:
        """Test status returns data counts."""
        client = _create_test_client(container_with_store)
        response = client.get("/status")

        assert response.status_code == 200
        data = narrow_json_to_dict(load_json_str(response.text))
        data_counts = narrow_json_to_dict(data["data"])
        # Initially no deals
        assert data_counts["deals"] == 0

    def test_status_counts_deals_after_creation(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Test status counts deals correctly after adding them."""
        from covenant_domain import Deal, DealId

        # Add a deal to the store
        container_with_store.store.deals["d1"] = Deal(
            id=DealId(value="d1"),
            name="Test Deal",
            borrower="Test Corp",
            sector="Technology",
            region="North America",
            commitment_amount_cents=100_000_000,
            currency="USD",
            maturity_date_iso="2025-12-31",
        )
        container_with_store.store._deal_order.append("d1")

        client = _create_test_client(container_with_store)
        response = client.get("/status")

        assert response.status_code == 200
        data = narrow_json_to_dict(load_json_str(response.text))
        data_counts = narrow_json_to_dict(data["data"])
        assert data_counts["deals"] == 1


class TestStatusDependencyChecks:
    """Tests for individual dependency check functions."""

    def test_check_redis_ok(self) -> None:
        """Test Redis check returns ok when ping succeeds."""
        from platform_workers.testing import FakeRedis

        from covenant_radar_api.api.routes.status import _check_redis

        fake_redis = FakeRedis()
        result = _check_redis(fake_redis)
        assert result["name"] == "redis"
        assert result["status"] == "ok"
        assert result["message"] is None
        fake_redis.assert_only_called({"ping"})

    def test_check_redis_false_pong(self) -> None:
        """Test Redis check returns error when ping returns False."""
        from platform_workers.testing import FakeRedisNoPong

        from covenant_radar_api.api.routes.status import _check_redis

        fake_redis = FakeRedisNoPong()
        result = _check_redis(fake_redis)
        assert result["name"] == "redis"
        assert result["status"] == "error"
        assert result["message"] == "ping returned false"
        fake_redis.assert_only_called({"ping"})

    def test_check_redis_exception(self) -> None:
        """Test Redis check returns error when ping raises exception."""
        from platform_workers.testing import FakeRedisNonRedisError

        from covenant_radar_api.api.routes.status import _check_redis

        fake_redis = FakeRedisNonRedisError()
        result = _check_redis(fake_redis)
        assert result["name"] == "redis"
        assert result["status"] == "error"
        assert "non-Redis failure" in str(result["message"])
        fake_redis.assert_only_called({"ping"})

    def test_check_database_ok(self, container_with_store: ContainerAndStore) -> None:
        """Test database check returns ok and count."""
        from covenant_radar_api.api.routes.status import _check_database

        repo = container_with_store.container.deal_repo()
        status, count = _check_database(repo)
        assert status["name"] == "postgres"
        assert status["status"] == "ok"
        assert count == 0

    def test_check_database_exception(self) -> None:
        """Test database check returns error when list_all raises exception."""
        from collections.abc import Sequence

        from covenant_domain import Deal, DealId

        from covenant_radar_api.api.routes.status import _check_database

        class _FailingDealRepo:
            def list_all(self) -> Sequence[Deal]:
                raise RuntimeError("Database connection failed")

            def get(self, deal_id: DealId) -> Deal:
                raise NotImplementedError

            def create(self, deal: Deal) -> None:
                raise NotImplementedError

            def update(self, deal: Deal) -> None:
                raise NotImplementedError

            def delete(self, deal_id: DealId) -> None:
                raise NotImplementedError

        repo = _FailingDealRepo()
        status, count = _check_database(repo)
        assert status["name"] == "postgres"
        assert status["status"] == "error"
        assert "Database connection failed" in str(status["message"])
        assert count == 0
