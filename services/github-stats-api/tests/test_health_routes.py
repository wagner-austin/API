from __future__ import annotations

import httpx
import pytest

from github_stats_api.api.main import create_app
from github_stats_api.settings import Settings


def _make_test_settings() -> Settings:
    """Test settings fixture."""
    return {
        "github_token": "test-token",
        "cache_ttl_seconds": 60,
        "port": 8000,
    }


def _make_app(test_settings: Settings) -> httpx.ASGITransport:
    """Create test app with ASGI transport."""
    fastapi_app = create_app(test_settings)
    return httpx.ASGITransport(app=fastapi_app)


test_settings = pytest.fixture(_make_test_settings)
app = pytest.fixture(_make_app)


class TestHealthRoutes:
    """Tests for health check routes."""

    async def test_health_endpoint(self, app: httpx.ASGITransport) -> None:
        """Test /health endpoint returns ok."""
        async with httpx.AsyncClient(transport=app, base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        assert response.text == "ok"

    async def test_healthz_endpoint(self, app: httpx.ASGITransport) -> None:
        """Test /healthz endpoint returns ok."""
        async with httpx.AsyncClient(transport=app, base_url="http://test") as client:
            response = await client.get("/healthz")

        assert response.status_code == 200
        assert response.text == "ok"

    async def test_readyz_endpoint(self, app: httpx.ASGITransport) -> None:
        """Test /readyz endpoint returns ok."""
        async with httpx.AsyncClient(transport=app, base_url="http://test") as client:
            response = await client.get("/readyz")

        assert response.status_code == 200
        assert response.text == "ok"
