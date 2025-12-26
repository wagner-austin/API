from __future__ import annotations

import httpx


async def test_asgi_module_app_health_endpoint() -> None:
    """Test that asgi module app responds to health endpoint."""
    from github_stats_api.asgi import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200, "app should have /health route"


async def test_asgi_module_exports_app_in_all() -> None:
    """Test that asgi module exports app in __all__."""
    from github_stats_api import asgi

    assert "app" in asgi.__all__, "__all__ should contain 'app'"


async def test_asgi_app_title() -> None:
    """Test that asgi app has correct title."""
    from github_stats_api.asgi import app

    assert app.title == "github-stats-api", "app should have correct title"
