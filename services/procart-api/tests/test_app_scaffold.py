from __future__ import annotations

import httpx
import pytest

from procart_api.app import create_app


@pytest.mark.asyncio
async def test_health_endpoint_asgi_transport() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/health")
        assert resp.status_code == 200
        data: dict[str, str] = resp.json()
        assert data == {"status": "ok"}
