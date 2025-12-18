from __future__ import annotations

import httpx
import pytest

from procart_api.app import create_app


@pytest.mark.asyncio
async def test_registries_endpoints_list_names() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        r1 = await ac.get("/registries/modules")
        assert r1.status_code == 200
        mods: dict[str, list[str]] = r1.json()
        assert "modules" in mods and "neon_orbs" in mods["modules"]

        r2 = await ac.get("/registries/camera-paths")
        cams: dict[str, list[str]] = r2.json()
        assert "circular" in cams["camera_paths"]

        r3 = await ac.get("/registries/tone-mappers")
        tones: dict[str, list[str]] = r3.json()
        assert "exposure_gamma" in tones["tone_mappers"]

        r4 = await ac.get("/registries/post-effects")
        posts: dict[str, list[str]] = r4.json()
        assert posts["post_effects"] == ["bloom"]

        r5 = await ac.get("/registries/composite-ops")
        comps: dict[str, list[str]] = r5.json()
        assert "normal" in comps["composite_ops"] and "screen" in comps["composite_ops"]
