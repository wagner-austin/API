"""Tests for the fleet HUD route serving a written frame.

The route has three outcomes and the interesting one is the success:
a run that HAS written ``hud.json`` gets it back verbatim, because
the control page draws the same card the bot's own HUD does. Serving
it re-encoded would let the two drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot._test_hooks.fs import ReadTextProtocol


@pytest.mark.asyncio
async def test_hud_route_serves_a_written_frame_verbatim(
    fleet_client: TestClient[web.Request, web.Application],
) -> None:
    """A run with a HUD frame gets exactly the bytes it wrote."""
    spawn = await fleet_client.post("/bots", json={"instance": "alpha", "kills": 5})
    assert spawn.status == 201

    written = '{"mode": "HUNT", "fuel": 812, "kills": 3}'
    requested: list[Path] = []

    def _read_text(path: Path) -> str:
        requested.append(path)
        return written

    original_read: ReadTextProtocol = top_hooks.read_text
    top_hooks.read_text = _read_text
    try:
        response = await fleet_client.get("/bots/alpha/hud")
        body = await response.text()
    finally:
        top_hooks.read_text = original_read

    assert response.status == 200
    assert response.content_type == "application/json"
    assert body == written
    assert narrow_json_to_dict(load_json_str(body)) == {
        "mode": "HUNT",
        "fuel": 812,
        "kills": 3,
    }
    assert [path.name for path in requested] == ["hud.json"]
