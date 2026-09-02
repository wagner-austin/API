"""Tests for the fleet manager's HTTP surface.

Split from ``test_fleet.py`` (2026-08-28, the room-dropdown lift) when
the combined module crossed the 600-line ceiling. This half drives the
aiohttp app end to end through ``fleet_client``; the domain half keeps
:class:`~tankpit_bot.service.fleet_manager.FleetManager` itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks as top_hooks
from tests.service._fleet_fixtures import (
    _FakeSpawner,
    _restore_account_hooks,
    _with_configured_accounts,
)


@pytest.mark.asyncio
async def test_http_spawn_list_stop_remove_cycle(
    fleet_client: TestClient[web.Request, web.Application],
    spawner: _FakeSpawner,
) -> None:
    """The full AI-driven lifecycle over plain HTTP."""
    written: list[tuple[Path, str]] = []

    def fake_write(path: Path, content: str) -> None:
        written.append((Path(path), content))

    original_write = top_hooks.write_text
    top_hooks.write_text = fake_write
    try:
        payload: dict[str, str | int] = {"instance": "alpha", "kills": 30, "role": "gatherer"}
        created = await fleet_client.post("/bots", json=payload)
        assert created.status == 201
        created_row = narrow_json_to_dict(load_json_str(await created.text()))
        assert created_row["role"] == "gatherer"

        listed = await fleet_client.get("/bots")
        body = narrow_json_to_dict(load_json_str(await listed.text()))
        bots = body["bots"]
        assert isinstance(bots, list) and len(bots) == 1
        first = narrow_json_to_dict(bots[0])
        assert first["role"] == "gatherer"

        stopped = await fleet_client.post("/bots/alpha/stop")
        assert stopped.status == 200
        assert written == [(Path("runs/bot/alpha/STOP"), "")]

        blocked = await fleet_client.delete("/bots/alpha")
        assert blocked.status == 409

        spawner.processes[0].returncode = 0
        removed = await fleet_client.delete("/bots/alpha")
        assert removed.status == 200
    finally:
        top_hooks.write_text = original_write


@pytest.mark.asyncio
async def test_http_page_stats_and_restart(
    fleet_client: TestClient[web.Request, web.Application],
    spawner: _FakeSpawner,
) -> None:
    """The control page serves, stats answer JSON, restart respawns."""
    page = await fleet_client.get("/")
    assert page.status == 200
    assert (page.headers["Content-Type"]).startswith("text/html")
    body = await page.text()
    assert "Tankpit Fleet" in body and "/bots" in body and "/accounts" in body
    # The spawn form offers every fleet role — a gatherer is a spawn
    # choice, not an env var the operator has to remember.
    assert 'id="role"' in body and "gatherer" in body
    # Room is a dropdown fed by /rooms, not a text box: the lobby has
    # two rooms and neither should have to be typed from memory.
    assert '<select id="room">' in body and '"/rooms"' in body

    listed_rooms = await fleet_client.get("/rooms")
    assert listed_rooms.status == 200
    rooms_payload = narrow_json_to_dict(load_json_str(await listed_rooms.text()))
    assert rooms_payload == {"rooms": ["World", "Practice"]}

    # Color has no "account default" option: an account holds one tank
    # PER COLOR with its own rank, so the operator states which tank
    # plays rather than inheriting whichever was played last.
    assert '<select id="troop"></select>' in body and '"/troops"' in body
    # Roles read as proper nouns on the form; the wire values stay lower.
    assert ">Fighter<" in body and '"fighter"' in body
    # The colour panel's source: measured rank per account/world/colour.
    # The page must index the SERVED shape — rows nest under
    # "accounts", and reading tanks[account] directly silently found
    # nothing and printed "no reading" for every colour (2026-08-31).
    tanks = await fleet_client.get("/tanks")
    assert tanks.status == 200
    assert "tanks" in narrow_json_to_dict(load_json_str(await tanks.text()))
    assert "tanks.accounts" in body

    # Doctrine is the fourth served vocabulary, same pattern as the
    # rest: the page never asks a human to spell one.
    assert '<select id="doctrine">' in body and '"/doctrines"' in body
    listed_doctrines = await fleet_client.get("/doctrines")
    assert listed_doctrines.status == 200
    doctrines_payload = narrow_json_to_dict(load_json_str(await listed_doctrines.text()))
    assert doctrines_payload == {
        "doctrines": ["skirmish", "swarm", "duelist", "passive"]
    }

    listed_troops = await fleet_client.get("/troops")
    assert listed_troops.status == 200
    troops_payload = narrow_json_to_dict(load_json_str(await listed_troops.text()))
    assert troops_payload == {"troops": ["red", "purple", "blue", "orange"]}

    originals = _with_configured_accounts()
    try:
        listed_accounts = await fleet_client.get("/accounts")
        assert listed_accounts.status == 200
        accounts_payload = narrow_json_to_dict(load_json_str(await listed_accounts.text()))
        assert accounts_payload == {"accounts": ["artax", "second"]}
    finally:
        _restore_account_hooks(originals)

    ok: dict[str, str | int] = {"instance": "alpha", "kills": 5}
    assert (await fleet_client.post("/bots", json=ok)).status == 201

    def fake_read(path: Path) -> str:
        raise OSError(f"no events at {path}")

    original_read = top_hooks.read_text
    top_hooks.read_text = fake_read
    try:
        stats = await fleet_client.get("/bots/alpha/stats")
        assert stats.status == 200
        payload = narrow_json_to_dict(load_json_str(await stats.text()))
        assert payload == {"available": False}
        assert (await fleet_client.get("/bots/ghost/stats")).status == 404
        activity = await fleet_client.get("/bots/alpha/activity")
        assert activity.status == 200
        activity_payload = narrow_json_to_dict(load_json_str(await activity.text()))
        assert activity_payload == {"available": False}
        assert (await fleet_client.get("/bots/ghost/activity")).status == 404
        hud = await fleet_client.get("/bots/alpha/hud")
        assert hud.status == 200
        hud_payload = narrow_json_to_dict(load_json_str(await hud.text()))
        assert hud_payload == {"available": False}
        assert (await fleet_client.get("/bots/ghost/hud")).status == 404
    finally:
        top_hooks.read_text = original_read

    assert (await fleet_client.post("/bots/alpha/restart")).status == 409
    assert (await fleet_client.post("/bots/ghost/restart")).status == 404
    spawner.processes[0].returncode = 0
    restarted = await fleet_client.post("/bots/alpha/restart")
    assert restarted.status == 201
    row = narrow_json_to_dict(load_json_str(await restarted.text()))
    assert row["pid"] == 1002


@pytest.mark.asyncio
async def test_http_rejections_are_typed_statuses(
    fleet_client: TestClient[web.Request, web.Application],
    spawner: _FakeSpawner,
) -> None:
    """400 for malformed spawns, 409 for duplicates, 404 for ghosts."""
    bad: dict[str, str | int] = {"instance": "../escape"}
    assert (await fleet_client.post("/bots", json=bad)).status == 409

    bad_role: dict[str, str | int] = {"instance": "bravo", "role": "scout"}
    refused_role = await fleet_client.post("/bots", json=bad_role)
    assert refused_role.status == 409
    assert "not a fleet role" in await refused_role.text()

    malformed: dict[str, str] = {"kills": "many"}
    assert (await fleet_client.post("/bots", json=malformed)).status == 400

    ok: dict[str, str | int] = {"instance": "alpha"}
    assert (await fleet_client.post("/bots", json=ok)).status == 201
    assert (await fleet_client.post("/bots", json=ok)).status == 409

    assert (await fleet_client.post("/bots/ghost/stop")).status == 404
    assert (await fleet_client.delete("/bots/ghost")).status == 404
