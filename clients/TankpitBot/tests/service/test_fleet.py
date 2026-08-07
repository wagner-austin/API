"""Tests for the AI-operated fleet manager."""

from __future__ import annotations

from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot._test_hooks.fs import PathExistsProtocol
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service._test_hooks import _real_run_web_app, _real_spawn_bot_process
from tankpit_bot.service.fleet import main
from tankpit_bot.service.fleet_manager import (
    FLEET_PORT_DEFAULT,
    FleetError,
    FleetManager,
    resolve_fleet_port,
)
from tests.conftest import FakeEnv
from tests.service._fleet_fixtures import (
    _FakeSpawner,
    _restore_account_hooks,
    _with_configured_accounts,
)


def _without_accounts() -> PathExistsProtocol:
    """Make accounts.json absent so tests never read the real file.

    Returns:
        The original ``path_exists`` hook to restore.
    """

    def fake_missing(path: Path) -> bool:
        _ = path
        return False

    original = top_hooks.path_exists
    top_hooks.path_exists = fake_missing
    return original


def test_spawn_builds_the_instance_environment(spawner: _FakeSpawner) -> None:
    """The child receives instance, bounds, and account via env."""
    originals = _with_configured_accounts()
    try:
        manager = FleetManager()
        row = manager.spawn(instance="alpha", account="second", kills=30, seconds=2700)
    finally:
        _restore_account_hooks(originals)

    assert spawner.envs == [
        {
            "TANKPIT_BOT_INSTANCE": "alpha",
            "TANKPIT_BOT_SESSION_KILLS": "30",
            "TANKPIT_BOT_SESSION_SECONDS": "2700",
            "TANKPIT_ACCOUNT": "second",
        }
    ]
    assert row["instance"] == "alpha"
    assert row["alive"] is True
    assert row["pid"] == 1001


def test_accounts_lists_configured_usernames_only(spawner: _FakeSpawner) -> None:
    """The account surface is accounts.json usernames — never passwords."""
    _ = spawner
    originals = _with_configured_accounts()
    try:
        manager = FleetManager()
        names = manager.accounts()
        row = manager.spawn(instance="alpha", account="second", kills=0, seconds=0)
        with pytest.raises(FleetError, match=r"not in accounts\.json"):
            manager.spawn(instance="bravo", account="intruder", kills=0, seconds=0)
    finally:
        _restore_account_hooks(originals)

    assert names == ["artax", "second"]
    assert row["account"] == "second"


def test_accounts_without_a_file_is_empty_and_default_still_spawns(
    spawner: _FakeSpawner,
) -> None:
    """No accounts.json: the list is empty and only default spawns."""

    def fake_exists(path: Path) -> bool:
        _ = path
        return False

    original_exists = top_hooks.path_exists
    top_hooks.path_exists = fake_exists
    try:
        manager = FleetManager()
        names = manager.accounts()
        manager.spawn(instance="alpha", account="", kills=0, seconds=0)
        with pytest.raises(FleetError, match="none configured"):
            manager.spawn(instance="bravo", account="anyone", kills=0, seconds=0)
    finally:
        top_hooks.path_exists = original_exists

    assert names == []
    assert len(spawner.envs) == 1


def test_spawn_without_account_omits_the_selector(spawner: _FakeSpawner) -> None:
    """An empty account means the accounts.json default, not an empty var."""
    manager = FleetManager()

    manager.spawn(instance="alpha", account="", kills=0, seconds=0)

    assert "TANKPIT_ACCOUNT" not in spawner.envs[0]


def test_spawn_rejects_invalid_instance_names(spawner: _FakeSpawner) -> None:
    """Path characters and uppercase never reach the filesystem layer."""
    manager = FleetManager()

    with pytest.raises(FleetError, match="not a valid instance name"):
        manager.spawn(instance="../escape", account="", kills=0, seconds=0)
    with pytest.raises(FleetError, match="not a valid instance name"):
        manager.spawn(instance="UPPER", account="", kills=0, seconds=0)
    assert spawner.envs == []


def test_spawn_rejects_negative_bounds(spawner: _FakeSpawner) -> None:
    """Negative bounds are a loud error, not a weird session."""
    manager = FleetManager()

    with pytest.raises(FleetError, match="non-negative"):
        manager.spawn(instance="alpha", account="", kills=-1, seconds=0)
    assert spawner.envs == []


def test_spawn_refuses_a_live_duplicate_but_replaces_a_dead_one(
    spawner: _FakeSpawner,
) -> None:
    """One live process per instance; a finished one may be respawned."""
    original = _without_accounts()
    try:
        manager = FleetManager()
        manager.spawn(instance="alpha", account="", kills=0, seconds=0)

        with pytest.raises(FleetError, match="already running"):
            manager.spawn(instance="alpha", account="", kills=0, seconds=0)

        spawner.processes[0].returncode = 0
        row = manager.spawn(instance="alpha", account="", kills=0, seconds=0)
    finally:
        top_hooks.path_exists = original
    assert row["pid"] == 1002


def test_report_sorts_and_reflects_liveness(spawner: _FakeSpawner) -> None:
    """The report row set is sorted and tracks process exit."""
    original = _without_accounts()
    try:
        manager = FleetManager()
        manager.spawn(instance="bravo", account="", kills=0, seconds=0)
        manager.spawn(instance="alpha", account="", kills=0, seconds=0)
        spawner.processes[0].returncode = 7
        rows = manager.report()
    finally:
        top_hooks.path_exists = original

    assert [row["instance"] for row in rows] == ["alpha", "bravo"]
    assert rows[1]["alive"] is False
    assert rows[1]["returncode"] == 7


def test_stop_writes_the_instance_sentinel(spawner: _FakeSpawner) -> None:
    """A graceful stop is the instance's STOP file, nothing more."""
    written: list[tuple[Path, str]] = []

    def fake_write(path: Path, content: str) -> None:
        written.append((Path(path), content))

    original_write = top_hooks.write_text
    top_hooks.write_text = fake_write
    try:
        manager = FleetManager()
        manager.spawn(instance="alpha", account="", kills=0, seconds=0)
        manager.stop("alpha")
    finally:
        top_hooks.write_text = original_write

    assert written == [(Path("runs/bot/alpha/STOP"), "")]


def test_stop_unknown_instance_is_a_fleet_error(spawner: _FakeSpawner) -> None:
    """Stopping a name that was never spawned names the problem."""
    manager = FleetManager()

    with pytest.raises(FleetError, match="unknown instance"):
        manager.stop("ghost")


def test_remove_refuses_a_live_instance_and_drops_a_dead_one(
    spawner: _FakeSpawner,
) -> None:
    """The fleet never silently kills — stop first, then remove."""
    manager = FleetManager()
    manager.spawn(instance="alpha", account="", kills=0, seconds=0)

    with pytest.raises(FleetError, match="still running"):
        manager.remove("alpha")

    spawner.processes[0].returncode = 0
    row = manager.remove("alpha")
    assert row["alive"] is False
    assert manager.report() == []


def test_restart_respawns_a_dead_instance_with_its_parameters(
    spawner: _FakeSpawner,
) -> None:
    """Restart reuses the stored account and bounds, refusing while alive."""
    originals = _with_configured_accounts()
    try:
        manager = FleetManager()
        manager.spawn(instance="alpha", account="second", kills=30, seconds=2700)

        with pytest.raises(FleetError, match="still running"):
            manager.restart("alpha")
        with pytest.raises(FleetError, match="unknown instance"):
            manager.restart("ghost")

        spawner.processes[0].returncode = 0
        row = manager.restart("alpha")
    finally:
        _restore_account_hooks(originals)
    assert row["pid"] == 1002
    assert spawner.envs[1] == spawner.envs[0]


def test_stats_summarizes_the_instance_events(spawner: _FakeSpawner) -> None:
    """The stats summary reads the instance's events via the digest."""
    events = "\n".join(
        [
            '{"timestamp":"2026-08-06T10:00:00","level":"INFO","logger":"l",'
            '"mode":"bot","channel":"STATE","message":"INITIALIZING"}',
            '{"timestamp":"2026-08-06T10:05:00","level":"INFO","logger":"l",'
            '"mode":"bot","channel":"COMBAT","message":"kill registered"}',
        ]
    )
    reads: list[Path] = []

    def fake_read(path: Path) -> str:
        reads.append(Path(path))
        return events

    original_exists = _without_accounts()
    original_read = top_hooks.read_text
    top_hooks.read_text = fake_read
    try:
        manager = FleetManager()
        manager.spawn(instance="alpha", account="", kills=0, seconds=0)
        summary = manager.stats("alpha")
    finally:
        top_hooks.read_text = original_read
        top_hooks.path_exists = original_exists

    assert reads == [Path("runs/bot/alpha/latest.events.jsonl")]
    assert summary["available"] is True
    assert summary["kills"] == 1
    assert summary["deaths"] == 0
    assert summary["duration_s"] == 300
    assert summary["clean_exit"] is False


def test_stats_without_events_is_unavailable_not_an_error(
    spawner: _FakeSpawner,
) -> None:
    """A just-spawned bot with no events yet reports available=False."""

    def fake_read(path: Path) -> str:
        raise OSError(f"no such file {path}")

    original_exists = _without_accounts()
    original_read = top_hooks.read_text
    top_hooks.read_text = fake_read
    try:
        manager = FleetManager()
        manager.spawn(instance="alpha", account="", kills=0, seconds=0)
        summary = manager.stats("alpha")
        with pytest.raises(FleetError, match="unknown instance"):
            manager.stats("ghost")
    finally:
        top_hooks.read_text = original_read
        top_hooks.path_exists = original_exists

    assert summary == {"available": False}


def test_resolve_fleet_port_contract() -> None:
    """Default, override, and loud rejection."""
    original_get_env = top_hooks.get_env
    try:
        top_hooks.get_env = FakeEnv({})
        assert resolve_fleet_port() == FLEET_PORT_DEFAULT
        top_hooks.get_env = FakeEnv({"TANKPIT_FLEET_PORT": "27301"})
        assert resolve_fleet_port() == 27301
        top_hooks.get_env = FakeEnv({"TANKPIT_FLEET_PORT": "80"})
        with pytest.raises(ValueError, match="outside"):
            resolve_fleet_port()
    finally:
        top_hooks.get_env = original_get_env


def test_main_wires_the_app_onto_the_resolved_port() -> None:
    """``main`` loads dotenv, resolves the port, and serves the routes."""
    served: list[tuple[web.Application, str, int]] = []
    loads: list[str] = []

    def fake_dotenv() -> None:
        loads.append("dotenv")

    def fake_run(app: web.Application, *, host: str, port: int) -> None:
        served.append((app, host, port))

    original_dotenv = core_hooks.load_dotenv
    original_run = service_hooks.run_web_app
    original_get_env = top_hooks.get_env
    try:
        core_hooks.load_dotenv = fake_dotenv
        service_hooks.run_web_app = fake_run
        top_hooks.get_env = FakeEnv({"TANKPIT_FLEET_PORT": "27311"})
        main()
    finally:
        core_hooks.load_dotenv = original_dotenv
        service_hooks.run_web_app = original_run
        top_hooks.get_env = original_get_env

    assert loads == ["dotenv"]
    if len(served) != 1:
        raise AssertionError(f"expected one serve call, got {served!r}")
    app, host, port = served[0]
    assert (host, port) == ("127.0.0.1", 27311)
    canonical = {resource.canonical for resource in app.router.resources()}
    assert canonical == {
        "/",
        "/accounts",
        "/bots",
        "/bots/{instance}/stats",
        "/bots/{instance}/hud",
        "/bots/{instance}/activity",
        "/bots/{instance}/stop",
        "/bots/{instance}/restart",
        "/bots/{instance}",
    }


def test_real_run_web_app_drives_aiohttp_until_interrupted() -> None:
    """The production runner reaches app startup and unwinds cleanly."""
    app = web.Application()
    reached: list[str] = []

    async def interrupt(started_app: web.Application) -> None:
        _ = started_app
        reached.append("startup")
        raise KeyboardInterrupt

    app.on_startup.append(interrupt)
    _real_run_web_app(app, host="127.0.0.1", port=0)
    assert reached == ["startup"]


def test_real_spawn_bot_process_launches_a_live_python_child() -> None:
    """The production spawner starts a real child; killed before it acts.

    The child is terminated immediately — interpreter startup takes far
    longer than the kill lands, so it never reaches the bot entry point
    (which would open a browser).
    """
    process = _real_spawn_bot_process({"TANKPIT_BOT_INSTANCE": "covspawn"})
    try:
        assert process.pid > 0
    finally:
        process.kill()
        process.wait(timeout=30)
    if process.poll() is None:
        raise AssertionError("child still running after kill + wait")


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
        payload: dict[str, str | int] = {"instance": "alpha", "kills": 30}
        created = await fleet_client.post("/bots", json=payload)
        assert created.status == 201

        listed = await fleet_client.get("/bots")
        body = narrow_json_to_dict(load_json_str(await listed.text()))
        bots = body["bots"]
        assert isinstance(bots, list) and len(bots) == 1

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

    malformed: dict[str, str] = {"kills": "many"}
    assert (await fleet_client.post("/bots", json=malformed)).status == 400

    ok: dict[str, str | int] = {"instance": "alpha"}
    assert (await fleet_client.post("/bots", json=ok)).status == 201
    assert (await fleet_client.post("/bots", json=ok)).status == 409

    assert (await fleet_client.post("/bots/ghost/stop")).status == 404
    assert (await fleet_client.delete("/bots/ghost")).status == 404
