"""Tests for the AI-operated fleet manager."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service._test_hooks import _real_run_web_app, _real_spawn_bot_process
from tankpit_bot.service.fleet import (
    FLEET_PORT_DEFAULT,
    FleetError,
    FleetManager,
    main,
    make_fleet_app,
    resolve_fleet_port,
)
from tests.conftest import FakeEnv


class _FakeProcess:
    """Controllable child-process double."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode


class _FakeSpawner:
    """Records spawn environments and hands out process doubles."""

    def __init__(self) -> None:
        self.envs: list[dict[str, str]] = []
        self.processes: list[_FakeProcess] = []

    def __call__(self, env_overrides: dict[str, str]) -> _FakeProcess:
        self.envs.append(dict(env_overrides))
        process = _FakeProcess(pid=1001 + len(self.processes))
        self.processes.append(process)
        return process


@pytest.fixture()
def spawner() -> Generator[_FakeSpawner, None, None]:
    """Install a recording spawner for the duration of one test."""
    original = service_hooks.spawn_bot_process
    fake = _FakeSpawner()
    service_hooks.spawn_bot_process = fake
    yield fake
    service_hooks.spawn_bot_process = original


def test_spawn_builds_the_instance_environment(spawner: _FakeSpawner) -> None:
    """The child receives instance, bounds, and account via env."""
    manager = FleetManager()

    row = manager.spawn(instance="alpha", account="second", kills=30, seconds=2700)

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
    manager = FleetManager()
    manager.spawn(instance="alpha", account="", kills=0, seconds=0)

    with pytest.raises(FleetError, match="already running"):
        manager.spawn(instance="alpha", account="", kills=0, seconds=0)

    spawner.processes[0].returncode = 0
    row = manager.spawn(instance="alpha", account="", kills=0, seconds=0)
    assert row["pid"] == 1002


def test_report_sorts_and_reflects_liveness(spawner: _FakeSpawner) -> None:
    """The report row set is sorted and tracks process exit."""
    manager = FleetManager()
    manager.spawn(instance="bravo", account="", kills=0, seconds=0)
    manager.spawn(instance="alpha", account="", kills=0, seconds=0)
    spawner.processes[0].returncode = 7

    rows = manager.report()

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

    original_dotenv = service_hooks.load_dotenv
    original_run = service_hooks.run_web_app
    original_get_env = top_hooks.get_env
    try:
        service_hooks.load_dotenv = fake_dotenv
        service_hooks.run_web_app = fake_run
        top_hooks.get_env = FakeEnv({"TANKPIT_FLEET_PORT": "27311"})
        main()
    finally:
        service_hooks.load_dotenv = original_dotenv
        service_hooks.run_web_app = original_run
        top_hooks.get_env = original_get_env

    assert loads == ["dotenv"]
    if len(served) != 1:
        raise AssertionError(f"expected one serve call, got {served!r}")
    app, host, port = served[0]
    assert (host, port) == ("127.0.0.1", 27311)
    canonical = {resource.canonical for resource in app.router.resources()}
    assert canonical == {"/bots", "/bots/{instance}/stop", "/bots/{instance}"}


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


@pytest.fixture()
async def client(
    spawner: _FakeSpawner,
) -> AsyncGenerator[TestClient[web.Request, web.Application], None]:
    """Serve the fleet app on a random test port."""
    manager = FleetManager()
    app = make_fleet_app(manager)
    test_client: TestClient[web.Request, web.Application] = TestClient(TestServer(app))
    await test_client.start_server()
    yield test_client
    await test_client.close()


@pytest.mark.asyncio
async def test_http_spawn_list_stop_remove_cycle(
    client: TestClient[web.Request, web.Application],
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
        created = await client.post("/bots", json=payload)
        assert created.status == 201

        listed = await client.get("/bots")
        body = narrow_json_to_dict(load_json_str(await listed.text()))
        bots = body["bots"]
        assert isinstance(bots, list) and len(bots) == 1

        stopped = await client.post("/bots/alpha/stop")
        assert stopped.status == 200
        assert written == [(Path("runs/bot/alpha/STOP"), "")]

        blocked = await client.delete("/bots/alpha")
        assert blocked.status == 409

        spawner.processes[0].returncode = 0
        removed = await client.delete("/bots/alpha")
        assert removed.status == 200
    finally:
        top_hooks.write_text = original_write


@pytest.mark.asyncio
async def test_http_rejections_are_typed_statuses(
    client: TestClient[web.Request, web.Application],
    spawner: _FakeSpawner,
) -> None:
    """400 for malformed spawns, 409 for duplicates, 404 for ghosts."""
    bad: dict[str, str | int] = {"instance": "../escape"}
    assert (await client.post("/bots", json=bad)).status == 409

    malformed: dict[str, str] = {"kills": "many"}
    assert (await client.post("/bots", json=malformed)).status == 400

    ok: dict[str, str | int] = {"instance": "alpha"}
    assert (await client.post("/bots", json=ok)).status == 201
    assert (await client.post("/bots", json=ok)).status == 409

    assert (await client.post("/bots/ghost/stop")).status == 404
    assert (await client.delete("/bots/ghost")).status == 404
