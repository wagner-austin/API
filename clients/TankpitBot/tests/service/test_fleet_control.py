"""``make up`` / ``make down``: the fleet's lifecycle client.

Driven against a REAL manager app on a real loopback port, because
what these commands are for is exactly the wire: whether a manager is
listening, what it says about its bots, and whether the port has gone
quiet. A fake HTTP layer would test the parts that were never the
question.

The client is synchronous and the test server runs on this test's own
event loop, so every client call goes through ``asyncio.to_thread``.
Calling it inline blocks the loop that is supposed to be answering,
and the request times out against a server that was never given a
chance to reply.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer, unused_port

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service.fleet_control import down, fleet_snapshot, up
from tankpit_bot.service.fleet_manager import FleetManager
from tankpit_bot.service.fleet_routes import make_fleet_app
from tests.conftest import FakeEnv
from tests.service._fleet_fixtures import (
    _FakeProcess,
    _FakeSpawner,
    _restore_account_hooks,
    _with_configured_accounts,
)


def _closed_port() -> int:
    """Return a port in the valid range with nothing listening on it.

    Returns:
        A port number no server is bound to.
    """
    return unused_port()


class _Sleeps:
    """sleep_seconds double that records instead of waiting."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


@pytest.fixture()
async def live_fleet(spawner: _FakeSpawner) -> AsyncIterator[tuple[FleetManager, TestServer]]:
    """Serve a real fleet manager on a real loopback port.

    Yields:
        The manager and the server holding its port, so a test can
        take it off the air the way an exiting manager would.
    """
    _ = spawner
    manager = FleetManager()
    server = TestServer(make_fleet_app(manager), host="127.0.0.1")
    await server.start_server()
    yield manager, server
    await server.close()


def _port_of(server: TestServer) -> int:
    """Return a started test server's port.

    Args:
        server: A server that has already started.

    Returns:
        The bound port.

    Raises:
        AssertionError: If the server has not bound a port.
    """
    port = server.port
    if port is None:
        raise AssertionError("test server reported no port after starting")
    return port


def _point_at(port: int) -> None:
    """Resolve the fleet port to a chosen port.

    Args:
        port: Port the client should talk to.

    Returns:
        None.
    """
    top_hooks.get_env = FakeEnv({"TANKPIT_FLEET_PORT": str(port)})
    core_hooks.load_dotenv = lambda: None


def _spawn(manager: FleetManager, instance: str, account: str) -> None:
    """Register one bot on a manager.

    Args:
        manager: The registry to spawn into.
        instance: Instance name.
        account: Account selector.

    Returns:
        None.
    """
    originals = _with_configured_accounts()
    try:
        manager.spawn(instance=instance, account=account, kills=0, seconds=0)
    finally:
        _restore_account_hooks(originals)


class TestSnapshot:
    """Reading a live manager's state off the wire."""

    async def test_reads_boot_drain_state_and_rows(
        self,
        live_fleet: tuple[FleetManager, TestServer],
        restore_service_hooks: None,
    ) -> None:
        """The client decodes exactly what the manager serves."""
        _ = restore_service_hooks
        manager, server = live_fleet
        _spawn(manager, "alpha", "artax")

        snapshot = await asyncio.to_thread(fleet_snapshot, _port_of(server))

        if snapshot is None:
            raise AssertionError("the live manager reported no snapshot")
        assert snapshot["boot"] == manager.boot_id
        assert snapshot["draining"] is False
        assert [row["instance"] for row in snapshot["bots"]] == ["alpha"]

    def test_an_idle_port_reads_as_no_manager(self) -> None:
        """Nothing listening is an answer, not a failure."""
        assert fleet_snapshot(_closed_port()) is None


class TestUp:
    """Starting a manager, and not starting a second one."""

    async def test_a_manager_already_listening_is_left_alone(
        self,
        live_fleet: tuple[FleetManager, TestServer],
        restore_service_hooks: None,
    ) -> None:
        """``make up`` twice is a no-op, not a port-bind crash."""
        _ = restore_service_hooks
        manager, server = live_fleet
        _spawn(manager, "alpha", "artax")
        _point_at(_port_of(server))
        launches: list[str] = []

        def unexpected_launch() -> _FakeProcess:
            launches.append("launch")
            return _FakeProcess(pid=1)

        service_hooks.spawn_fleet_manager = unexpected_launch

        code = await asyncio.to_thread(up)

        assert code == 0
        assert launches == []

    async def test_launches_then_waits_until_the_manager_answers(
        self,
        restore_service_hooks: None,
        spawner: _FakeSpawner,
    ) -> None:
        """Nothing is listening; the launch is what makes that change.

        The fake launch really does start a manager on the port ``up``
        is watching, so the poll loop is exercised against a socket
        that genuinely was not there a moment earlier.
        """
        _ = (restore_service_hooks, spawner)
        loop = asyncio.get_running_loop()
        port = _closed_port()
        manager = FleetManager()
        server = TestServer(make_fleet_app(manager), host="127.0.0.1", port=port)
        sleeps = _Sleeps()
        service_hooks.sleep_seconds = sleeps

        def launch() -> _FakeProcess:
            asyncio.run_coroutine_threadsafe(server.start_server(), loop).result()
            return _FakeProcess(pid=4242)

        service_hooks.spawn_fleet_manager = launch
        _point_at(port)

        try:
            code = await asyncio.to_thread(up)
        finally:
            await server.close()

        assert code == 0
        assert sleeps.waits == [0.5]

    def test_a_manager_that_never_answers_is_reported_as_a_failure(
        self,
        restore_service_hooks: None,
    ) -> None:
        """A boot that fails exits non-zero instead of hanging."""
        _ = restore_service_hooks
        _point_at(_closed_port())
        sleeps = _Sleeps()
        service_hooks.sleep_seconds = sleeps
        service_hooks.spawn_fleet_manager = lambda: _FakeProcess(pid=4242)

        code = up(startup_timeout_s=1.5)

        assert code == 1
        assert sleeps.waits == [0.5, 0.5, 0.5]


class TestDown:
    """Draining a manager and waiting for it to go quiet."""

    def test_nothing_listening_is_success_not_an_error(
        self,
        restore_service_hooks: None,
    ) -> None:
        """Stopping an already-stopped fleet is a no-op."""
        _ = restore_service_hooks
        _point_at(_closed_port())

        assert down() == 0

    async def test_drains_every_bot_and_waits_for_the_manager_to_exit(
        self,
        live_fleet: tuple[FleetManager, TestServer],
        restore_service_hooks: None,
    ) -> None:
        """The real POST reaches the real manager and starts the drain.

        The client is then pointed at a dead port, standing in for the
        manager exiting once its last bot had landed.
        """
        _ = restore_service_hooks
        manager, server = live_fleet
        loop = asyncio.get_running_loop()
        _spawn(manager, "alpha", "artax")
        _spawn(manager, "bravo", "second")
        _point_at(_port_of(server))

        stops: list[str] = []
        polls = 0

        def record_stop(path: Path, content: str) -> None:
            _ = content
            stops.append(str(path).replace("\\", "/"))

        def sleep_then_land(seconds: float) -> None:
            nonlocal polls
            _ = seconds
            polls += 1
            if polls == 3:
                # Both bots have torn down, so the manager exits and
                # its port goes quiet -- which is the only signal the
                # client waits on. Poll 2 sees the SAME bots still
                # draining, which is the quiet path: report once, then
                # keep waiting without repeating yourself.
                asyncio.run_coroutine_threadsafe(server.close(), loop).result()

        original_write = top_hooks.write_text
        top_hooks.write_text = record_stop
        service_hooks.sleep_seconds = sleep_then_land
        try:
            code = await asyncio.to_thread(down)
        finally:
            top_hooks.write_text = original_write

        assert code == 0
        assert stops == ["runs/bot/alpha/STOP", "runs/bot/bravo/STOP"]
        assert manager.draining() is True
        assert polls == 3

    async def test_a_manager_with_no_bots_still_drains_and_exits(
        self,
        live_fleet: tuple[FleetManager, TestServer],
        restore_service_hooks: None,
    ) -> None:
        """An empty fleet needs no teardown, only the shutdown."""
        _ = restore_service_hooks
        manager, server = live_fleet
        loop = asyncio.get_running_loop()
        _point_at(_port_of(server))

        def land_immediately(seconds: float) -> None:
            _ = seconds
            asyncio.run_coroutine_threadsafe(server.close(), loop).result()

        service_hooks.sleep_seconds = land_immediately

        code = await asyncio.to_thread(down)

        assert code == 0
        assert manager.draining() is True

    async def test_a_stranger_on_the_fleet_port_is_surfaced(
        self,
        restore_service_hooks: None,
    ) -> None:
        """Something answering that is not the fleet is a misconfiguration."""
        _ = restore_service_hooks

        async def refuse(request: web.Request) -> web.Response:
            _ = request
            return web.Response(status=503, text="not the fleet")

        app = web.Application()
        app.router.add_get("/bots", refuse)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            _point_at(_port_of(server))
            with pytest.raises(RuntimeError, match="503"):
                await asyncio.to_thread(down)
        finally:
            await server.close()
