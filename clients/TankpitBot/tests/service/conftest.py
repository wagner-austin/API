"""Shared fixtures for the service test modules.

``test_http_server.py`` was 1,080 lines and is now three modules; these
five fixtures are shared by all of them. They live here rather than in
:mod:`tests.service._http_fixtures` because a pytest fixture cannot
travel by import without becoming an unused-name violation at every
call site.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Generator

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.bus.frame_bus import FrameBus
from tankpit_bot.bus.mode_bridge import ModeBridge
from tankpit_bot.bus.status_bus import StatusBus
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service.fleet_manager import FleetManager
from tankpit_bot.service.fleet_routes import make_fleet_app
from tankpit_bot.service.http_server import make_app
from tests.service._fleet_fixtures import _FakeSpawner
from tests.service._http_fixtures import _noop_shutdown, _RecordingRunner


@pytest.fixture()
async def bus() -> StatusBus:
    """Fresh :class:`StatusBus` per test."""
    return StatusBus()


@pytest.fixture()
async def fbus() -> FrameBus:
    """Fresh :class:`FrameBus` per test."""
    return FrameBus()


@pytest.fixture()
async def bridge() -> ModeBridge:
    """Fresh :class:`ModeBridge` per test."""
    return ModeBridge()


@pytest.fixture()
async def runner() -> _RecordingRunner:
    """Recording runner in the idle-not-rejecting default."""
    return _RecordingRunner()


@pytest.fixture()
async def client(
    runner: _RecordingRunner,
    bridge: ModeBridge,
    bus: StatusBus,
    fbus: FrameBus,
) -> AsyncIterator[TestClient[web.Request, web.Application]]:
    """aiohttp TestClient bound to a real app."""
    app = make_app(runner, bridge, bus, fbus, _noop_shutdown)
    server = TestServer(app)
    async with TestClient(server) as tc:
        yield tc


@pytest.fixture()
def restore_service_hooks() -> Generator[None, None, None]:
    """Snapshot + restore ``service._test_hooks`` symbols around a test.

    Yields:
        Nothing — the fixture exists solely for its side-effect on the
        module-level hook symbols.
    """
    original_build_site = service_hooks.build_site
    original_load_dotenv = core_hooks.load_dotenv
    original_serve = service_hooks.serve
    original_build_bot_factory = service_hooks.build_bot_factory
    original_probe_existing_instance = service_hooks.probe_existing_instance
    try:
        yield
    finally:
        service_hooks.build_site = original_build_site
        core_hooks.load_dotenv = original_load_dotenv
        service_hooks.serve = original_serve
        service_hooks.build_bot_factory = original_build_bot_factory
        service_hooks.probe_existing_instance = original_probe_existing_instance


@pytest.fixture()
def spawner() -> Generator[_FakeSpawner, None, None]:
    """Install a recording spawner for the duration of one test."""
    original = service_hooks.spawn_bot_process
    fake = _FakeSpawner()
    service_hooks.spawn_bot_process = fake
    yield fake
    service_hooks.spawn_bot_process = original


@pytest.fixture()
async def fleet_client(
    spawner: _FakeSpawner,
) -> AsyncIterator[TestClient[web.Request, web.Application]]:
    """Serve the FLEET app on a random test port.

    Distinct from ``client``, which serves the per-bot HTTP
    server; both live here because a pytest fixture cannot
    travel by import.
    """
    _ = spawner
    manager = FleetManager()
    app = make_fleet_app(manager)
    test_client: TestClient[web.Request, web.Application] = TestClient(TestServer(app))
    await test_client.start_server()
    yield test_client
    await test_client.close()
