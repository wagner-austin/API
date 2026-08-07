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

from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service.frame_bus import FrameBus
from tankpit_bot.service.http_server import make_app
from tankpit_bot.service.mode_bridge import ModeBridge
from tankpit_bot.service.status_bus import StatusBus
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
    original_load_dotenv = service_hooks.load_dotenv
    original_serve = service_hooks.serve
    original_build_bot_factory = service_hooks.build_bot_factory
    original_probe_existing_instance = service_hooks.probe_existing_instance
    try:
        yield
    finally:
        service_hooks.build_site = original_build_site
        service_hooks.load_dotenv = original_load_dotenv
        service_hooks.serve = original_serve
        service_hooks.build_bot_factory = original_build_bot_factory
        service_hooks.probe_existing_instance = original_probe_existing_instance
