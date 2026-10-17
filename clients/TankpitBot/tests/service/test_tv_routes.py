"""The operator's TV routes on the real fleet app.

Cast and stop send real HTTP to the loopback renderer of
:mod:`tests.service._tv_server`. What is pinned: every TV outcome,
from playing to a TV that is switched off, reaches the operator as its
own status and code, and the status route reads the director without
asking the TV anything.
"""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from platform_core.json_utils import load_json_str, narrow_json_to_dict, require_dict, require_str

from tankpit_bot.service.fleet_manager import FleetManager
from tankpit_bot.service.fleet_routes import make_fleet_app
from tankpit_bot.service.mosaic_director import MosaicDirector
from tankpit_bot.service.tv_routes import add_tv_routes
from tests.conftest import FakeEnv
from tests.service._fleet_fixtures import (
    _FakeSpawner,
    _restore_account_hooks,
    _with_account_pool,
)
from tests.service._tv_server import TvServer, fault_envelope


@pytest.fixture()
async def client(
    spawner: _FakeSpawner, tmp_path: Path
) -> AsyncIterator[TestClient[web.Request, web.Application]]:
    """Serve the fleet app with the TV routes added.

    Yields:
        A client on the real routes.
    """
    _ = spawner
    originals = _with_account_pool("alpha")
    director = MosaicDirector(FleetManager(), Path(os.path.relpath(tmp_path / "mosaic")))
    app = make_fleet_app(FleetManager())
    add_tv_routes(app, director)
    test_client: TestClient[web.Request, web.Application] = TestClient(TestServer(app))
    await test_client.start_server()
    yield test_client
    await test_client.close()
    _restore_account_hooks(originals)


def _aim(env: FakeEnv, control_url: str) -> None:
    """Point the fleet at a TV.

    Args:
        env: The environment seam.
        control_url: The TV's control URL.
    """
    env.set("TANKPIT_TV_CONTROL_URL", control_url)
    env.set("TANKPIT_MOSAIC_URL", "https://tankpit.example/demo/mosaic/index.m3u8")


@pytest.mark.asyncio
async def test_the_status_says_what_the_stream_shows(
    client: TestClient[web.Request, web.Application],
) -> None:
    """Before the first tick: four waiting tiles, no encoder, no fault."""
    response = await client.get("/tv")
    body = narrow_json_to_dict(load_json_str(await response.text()))

    assert response.status == 200
    assert require_dict(body, "mosaic") == {
        "tiles": ["", "", "", ""],
        "running": False,
        "failure": "",
    }


@pytest.mark.asyncio
async def test_cast_answers_with_what_the_tv_reports(
    client: TestClient[web.Request, web.Application], fake_env: FakeEnv
) -> None:
    """The TV's own transport state, and the URL it was given."""
    with TvServer() as server:
        server.answer_state("PLAYING")
        _aim(fake_env, server.control_url)
        response = await client.post("/tv/cast")
        body = narrow_json_to_dict(load_json_str(await response.text()))

    assert response.status == 200
    assert require_str(body, "transport_state") == "PLAYING"
    assert require_str(body, "media_url") == "https://tankpit.example/demo/mosaic/index.m3u8"


@pytest.mark.asyncio
async def test_stop_answers_with_what_the_tv_reports(
    client: TestClient[web.Request, web.Application], fake_env: FakeEnv
) -> None:
    """Stopped, as the TV says."""
    with TvServer() as server:
        server.answer_state("STOPPED")
        _aim(fake_env, server.control_url)
        response = await client.post("/tv/stop")
        body = narrow_json_to_dict(load_json_str(await response.text()))

    assert response.status == 200
    assert require_str(body, "transport_state") == "STOPPED"


@pytest.mark.asyncio
async def test_no_tv_configured_is_a_conflict(
    client: TestClient[web.Request, web.Application], fake_env: FakeEnv
) -> None:
    """Unset addresses answer 409 with the code."""
    _ = fake_env
    response = await client.post("/tv/cast")
    assert response.status == 409
    assert (await response.text()).startswith("TV_NOT_CONFIGURED: ")


@pytest.mark.asyncio
async def test_a_tv_fault_is_a_bad_gateway(
    client: TestClient[web.Request, web.Application], fake_env: FakeEnv
) -> None:
    """A refusal answers 502 with the renderer's own error."""
    with TvServer() as server:
        server.answers["Play"] = (500, fault_envelope(701, "Transition not available"))
        _aim(fake_env, server.control_url)
        response = await client.post("/tv/cast")
        text = await response.text()

    assert response.status == 502
    assert text.startswith("TV_REFUSED: Play answered HTTP 500 (UPnP error 701")


@pytest.mark.asyncio
async def test_a_tv_that_is_off_is_a_gateway_timeout(
    client: TestClient[web.Request, web.Application], fake_env: FakeEnv
) -> None:
    """Nothing listening answers 504 and says what to check."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
    _aim(fake_env, f"http://127.0.0.1:{port}/upnp/control/AVTransport1")
    response = await client.post("/tv/stop")
    text = await response.text()

    assert response.status == 504
    assert text.startswith("TV_UNREACHABLE: the TV did not answer")
    assert text.endswith("is it switched on?")


@pytest.mark.asyncio
async def test_an_answer_that_is_not_xml_is_unreadable(
    client: TestClient[web.Request, web.Application], fake_env: FakeEnv
) -> None:
    """A web page where a fault belongs answers 502 TV_UNREADABLE."""
    with TvServer() as server:
        server.answers["Play"] = (404, "<html><body>Not Found")
        _aim(fake_env, server.control_url)
        response = await client.post("/tv/cast")
        text = await response.text()

    assert response.status == 502
    assert text.startswith("TV_UNREADABLE: the TV's answer to cast is not XML")
