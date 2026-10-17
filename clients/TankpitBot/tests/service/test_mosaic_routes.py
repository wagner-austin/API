"""The four-up stream's public route on the real fleet app.

Served from real files in the director's directory, with the HLS
semantics the per-bot streams have: warming before the first playlist,
the two file shapes and their media types, and nothing else.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Generator
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tankpit_bot.service.fleet_manager import FleetManager
from tankpit_bot.service.fleet_routes import make_fleet_app
from tankpit_bot.service.mosaic_director import MosaicDirector
from tankpit_bot.service.mosaic_routes import add_mosaic_route
from tankpit_bot.stream.hls import PLAYLIST_CONTENT_TYPE, SEGMENT_CONTENT_TYPE
from tests.service._fleet_fixtures import (
    _FakeSpawner,
    _restore_account_hooks,
    _with_account_pool,
)


@pytest.fixture()
def director(spawner: _FakeSpawner, tmp_path: Path) -> Generator[MosaicDirector, None, None]:
    """A director over an empty registry, rooted in the test's tree.

    Yields:
        The director; its encoder is never started here.
    """
    _ = spawner
    originals = _with_account_pool("alpha")
    yield MosaicDirector(FleetManager(), Path(os.path.relpath(tmp_path / "mosaic")))
    _restore_account_hooks(originals)


@pytest.fixture()
async def client(
    director: MosaicDirector,
) -> AsyncIterator[TestClient[web.Request, web.Application]]:
    """Serve the fleet app with the four-up route added.

    Yields:
        A client on the real routes.
    """
    app = make_fleet_app(FleetManager())
    add_mosaic_route(app, director)
    test_client: TestClient[web.Request, web.Application] = TestClient(TestServer(app))
    await test_client.start_server()
    yield test_client
    await test_client.close()


@pytest.mark.asyncio
async def test_before_the_first_playlist_the_stream_is_warming(
    client: TestClient[web.Request, web.Application],
) -> None:
    """No playlist yet is a 503 with a retry, as for a bot's own stream."""
    response = await client.get("/demo/mosaic/index.m3u8")
    assert response.status == 503
    assert response.headers["Retry-After"]


@pytest.mark.asyncio
async def test_the_playlist_and_a_segment_are_served(
    client: TestClient[web.Request, web.Application], director: MosaicDirector
) -> None:
    """Real files off the director's directory, with their media types."""
    director.hls_dir.mkdir(parents=True)
    (director.hls_dir / "index.m3u8").write_text("#EXTM3U\n", encoding="utf-8")
    (director.hls_dir / "seg00007.ts").write_bytes(b"\x47" * 188)

    playlist = await client.get("/demo/mosaic/index.m3u8")
    segment = await client.get("/demo/mosaic/seg00007.ts")

    assert playlist.status == 200
    assert playlist.headers["Content-Type"].startswith(PLAYLIST_CONTENT_TYPE)
    assert await playlist.text() == "#EXTM3U\n"
    assert segment.status == 200
    assert segment.headers["Content-Type"].startswith(SEGMENT_CONTENT_TYPE)
    assert await segment.read() == b"\x47" * 188


@pytest.mark.asyncio
async def test_a_name_outside_the_grammar_is_not_found(
    client: TestClient[web.Request, web.Application],
) -> None:
    """Only the playlist and segment names are files; the log is not one."""
    response = await client.get("/demo/mosaic/ffmpeg.log")
    assert response.status == 404
