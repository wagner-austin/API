"""The bot service's own video surface: HLS files and the watch page."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tankpit_bot.bus.mode_bridge import ModeBridge
from tankpit_bot.bus.status_bus import StatusBus
from tankpit_bot.service.http_server import make_app
from tests.service._http_fixtures import _noop_shutdown, _RecordingRunner


@pytest.fixture()
async def streaming_client(
    tmp_path: Path,
) -> AsyncIterator[tuple[TestClient[web.Request, web.Application], Path]]:
    """A client whose app serves a real HLS directory.

    Yields:
        The client and the directory its ``/video`` reads from.
    """
    hls_dir = tmp_path / "hls"
    hls_dir.mkdir()
    app = make_app(_RecordingRunner(), ModeBridge(), StatusBus(), hls_dir, _noop_shutdown)
    server = TestServer(app)
    async with TestClient(server) as tc:
        yield tc, hls_dir


@pytest.mark.asyncio
async def test_a_session_without_a_stream_says_so(
    client: TestClient[web.Request, web.Application],
) -> None:
    """No stream directory means an honest 404, not an eternal 503."""
    response = await client.get("/video/index.m3u8")
    assert response.status == 404
    assert "this session has no stream" in await response.text()


@pytest.mark.asyncio
async def test_the_playlist_is_served_from_the_capture_directory(
    streaming_client: tuple[TestClient[web.Request, web.Application], Path],
) -> None:
    """Bytes the encoder wrote come back under the playlist type."""
    tc, hls_dir = streaming_client
    (hls_dir / "index.m3u8").write_bytes(b"#EXTM3U\n")

    response = await tc.get("/video/index.m3u8")

    assert response.status == 200
    assert response.headers["Content-Type"].startswith("application/vnd.apple.mpegurl")
    assert await response.read() == b"#EXTM3U\n"


@pytest.mark.asyncio
async def test_an_empty_capture_directory_is_warming(
    streaming_client: tuple[TestClient[web.Request, web.Application], Path],
) -> None:
    """Before the first playlist lands, the answer is come-back."""
    tc, _hls_dir = streaming_client
    response = await tc.get("/video/index.m3u8")
    assert response.status == 503
    assert response.headers["Retry-After"] == "3"


@pytest.mark.asyncio
async def test_the_watch_page_plays_hls_both_ways(
    client: TestClient[web.Request, web.Application],
) -> None:
    """The page carries native playback AND the hls.js path.

    Both are required to cover real devices: iOS Safari has native
    HLS and (before 17.1) no MSE, Chrome has MSE and no native HLS.
    """
    response = await client.get("/watch")
    page = await response.text()
    assert response.status == 200
    assert 'src="watch/hls.js"' in page
    assert 'HLS_URL = "video/index.m3u8"' in page
    assert "canPlayType" in page
    assert "new Hls()" in page


@pytest.mark.asyncio
async def test_the_vendored_hls_js_is_served(
    client: TestClient[web.Request, web.Application],
) -> None:
    """The wheel's own hls.js build answers, so the page needs no CDN."""
    response = await client.get("/watch/hls.js")
    body = await response.read()
    assert response.status == 200
    assert response.headers["Content-Type"].startswith("application/javascript")
    # Real library bytes, not a placeholder: the UMD banner and size.
    assert body.startswith(b"!function")
    assert len(body) > 100_000
