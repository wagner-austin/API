"""The operator's per-instance video files, served off the shared disk.

The demo suite (:mod:`tests.service.test_demo`) exercises the slot
grammar and the status semantics of :func:`read_hls_file`; this file
pins what is specific to the OPERATOR surface — any registered
instance may be watched, a dead one may not, and the file grammar
holds on this route too.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Generator
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.service.fleet_manager import FleetManager
from tankpit_bot.service.fleet_routes import make_fleet_app
from tests.service._fleet_fixtures import (
    _FakeSpawner,
    _restore_account_hooks,
    _with_account_pool,
)

_SPAWN: dict[str, str | int] = {"instance": "artax", "account": "alpha"}


@pytest.fixture()
def one_account() -> Generator[None, None, None]:
    """Configure a one-account machine for the duration of one test.

    Yields:
        Nothing — the fixture exists for its effect on the account
        config seams.
    """
    originals = _with_account_pool("alpha")
    yield
    _restore_account_hooks(originals)


@pytest.fixture()
async def fleet_client(
    spawner: _FakeSpawner,
    one_account: None,
) -> AsyncIterator[tuple[TestClient[web.Request, web.Application], _FakeSpawner]]:
    """Serve the fleet app, exposing the spawner for liveness control.

    Yields:
        The client and the spawner whose processes a test may end.
    """
    _ = one_account
    manager = FleetManager()
    test_client: TestClient[web.Request, web.Application] = TestClient(
        TestServer(make_fleet_app(manager))
    )
    await test_client.start_server()
    yield test_client, spawner
    await test_client.close()


class _FakeHlsFiles:
    """``read_bytes_from`` over an in-memory directory of HLS files."""

    def __init__(self, files: dict[str, bytes]) -> None:
        """Bind the fake to its file contents.

        Args:
            files: Path-string to bytes; anything else is absent.
        """
        self._files = files

    def __call__(self, path: Path, offset: int) -> bytes:
        """Serve one read the way the real hook does.

        Args:
            path: File the caller asked for.
            offset: Byte offset to start at.

        Returns:
            The bytes from ``offset`` on.

        Raises:
            FileNotFoundError: The path is not in the directory.
        """
        key = str(path)
        if key not in self._files:
            raise FileNotFoundError(key)
        return self._files[key][offset:]


@pytest.mark.asyncio
async def test_a_running_instance_serves_its_playlist(
    fleet_client: tuple[TestClient[web.Request, web.Application], _FakeSpawner],
) -> None:
    """The operator watches any registered live bot by name."""
    client, _spawner = fleet_client
    assert (await client.post("/bots", json=_SPAWN)).status == 201
    playlist = b"#EXTM3U\n#EXT-X-TARGETDURATION:2\n"
    original = top_hooks.read_bytes_from
    top_hooks.read_bytes_from = _FakeHlsFiles(
        {str(Path("runs/bot/artax/hls/index.m3u8")): playlist}
    )
    try:
        response = await client.get("/bots/artax/video/index.m3u8")
        body = await response.read()
    finally:
        top_hooks.read_bytes_from = original

    assert response.status == 200
    assert body == playlist


@pytest.mark.asyncio
async def test_an_unknown_instance_is_404(
    fleet_client: tuple[TestClient[web.Request, web.Application], _FakeSpawner],
) -> None:
    """A name the registry never issued has no picture."""
    client, _spawner = fleet_client
    response = await client.get("/bots/nobody/video/index.m3u8")
    assert response.status == 404
    assert "unknown instance" in await response.text()


@pytest.mark.asyncio
async def test_a_finished_instance_is_404_not_stale_video(
    fleet_client: tuple[TestClient[web.Request, web.Application], _FakeSpawner],
) -> None:
    """A dead bot's leftover segments are not served under its name.

    The name may be reissued to a fresh spawn moments later; a viewer
    handed the old files would be watching the wrong session and
    nothing would say so.
    """
    client, spawner = fleet_client
    assert (await client.post("/bots", json=_SPAWN)).status == 201
    spawner.processes[0].returncode = 0

    response = await client.get("/bots/artax/video/index.m3u8")

    assert response.status == 404
    assert "is not running" in await response.text()
