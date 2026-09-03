"""Tests for the fleet video relay.

The manager serves one published port, so a child's video reaches a
caller by being relayed rather than by handing out the child's own
address. These drive the real route, the real ``StreamResponse``
writing and the real teardown; only the upstream is supplied by the
test, through the same hook production assigns at import.

The two properties worth pinning are that the upstream ``Content-Type``
survives the relay unaltered -- it carries the multipart boundary, and
a caller given a different token cannot split the frames -- and that
the upstream is released on every exit, including the one where the
caller vanishes mid-frame.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_str,
    require_int,
    require_list,
)

from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service._test_hooks import ChildVideoStreamProtocol, OpenChildVideoProtocol
from tests.service._fleet_fixtures import _FakeSpawner

# A boundary token that is obviously the child's own, so a test failure
# reading "boundary mismatch" cannot be confused with a default.
CHILD_CONTENT_TYPE = "multipart/x-mixed-replace; boundary=childframe42"


class _FakeChildVideoStream:
    """A :class:`ChildVideoStreamProtocol` over a fixed chunk list."""

    def __init__(self, chunks: list[bytes]) -> None:
        """Bind the stream to the bytes it will yield.

        Args:
            chunks: Body chunks to yield in order.
        """
        self._chunks = chunks
        self.closes = 0

    @property
    def content_type(self) -> str:
        """The upstream content type.

        Returns:
            A multipart type carrying the child's own boundary.
        """
        return CHILD_CONTENT_TYPE

    async def chunks(self) -> AsyncIterator[bytes]:
        """Yield the bound chunks.

        Yields:
            Each chunk in order.
        """
        for chunk in self._chunks:
            yield chunk

    async def close(self) -> None:
        """Record one release."""
        self.closes += 1


class _FailingChildVideoStream:
    """A stream that dies partway through, as a dropped upstream does."""

    def __init__(self) -> None:
        """Start with nothing released."""
        self.closes = 0

    @property
    def content_type(self) -> str:
        """The upstream content type.

        Returns:
            A multipart type carrying the child's own boundary.
        """
        return CHILD_CONTENT_TYPE

    async def chunks(self) -> AsyncIterator[bytes]:
        """Yield one chunk, then fail.

        Yields:
            One chunk before raising.

        Raises:
            ConnectionResetError: Always, after the first chunk.
        """
        yield b"first"
        raise ConnectionResetError("upstream went away")

    async def close(self) -> None:
        """Record one release."""
        self.closes += 1


class _RecordingOpener:
    """An :class:`OpenChildVideoProtocol` recording the URLs it opened."""

    def __init__(self, stream: ChildVideoStreamProtocol) -> None:
        """Bind the opener to the stream it hands back.

        Args:
            stream: Stream returned for every call.
        """
        self._stream = stream
        self.urls: list[str] = []

    async def __call__(self, url: str) -> ChildVideoStreamProtocol:
        """Record the URL and return the bound stream.

        Args:
            url: Upstream URL the route asked for.

        Returns:
            The bound stream.
        """
        self.urls.append(url)
        return self._stream


async def _spawn(
    client: TestClient[web.Request, web.Application],
    instance: str,
    account: str = "",
) -> int:
    """Spawn one bot and return the port the manager allocated it.

    Args:
        client: The fleet test client.
        instance: Instance name to spawn.
        account: Account selector; empty uses the configured default.

    Returns:
        The allocated service port.
    """
    payload: dict[str, str | int] = {"instance": instance, "kills": 5}
    if account:
        payload["account"] = account
    response = await client.post("/bots", json=payload)
    assert response.status == 201
    body = narrow_json_to_dict(load_json_str(await response.text()))
    return require_int(body, "service_port")


async def _second_account(client: TestClient[web.Request, web.Application]) -> str:
    """Return a configured account other than the default.

    Asked for rather than hardcoded: the manager refuses a second live
    bot on one account because the game refuses a second login, so a
    two-bot test needs a real second account and the configured set is
    the only authority on what that is.

    Args:
        client: The fleet test client.

    Returns:
        The second configured account name.
    """
    response = await client.get("/accounts")
    assert response.status == 200
    body = narrow_json_to_dict(load_json_str(await response.text()))
    accounts = require_list(body, "accounts")
    if len(accounts) < 2:
        raise AssertionError("this test needs two configured accounts")
    return narrow_json_to_str(accounts[1])


@pytest.mark.asyncio
async def test_video_relays_the_child_bytes_and_its_content_type(
    fleet_client: TestClient[web.Request, web.Application],
) -> None:
    """The caller receives the child's bytes under the child's own type."""
    port = await _spawn(fleet_client, "alpha")
    stream = _FakeChildVideoStream([b"aaa", b"bbb", b"ccc"])
    opener = _RecordingOpener(stream)
    original: OpenChildVideoProtocol = service_hooks.open_child_video
    service_hooks.open_child_video = opener
    try:
        response = await fleet_client.get("/bots/alpha/video")
        body = await response.read()
    finally:
        service_hooks.open_child_video = original

    assert response.status == 200
    assert body == b"aaabbbccc"
    assert response.headers["Content-Type"] == CHILD_CONTENT_TYPE
    assert opener.urls == [f"http://127.0.0.1:{port}/video"]


@pytest.mark.asyncio
async def test_video_releases_the_upstream_when_the_relay_finishes(
    fleet_client: TestClient[web.Request, web.Application],
) -> None:
    """A completed relay closes the upstream exactly once."""
    await _spawn(fleet_client, "alpha")
    stream = _FakeChildVideoStream([b"aaa"])
    original: OpenChildVideoProtocol = service_hooks.open_child_video
    service_hooks.open_child_video = _RecordingOpener(stream)
    try:
        response = await fleet_client.get("/bots/alpha/video")
        await response.read()
    finally:
        service_hooks.open_child_video = original

    assert stream.closes == 1


@pytest.mark.asyncio
async def test_video_releases_the_upstream_when_it_dies_mid_stream(
    fleet_client: TestClient[web.Request, web.Application],
) -> None:
    """An upstream that fails partway is still released.

    The connection is the resource; whether the stream ended tidily is
    beside the point. An unreleased upstream would hold a connection
    against the child for the life of the manager.

    The caller sees a torn connection rather than a status code, and
    that is correct: the response was committed the moment the first
    chunk went out, so there is no longer a status left to change. The
    failure is NOT softened into a tidy error page.
    """
    await _spawn(fleet_client, "alpha")
    stream = _FailingChildVideoStream()
    original: OpenChildVideoProtocol = service_hooks.open_child_video
    service_hooks.open_child_video = _RecordingOpener(stream)
    try:
        with pytest.raises(aiohttp.ClientError):
            response = await fleet_client.get("/bots/alpha/video")
            await response.read()
    finally:
        service_hooks.open_child_video = original

    assert stream.closes == 1


@pytest.mark.asyncio
async def test_video_for_an_unknown_instance_is_refused(
    fleet_client: TestClient[web.Request, web.Application],
) -> None:
    """An instance the manager never spawned gets a 404, not a relay."""
    response = await fleet_client.get("/bots/nosuchbot/video")

    assert response.status == 404
    assert "unknown instance" in await response.text()


@pytest.mark.asyncio
async def test_video_for_a_dead_instance_is_refused(
    fleet_client: TestClient[web.Request, web.Application],
    spawner: _FakeSpawner,
) -> None:
    """A dead bot is refused because its port may already be someone else's.

    Ports return to the pool the moment their holder dies, so relaying
    on the strength of a stale row could serve a different bot's video
    under this instance's name. The refusal happens before any upstream
    is opened -- the hook is left as production assigned it, so a route
    that tried to connect would fail loudly here rather than pass.
    """
    await _spawn(fleet_client, "alpha")
    spawner.processes[0].returncode = 0

    response = await fleet_client.get("/bots/alpha/video")

    assert response.status == 404
    assert "not running" in await response.text()


@pytest.mark.asyncio
async def test_a_spawned_child_is_given_its_own_service_port(
    fleet_client: TestClient[web.Request, web.Application],
    spawner: _FakeSpawner,
) -> None:
    """The child environment names the port its video will be served on.

    The port is what makes a child reachable at all; without it every
    child would bind the same default and the relay would serve
    whichever answered first.
    """
    port = await _spawn(fleet_client, "alpha")

    assert spawner.envs[0]["TANKPIT_BOT_SERVICE_PORT"] == str(port)


@pytest.mark.asyncio
async def test_two_live_children_never_share_a_port(
    fleet_client: TestClient[web.Request, web.Application],
) -> None:
    """Concurrent children get distinct ports.

    Two bots on one port would serve each other's video, which is the
    confusion the allocator exists to prevent.
    """
    first = await _spawn(fleet_client, "alpha")
    second = await _spawn(fleet_client, "bravo", await _second_account(fleet_client))

    assert first != second


@pytest.mark.asyncio
async def test_a_dead_child_returns_its_port_to_the_pool(
    fleet_client: TestClient[web.Request, web.Application],
    spawner: _FakeSpawner,
) -> None:
    """A finished bot's port is reused rather than retired.

    Only live children reserve a port; otherwise a long-lived fleet
    would march through the range and exhaust it while running two bots.
    """
    first = await _spawn(fleet_client, "alpha")
    spawner.processes[0].returncode = 0
    second = await _spawn(fleet_client, "bravo")

    assert second == first
