"""Tests for the production child-video hook, against a real server.

The relay route is tested with a stream the test constructs; this file
tests the thing that constructs one for real. It runs an aiohttp server
on loopback and reads it over HTTP, so the session, the response, the
chunked read and the release all execute — the parts a substituted
stream would skip.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from aiohttp import web
from aiohttp.client_exceptions import ClientConnectionError, ClientResponseError
from aiohttp.test_utils import TestServer

from tankpit_bot.service._test_hooks import _real_open_child_video

# A boundary token the test server owns, so an assertion that it
# survived the read cannot be satisfied by a default.
UPSTREAM_CONTENT_TYPE = "multipart/x-mixed-replace; boundary=upstream77"

# Chunks the fake child writes, deliberately more than one so the
# chunked read is exercised rather than a single-shot body.
UPSTREAM_CHUNKS = (b"frame-one", b"frame-two", b"frame-three")

FAILING_PATH = "/broken"


async def _video(request: web.Request) -> web.StreamResponse:
    """Serve a multipart stream in several chunks.

    Args:
        request: The incoming request.

    Returns:
        The finished streaming response.
    """
    response = web.StreamResponse(status=200, headers={"Content-Type": UPSTREAM_CONTENT_TYPE})
    await response.prepare(request)
    for chunk in UPSTREAM_CHUNKS:
        await response.write(chunk)
    return response


async def _broken(request: web.Request) -> web.Response:
    """Answer with a server error.

    Args:
        request: The incoming request.

    Returns:
        A 500 response.
    """
    _ = request
    return web.Response(status=500, text="child is unwell")


@pytest.fixture()
async def upstream() -> AsyncIterator[TestServer]:
    """Run a stand-in child service on loopback.

    Yields:
        The running server.
    """
    app = web.Application()
    app.router.add_get("/video", _video)
    app.router.add_get(FAILING_PATH, _broken)
    server = TestServer(app)
    await server.start_server()
    yield server
    await server.close()


@pytest.mark.asyncio
async def test_the_real_hook_reads_a_child_stream_end_to_end(upstream: TestServer) -> None:
    """Every chunk arrives, in order, over a real connection."""
    stream = await _real_open_child_video(str(upstream.make_url("/video")))
    try:
        received = [chunk async for chunk in stream.chunks()]
    finally:
        await stream.close()

    assert b"".join(received) == b"".join(UPSTREAM_CHUNKS)


@pytest.mark.asyncio
async def test_the_real_hook_carries_the_upstream_content_type(upstream: TestServer) -> None:
    """The child's boundary token survives the read.

    Reconstructing the type here would hand the caller a boundary the
    child never used, and the caller could not split the frames.
    """
    stream = await _real_open_child_video(str(upstream.make_url("/video")))
    try:
        content_type = stream.content_type
    finally:
        await stream.close()

    assert content_type == UPSTREAM_CONTENT_TYPE


@pytest.mark.asyncio
async def test_a_child_answering_with_an_error_status_is_not_relayed(
    upstream: TestServer,
) -> None:
    """An unhealthy child raises rather than yielding an empty stream.

    Relaying a 500 body as though it were video would put a page-shaped
    payload behind a multipart content type, which reads as a silent
    black frame instead of a fault.
    """
    with pytest.raises(ClientResponseError):
        await _real_open_child_video(str(upstream.make_url(FAILING_PATH)))


@pytest.mark.asyncio
async def test_an_unreachable_child_raises(upstream: TestServer) -> None:
    """A child that is not listening is a connection error, not a hang."""
    port = upstream.port
    if port is None:
        raise AssertionError("the fixture server must be bound before it is closed")
    await upstream.close()

    with pytest.raises(ClientConnectionError):
        await _real_open_child_video(f"http://127.0.0.1:{port}/video")
