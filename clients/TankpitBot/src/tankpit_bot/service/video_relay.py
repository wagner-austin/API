"""Relaying one fleet child's MJPEG stream through the manager.

A child serves its own video on a port inside the manager's container.
Nothing outside can dial that port and nothing should: publishing one
port per bot would expose the fleet's internal surface in order to show
a picture. The manager already answers on the one published port, so it
relays bytes rather than handing out addresses.

Its own module because two route groups relay the same way and must not
drift: the operator surface serves any registered instance
(:mod:`tankpit_bot.service.fleet_routes`), the public demo serves only a
demo slot (:mod:`tankpit_bot.service.demo_routes`). What differs between
them is WHICH instance they will resolve, which each decides before
calling here; how the bytes move is one answer.
"""

from __future__ import annotations

from aiohttp import web
from aiohttp.client_exceptions import ClientConnectionError
from platform_core.logging import get_logger

from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service.constants import child_video_url
from tankpit_bot.service.fleet_error import FleetError
from tankpit_bot.service.fleet_manager import FleetManager

log = get_logger(__name__)

CHILD_WARMUP_RETRY_SECONDS = 3
"""How long a caller is told to wait for a child that is still booting.

A child registers the moment it is forked and reports ``alive`` from
that instant, but its own HTTP service does not bind until Playwright
has a browser up — on the order of fifteen seconds. Three seconds is
short enough that a watcher polling on this reaches the picture within
a few tries of the port opening, and long enough that the poll is not a
busy-wait against a process that is doing real work.
"""


async def relay_child_video(
    request: web.Request, manager: FleetManager, instance: str
) -> web.StreamResponse:
    """Stream one running child's video to the caller.

    The upstream ``Content-Type`` is passed through untouched because it
    carries the multipart boundary the child generated, and a caller
    given a different token cannot split the frames.

    The upstream is released on BOTH exits. A viewer closing a tab
    mid-frame is the ordinary case here rather than an exceptional one,
    and a stream left open would hold a connection against the child for
    as long as this manager lives.

    Args:
        request: The caller's request, prepared as the response stream.
        manager: The registry the child's port is resolved through.
        instance: The instance to relay. The caller has already decided
            this name is one it is willing to serve.

    Returns:
        The streamed response; a 404 when the instance is not
        registered or is no longer running; a 503 while the child is
        registered but its service has not bound its port yet.

        Liveness matters for the 404 because a finished bot's port
        returns to the allocator immediately, so relaying to it could
        serve a different bot's picture.

        The 503 is a SEPARATE answer from both on purpose. A child is
        ``alive`` from the instant it is forked, and its video port
        opens seconds later once Playwright has a browser — so "asked
        too early" is the ordinary case for anything that watches a bot
        it just started, not a fault. Left uncaught it reached the
        boundary as a 500, which says the server is broken when the
        truth is that the bot is still getting dressed.
    """
    try:
        port = manager.live_service_port(instance)
    except FleetError as error:
        log.warning("Fleet: refused video (404): %s", error)
        return web.Response(status=404, text=str(error))
    try:
        stream = await service_hooks.open_child_video(child_video_url(port))
    except ClientConnectionError as error:
        # Typed translation, not a swallow: this ONE failure means
        # "not yet" and every other failure still propagates to the
        # boundary. The cause is carried into the body because a
        # connection error that is NOT a warming child (a port taken by
        # something else, say) reads identically from the outside.
        log.info("Fleet: %r is not serving video yet: %s", instance, error)
        return web.Response(
            status=503,
            text=f"instance {instance!r} is not serving video yet: {error}",
            headers={"Retry-After": str(CHILD_WARMUP_RETRY_SECONDS)},
        )
    response = web.StreamResponse(status=200, headers={"Content-Type": stream.content_type})
    await response.prepare(request)
    try:
        async for chunk in stream.chunks():
            await response.write(chunk)
    finally:
        await stream.close()
    return response


__all__ = [
    "CHILD_WARMUP_RETRY_SECONDS",
    "log",
    "relay_child_video",
]
