"""The public demo's three HTTP routes.

Its own module rather than a fifth registrar inside
:mod:`tankpit_bot.service.fleet_routes` because the two surfaces answer
to different readers. Everything in ``fleet_routes`` assumes an operator
— it names accounts, hands out pids, stops and removes bots. Everything
here assumes a stranger. Keeping them in one file would make "is this
route public?" a question about which function a handler happens to sit
inside; keeping them apart makes it a question about which file.

Three routes, and the surface is deliberately not extensible by
accident: ``GET /demo/fleet`` says what is playing, ``POST /demo/spawn``
starts one bounded Practice bot, ``GET /demo/video/{slot}`` watches one.
There is no public stop, no public remove, no public shutdown — a demo
bot ends on its own clock (:data:`~tankpit_bot.service.demo.DEMO_SESSION_SECONDS`),
so nothing anonymous needs the power to end one.
"""

from __future__ import annotations

from aiohttp import web
from platform_core.json_utils import JSONObject, dump_json_str
from platform_core.logging import get_logger

from tankpit_bot.service.demo import (
    demo_fleet,
    demo_slot_or_refuse,
    demo_spawn,
    encode_demo_bot,
    encode_demo_fleet,
)
from tankpit_bot.service.fleet_error import FleetError
from tankpit_bot.service.fleet_manager import FleetManager
from tankpit_bot.service.video_files import instance_video_response

log = get_logger(__name__)


def _json_response(payload: JSONObject, status: int = 200) -> web.Response:
    """Build one JSON response.

    Args:
        payload: JSON-serializable body.
        status: HTTP status code.

    Returns:
        The response.
    """
    return web.Response(
        status=status,
        text=dump_json_str(payload, indent=1),
        content_type="application/json",
    )


def add_demo_routes(app: web.Application, manager: FleetManager) -> None:
    """Wire the public demo surface onto the manager's application.

    Args:
        app: Application under construction.
        manager: The fleet registry the demo policy operates on.
    """

    async def fleet_state(request: web.Request) -> web.Response:
        """``GET /demo/fleet`` — what is playing, in public terms."""
        _ = request
        return _json_response(encode_demo_fleet(demo_fleet(manager)))

    async def spawn(request: web.Request) -> web.Response:
        """``POST /demo/spawn`` — start one bounded Practice bot.

        The body is never read. A demo spawn has no parameters at all,
        so a caller sending some has sent something this surface has no
        field to put anywhere.
        """
        _ = request
        try:
            row = demo_spawn(manager)
        except FleetError as error:
            log.info("Demo: refused spawn (409): %s", error)
            return web.Response(status=409, text=str(error))
        return _json_response(encode_demo_bot(row), status=201)

    async def video(request: web.Request) -> web.Response:
        """``GET /demo/video/{slot}/{file}`` — one HLS file of one demo bot.

        The slot is resolved through the demo grammar BEFORE the
        registry sees it, so an operator instance name reaches the
        file serving as a 404 rather than as a picture. The viewer
        asks for ``index.m3u8`` first and the playlist's own relative
        segment names resolve back under this same prefix.
        """
        try:
            slot = demo_slot_or_refuse(request.match_info["slot"])
        except FleetError as error:
            log.info("Demo: refused video (404): %s", error)
            return web.Response(status=404, text=str(error))
        return instance_video_response(manager, slot, request.match_info["file"])

    app.router.add_get("/demo/fleet", fleet_state)
    app.router.add_post("/demo/spawn", spawn)
    app.router.add_get("/demo/video/{slot}/{file}", video)


__all__ = [
    "add_demo_routes",
    "log",
]
