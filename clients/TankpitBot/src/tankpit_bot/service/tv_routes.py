"""The operator's TV routes. The public edge does not forward them.

* ``GET /tv`` says what the four-up stream shows (each tile's bot,
  whether the encoder runs, any fault), with no request to the TV;
* ``POST /tv/cast`` makes the TV play the stream
  (:func:`tankpit_bot.service.tv.cast`);
* ``POST /tv/stop`` makes it stop.

The TV requests block for up to their timeout, so they run on a worker
thread and the fleet's own loop keeps serving while a TV in standby
fails to answer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from xml.etree import ElementTree

from aiohttp import web
from platform_core.json_utils import JSONObject, dump_json_str
from platform_core.logging import get_logger

from tankpit_bot.service import tv
from tankpit_bot.service.mosaic_director import MosaicDirector, encode_mosaic_status
from tankpit_bot.service.tv import TvError, TvStateDict, TvTargetDict

log = get_logger(__name__)

TV_ERROR_STATUS = {"TV_NOT_CONFIGURED": 409, "TV_REFUSED": 502, "TV_UNREADABLE": 502}
"""The HTTP status each named TV failure answers with: a fleet with no
TV configured is a conflict with the request, and a TV that faulted or
answered nonsense is a bad gateway."""

TV_UNREACHABLE_STATUS = 504
"""The answer when the TV could not be reached or did not answer in
time, which is what a TV in standby does."""


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


async def _tv_answer(action: str, send: Callable[[TvTargetDict], TvStateDict]) -> web.Response:
    """Run one TV request on a worker thread and answer with its outcome.

    This is the boundary every TV failure reaches: each is answered
    with its code and its words, and logged.

    Args:
        action: The request's name, for the log.
        send: The request: :func:`~tankpit_bot.service.tv.cast` or
            :func:`~tankpit_bot.service.tv.stop`.

    Returns:
        The TV's state, or the failure's status and message.
    """
    try:
        target = tv.resolve_tv_target()
        state = await asyncio.to_thread(send, target)
    except TvError as error:
        log.warning("Fleet: TV %s failed: %s", action, error)
        return web.Response(status=TV_ERROR_STATUS[error.code], text=str(error))
    except ElementTree.ParseError as error:
        log.warning("Fleet: TV %s answered with something not XML: %s", action, error)
        return web.Response(
            status=TV_ERROR_STATUS["TV_UNREADABLE"],
            text=f"TV_UNREADABLE: the TV's answer to {action} is not XML ({error})",
        )
    except OSError as error:
        log.warning("Fleet: TV %s could not reach the TV: %s", action, error)
        return web.Response(
            status=TV_UNREACHABLE_STATUS,
            text=f"TV_UNREACHABLE: the TV did not answer ({error}); is it switched on?",
        )
    log.info("Fleet: TV %s -> %s", action, state["transport_state"])
    return _json_response(tv.encode_tv_state(state))


def add_tv_routes(app: web.Application, director: MosaicDirector) -> None:
    """Wire the operator's TV routes onto the fleet's application.

    Args:
        app: Application under construction.
        director: The director whose stream the TV plays.
    """

    async def tv_status(request: web.Request) -> web.Response:
        """``GET /tv`` — what the stream shows, without asking the TV."""
        _ = request
        return _json_response({"mosaic": encode_mosaic_status(director.status())})

    async def tv_cast(request: web.Request) -> web.Response:
        """``POST /tv/cast`` — make the TV play the four-up stream."""
        _ = request
        return await _tv_answer("cast", tv.cast)

    async def tv_stop(request: web.Request) -> web.Response:
        """``POST /tv/stop`` — make the TV stop playing."""
        _ = request
        return await _tv_answer("stop", tv.stop)

    app.router.add_get("/tv", tv_status)
    app.router.add_post("/tv/cast", tv_cast)
    app.router.add_post("/tv/stop", tv_stop)


__all__ = [
    "TV_ERROR_STATUS",
    "TV_UNREACHABLE_STATUS",
    "add_tv_routes",
    "log",
]
