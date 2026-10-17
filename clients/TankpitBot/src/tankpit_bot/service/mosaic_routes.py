"""The four-up stream's public route.

``GET /demo/mosaic/{file}`` serves the stream's HLS files. It is the
public demo surface's fourth route (:mod:`tankpit_bot.service.demo_routes`
holds the other three), in its own module because it reads the
director rather than the registry. It sits under ``/demo/`` because
that is the one prefix the public edge forwards (``edge/nginx.conf``),
and it must be public: the operator's TV fetches it over the internet,
since it cannot reach the fleet's container. It shows whichever bots
the fleet is running, operator bots included; what any viewer can see
there is the game's picture and the captions, never an account, a pid,
a port or a control. The operator's TV controls are
:mod:`tankpit_bot.service.tv_routes`, which the edge does not forward.
"""

from __future__ import annotations

from aiohttp import web

from tankpit_bot.service.mosaic_director import MosaicDirector
from tankpit_bot.stream.hls import hls_web_response, read_hls_file


def add_mosaic_route(app: web.Application, director: MosaicDirector) -> None:
    """Wire the four-up stream's file route onto the fleet's application.

    Args:
        app: Application under construction.
        director: The director whose stream is served.
    """

    async def mosaic_video(request: web.Request) -> web.Response:
        """``GET /demo/mosaic/{file}`` — one HLS file of the four-up stream."""
        return hls_web_response(read_hls_file(director.hls_dir, request.match_info["file"]))

    app.router.add_get("/demo/mosaic/{file}", mosaic_video)


__all__ = [
    "add_mosaic_route",
]
