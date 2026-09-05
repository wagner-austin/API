"""Serving one fleet child's HLS files through the manager.

A child's capture pipeline writes its playlist and segments under the
shared ``runs/`` tree — the same filesystem the manager owns, because
the fleet's children are forked inside the manager's container. So the
manager serves video straight off disk: no relay, no per-child port,
no warmup connection dance. (Its predecessor, ``video_relay``, proxied
an endless MJPEG response from a port each child bound; every reason
it existed left with that architecture.)

Its own module because two route groups serve the same way and must
not drift: the operator surface serves any registered instance
(:mod:`tankpit_bot.service.fleet_routes`), the public demo serves only
a demo slot (:mod:`tankpit_bot.service.demo_routes`). What differs is
WHICH instance they will resolve, which each decides before calling
here; how the bytes are found is one answer.
"""

from __future__ import annotations

from aiohttp import web
from platform_core.logging import get_logger

from tankpit_bot.runtime_artifacts import bot_run_dir
from tankpit_bot.service.fleet_error import FleetError
from tankpit_bot.service.fleet_manager import FleetManager
from tankpit_bot.stream.hls import hls_web_response, read_hls_file

log = get_logger(__name__)


def instance_video_response(manager: FleetManager, instance: str, filename: str) -> web.Response:
    """Serve one HLS file of one RUNNING instance's stream.

    Liveness gates the read: a finished bot's directory still holds
    its final segments, and its name may soon belong to a fresh spawn
    — a 404 for a dead instance is the same honest answer the old
    port allocator's design gave, moved to the on-disk world. Past
    that gate, status semantics (503 warming, 404 rotated-out, strict
    filename shape) live in :func:`~tankpit_bot.stream.hls.read_hls_file`,
    shared with the child's own ``/video/{file}`` surface.

    Args:
        manager: The registry liveness is resolved through.
        instance: The instance to serve. The caller has already
            decided this name is one it is willing to serve.
        filename: Requested filename, straight from the URL.

    Returns:
        The file response; a 404 when the instance is not registered
        or is no longer running.
    """
    try:
        manager.require_running(instance)
    except FleetError as error:
        log.info("Fleet: refused video (404): %s", error)
        return web.Response(status=404, text=str(error))
    return hls_web_response(read_hls_file(bot_run_dir(instance) / "hls", filename))


__all__ = [
    "instance_video_response",
    "log",
]
