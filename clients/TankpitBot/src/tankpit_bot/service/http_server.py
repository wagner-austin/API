"""aiohttp HTTP surface for the bot service.

Exposes eight routes to the SPA and watch page (via the nginx
same-origin proxy):

* ``GET  /health``  — cheap liveness probe. Returns immediately.
* ``POST /start``   — spawn one game session. Returns 202 on accept,
  409 when a session is already running.
* ``POST /stop``    — request the running session end at the next
  tick boundary. Returns 202 (idempotent — a stop while idle is a
  no-op success).
* ``POST /mode``    — push one :class:`ModeCommandDict` into the
  cross-thread mode bridge. Returns 204 on success.
* ``GET  /status``  — SSE stream. Each :class:`SessionStatusDict`
  frame the tick loop publishes reaches the SPA within one Playwright
  tick + one aiohttp write.
* ``POST /shutdown`` — stop the whole SERVICE (2026-07-18 lifecycle
  pass): requests any running session end, then fires the service's
  shutdown signal. Returns 202. The process exits once the session
  thread (if any) observes its stop-file at the next tick boundary.
* ``GET  /watch``   — self-contained phone watch page (2026-07-28,
  the fiesta-free replacement for the vibeshine tankpit stream).
* ``GET  /video/{file}`` — one HLS file (playlist or segment) from
  this session's capture directory, written by the ffmpeg the bot's
  own run owns (:mod:`tankpit_bot.stream.capture`). 404 when this
  session has no stream at all; 503 while the encoder is warming up.

The two thread-crossings are deliberate:

* ``POST /start`` offloads :meth:`SessionRunner.start` to a background
  thread via :meth:`AbstractEventLoop.run_in_executor` — the sync
  Playwright greenlet must run off the aiohttp event loop.
* ``GET /status`` waits on the sync ``StatusBus`` subscriber inside
  ``run_in_executor`` so the event loop stays responsive; each yielded
  frame writes back into aiohttp's SSE response.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from aiohttp import web
from platform_core.json_utils import (
    JSONTypeError,
    dump_json_str,
    load_json_bytes,
    narrow_json_to_dict,
    narrow_json_to_int,
)
from platform_core.logging import get_logger

from tankpit_bot.bus.mode_bridge import ModeBridgeProtocol
from tankpit_bot.bus.status_bus import StatusBusProtocol
from tankpit_bot.resources import require_asset
from tankpit_bot.service.session_runner import SessionAlreadyRunningError
from tankpit_bot.service.types_codecs import (
    decode_mode_command,
    encode_session_status,
)
from tankpit_bot.service.watch_page import WATCH_PAGE_HTML
from tankpit_bot.stream.hls import hls_web_response, read_hls_file

log = get_logger(__name__)

# Timeout the SSE subscriber uses when waiting for the next frame.
# Wakes at least every ``_SSE_HEARTBEAT_SECONDS`` seconds so the
# event loop can honour cancellations promptly (client disconnect,
# service teardown) and so proxies (nginx, cloudflared) do not idle
# the TCP connection out.
_SSE_HEARTBEAT_SECONDS = 15.0


class SSEResponseProtocol(Protocol):
    """Minimum aiohttp response surface the SSE drain helper needs.

    :class:`aiohttp.web.StreamResponse` satisfies this structurally.
    The Protocol exists so tests can drive the drain helper with a
    lightweight stub that records writes without spinning up aiohttp.
    """

    async def write(self, data: bytes) -> None:
        """Append ``data`` to the response's send buffer."""
        ...


class SessionRunnerHTTPProtocol(Protocol):
    """The :class:`SessionRunner` methods the HTTP surface consumes."""

    def start(self, *, session_seconds: int = 0, session_kills: int = 0) -> None:
        """Run one session start-to-finish. Blocks the calling thread.

        Raises:
            SessionAlreadyRunningError: A session is already active.
        """
        ...

    def request_stop(self) -> None:
        """Request the running session end at the next tick boundary."""
        ...

    def is_running(self) -> bool:
        """Return True when a session is currently active."""
        ...


def make_app(
    runner: SessionRunnerHTTPProtocol,
    mode_bridge: ModeBridgeProtocol,
    status_bus: StatusBusProtocol,
    hls_dir: Path | None,
    on_shutdown: Callable[[], None],
) -> web.Application:
    """Build the aiohttp application backing the bot service.

    Args:
        runner: The single-session runner shared with the service main.
        mode_bridge: Cross-thread mode override channel.
        status_bus: Cross-thread status fan-out.
        hls_dir: Directory this session's capture pipeline writes HLS
            files into, or ``None`` when the session is not streamed —
            ``/video/{file}`` then answers an honest 404 rather than a
            warming 503 that would never resolve.
        on_shutdown: Fired by ``POST /shutdown`` after any running
            session has been asked to stop. Production wires the
            service main's ``stop_event.set``; tests pass a recorder.

    Returns:
        A fully-routed :class:`web.Application` ready to be handed to
        :class:`aiohttp.web.AppRunner`.
    """
    app = web.Application()

    async def health(request: web.Request) -> web.Response:
        """``GET /health`` — cheap liveness probe."""
        _ = request
        return web.Response(text="ok")

    async def start(request: web.Request) -> web.Response:
        """``POST /start`` — accept and offload the session start.

        Optional JSON body sets the session bounds:
        ``{"seconds": 2700, "kills": 30}`` — either key may be
        omitted; an empty body (the phone flow) runs unbounded, as
        before. Non-integer or negative bounds are a 400.
        """
        body = await request.read()
        try:
            session_seconds, session_kills = _parse_session_bounds(body)
        except (JSONTypeError, ValueError) as error:
            # Client error, surfaced in BOTH channels: the 400 body for
            # the caller, the server log for whoever is watching the
            # service (a phone-flow typo is otherwise invisible here).
            log.info("start rejected: bad session bounds: %s", error)
            return web.Response(status=400, text=f"bad session bounds: {error}")
        if runner.is_running():
            return web.Response(status=409, text="session already running")
        loop = asyncio.get_running_loop()
        loop.run_in_executor(
            None,
            _run_session_and_log_rejection,
            runner,
            session_seconds,
            session_kills,
        )
        return web.Response(
            status=202,
            text=f"starting (seconds={session_seconds}, kills={session_kills})",
        )

    async def stop(request: web.Request) -> web.Response:
        """``POST /stop`` — request the running session end."""
        _ = request
        runner.request_stop()
        return web.Response(status=202, text="stopping")

    async def mode(request: web.Request) -> web.Response:
        """``POST /mode`` — push a mode override onto the bridge."""
        body = await request.read()
        raw = load_json_bytes(body)
        data = narrow_json_to_dict(raw)
        command = decode_mode_command(data)
        mode_bridge.submit(command["manual_mode"])
        return web.Response(status=204)

    async def status(request: web.Request) -> web.StreamResponse:
        """``GET /status`` — SSE stream of ``SessionStatusDict`` frames."""
        response = web.StreamResponse(
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "X-Accel-Buffering": "no",
            }
        )
        await response.prepare(request)
        await _drain_status_bus_to_response(status_bus, response)
        return response

    async def shutdown(request: web.Request) -> web.Response:
        """``POST /shutdown`` — stop the whole service.

        Any running session is asked to stop first (idempotent when
        idle), then the service's shutdown signal fires. The 202 goes
        out before teardown because the response must reach the phone
        over an HTTP surface this call is about to dismantle.
        """
        _ = request
        log.info("Shutdown requested via POST /shutdown")
        runner.request_stop()
        on_shutdown()
        return web.Response(status=202, text="shutting down")

    app.router.add_get("/health", health)
    app.router.add_post("/start", start)
    app.router.add_post("/stop", stop)
    app.router.add_post("/mode", mode)
    app.router.add_get("/status", status)
    app.router.add_post("/shutdown", shutdown)
    _add_watch_routes(app, hls_dir)
    return app


def _add_watch_routes(app: web.Application, hls_dir: Path | None) -> None:
    """Register the viewer surface (2026-07-28, rebuilt 2026-09-05).

    Kept out of :func:`make_app` so the route builder stays under the
    complexity budget: these routes share only the capture directory
    and none of the session-control collaborators.

    Args:
        app: Application to register the routes on.
        hls_dir: This session's HLS directory, or ``None`` when the
            session is not streamed.
    """

    async def watch(request: web.Request) -> web.Response:
        """``GET /watch`` — the self-contained phone watch page."""
        _ = request
        return web.Response(
            text=WATCH_PAGE_HTML,
            content_type="text/html",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    async def video(request: web.Request) -> web.Response:
        """``GET /video/{file}`` — one HLS file from this session's stream.

        The bytes come off disk, where the ffmpeg owned by the bot's
        own run put them; this process never touches the page or the
        tick loop to serve video, which is the entire architecture.
        Status semantics live in :func:`read_hls_file`, shared with
        the fleet manager's demo route so the two surfaces cannot
        drift.
        """
        if hls_dir is None:
            return web.Response(status=404, text="this session has no stream")
        return hls_web_response(read_hls_file(hls_dir, request.match_info["file"]))

    async def hls_js(request: web.Request) -> web.Response:
        """``GET /watch/hls.js`` — the vendored hls.js build.

        Shipped inside the wheel ([[packaged-data-assets]]) so the
        watch page stays servable with no reach outside the service —
        a CDN tag would make the operator's phone page depend on a
        third party to show a picture the service already has.
        """
        _ = request
        return web.Response(
            body=require_asset("hls.min.js").read_bytes(),
            content_type="application/javascript",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    app.router.add_get("/watch", watch)
    app.router.add_get("/watch/hls.js", hls_js)
    app.router.add_get("/video/{file}", video)


def _parse_session_bounds(body: bytes) -> tuple[int, int]:
    """Parse the optional ``POST /start`` bounds body.

    Args:
        body: Raw request body; empty means unbounded (the phone
            flow's buttons send no body).

    Returns:
        ``(session_seconds, session_kills)`` — zero means unbounded.

    Raises:
        JSONTypeError: Malformed JSON or non-integer bound values.
        ValueError: Negative bounds.
    """
    if not body:
        return 0, 0
    data = narrow_json_to_dict(load_json_bytes(body))
    session_seconds = narrow_json_to_int(data.get("seconds", 0))
    session_kills = narrow_json_to_int(data.get("kills", 0))
    if session_seconds < 0 or session_kills < 0:
        raise ValueError("negative")
    return session_seconds, session_kills


def _run_session_and_log_rejection(
    runner: SessionRunnerHTTPProtocol,
    session_seconds: int,
    session_kills: int,
) -> None:
    """Invoke :meth:`SessionRunner.start` on the executor thread.

    A :class:`SessionAlreadyRunningError` from a two-``POST /start``
    race is expected and logged at ``WARNING``. EVERY other exception
    is logged with its traceback at ``ERROR`` — the executor future
    this function runs under is never awaited, so a propagating
    exception would be silently swallowed into the future object
    (observed 2026-07-19: two ``POST /start`` → 202, the session died
    before its run log existed, and nothing anywhere said why). The
    original docstring claimed the excepthook would catch it; it does
    not for ``run_in_executor``.
    """
    try:
        runner.start(session_seconds=session_seconds, session_kills=session_kills)
    except SessionAlreadyRunningError as exc:
        log.warning("Session start rejected: %s", exc)
    except Exception:
        log.exception("Session crashed before/during run — POST /start had already returned 202")
        # Re-raised into the executor future (which nobody awaits) —
        # the log line above is the real record; the raise satisfies
        # the log-and-raise contract without changing behaviour.
        raise


async def _drain_status_bus_to_response(
    bus: StatusBusProtocol,
    response: SSEResponseProtocol,
) -> None:
    """Subscribe to ``bus`` and pump frames into an SSE response.

    Owns the subscribe / unsubscribe pair so the status handler stays
    a straight-line ``prepare → drain → return``. On any exception
    (client disconnect, teardown) the finally cleans the subscriber
    off the bus.

    Args:
        bus: Shared status bus subscribed for the lifetime of the SSE
            connection.
        response: aiohttp SSE response whose write side receives
            ``data: <encoded frame>\\n\\n`` blocks + heartbeats.
    """
    subscriber = bus.subscribe()
    try:
        loop = asyncio.get_running_loop()
        while not subscriber.closed:
            frame = await loop.run_in_executor(None, subscriber.next_frame, _SSE_HEARTBEAT_SECONDS)
            if subscriber.closed:
                return
            if frame is None:
                await response.write(b": heartbeat\n\n")
                continue
            payload = dump_json_str(encode_session_status(frame), compact=True)
            await response.write(f"data: {payload}\n\n".encode())
    finally:
        bus.unsubscribe(subscriber)


__all__ = [
    "SessionRunnerHTTPProtocol",
    "make_app",
]
