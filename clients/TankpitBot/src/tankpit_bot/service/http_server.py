"""aiohttp HTTP surface for the bot service.

Exposes nine routes to the SPA and watch page (via the nginx
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
* ``GET  /video``   — MJPEG relay of the Chrome screencast frames on
  the frame bus. Subscribing here is the DEMAND signal that makes the
  tick loop start the screencast.
* ``GET  /frame``   — latest cached JPEG frame as a one-shot
  snapshot; 404 until a first frame has ever been published.

The three thread-crossings are deliberate:

* ``POST /start`` offloads :meth:`SessionRunner.start` to a background
  thread via :meth:`AbstractEventLoop.run_in_executor` — the sync
  Playwright greenlet must run off the aiohttp event loop.
* ``POST /stop`` calls :meth:`SessionRunner.request_stop` directly.
  The method writes a file; the tick loop polls it. Zero coordination
  needed.
* ``GET /status`` waits on the sync ``StatusBus`` subscriber inside
  ``run_in_executor`` so the event loop stays responsive; each yielded
  frame writes back into aiohttp's SSE response.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
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

from tankpit_bot.bus.frame_bus import FrameBusProtocol
from tankpit_bot.bus.mode_bridge import ModeBridgeProtocol
from tankpit_bot.bus.status_bus import StatusBusProtocol
from tankpit_bot.service.session_runner import SessionAlreadyRunningError
from tankpit_bot.service.types_codecs import (
    decode_mode_command,
    encode_session_status,
)
from tankpit_bot.service.watch_page import WATCH_PAGE_HTML

log = get_logger(__name__)

# Timeout the SSE subscriber uses when waiting for the next frame.
# Wakes at least every ``_SSE_HEARTBEAT_SECONDS`` seconds so the
# event loop can honour cancellations promptly (client disconnect,
# service teardown) and so proxies (nginx, cloudflared) do not idle
# the TCP connection out.
_SSE_HEARTBEAT_SECONDS = 15.0

# Same wake cadence for the MJPEG relay. MJPEG has no comment channel,
# so on a quiet stretch (bot idle, page static) the relay re-sends the
# last frame as the keepalive — visually idempotent, and it keeps
# nginx/cloudflared from idling the connection out.
_MJPEG_KEEPALIVE_SECONDS = 15.0

# Multipart boundary token for the ``/video`` MJPEG stream.
_MJPEG_BOUNDARY = "tankpitbotframe"

# How long ``GET /frame`` waits for a fresh frame before falling back
# to the cached one. Slightly over one tick: subscribing creates
# screencast demand, and the tick loop reacts at the next 2 s tick
# boundary.
_FRAME_SNAPSHOT_TIMEOUT_SECONDS = 3.0


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
    frame_bus: FrameBusProtocol,
    on_shutdown: Callable[[], None],
) -> web.Application:
    """Build the aiohttp application backing the bot service.

    Args:
        runner: The single-session runner shared with the service main.
        mode_bridge: Cross-thread mode override channel.
        status_bus: Cross-thread status fan-out.
        frame_bus: Cross-thread JPEG-frame fan-out feeding ``/video``
            and ``/frame``.
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
    _add_watch_routes(app, frame_bus)
    return app


def _add_watch_routes(app: web.Application, frame_bus: FrameBusProtocol) -> None:
    """Register the fiesta-free watch surface (2026-07-28).

    Kept out of :func:`make_app` so the route builder stays under the
    complexity budget: the three viewer routes share only the frame
    bus and none of the session-control collaborators.

    Args:
        app: Application to register the routes on.
        frame_bus: Cross-thread JPEG-frame fan-out feeding ``/video``
            and ``/frame``.
    """

    async def watch(request: web.Request) -> web.Response:
        """``GET /watch`` — the self-contained phone watch page."""
        _ = request
        return web.Response(
            text=WATCH_PAGE_HTML,
            content_type="text/html",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    async def video(request: web.Request) -> web.StreamResponse:
        """``GET /video`` — MJPEG stream of screencast frames."""
        response = web.StreamResponse(
            headers={
                "Content-Type": f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY}",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "X-Accel-Buffering": "no",
            }
        )
        await response.prepare(request)
        await _drain_frame_bus_to_response(frame_bus, response)
        return response

    async def frame(request: web.Request) -> web.Response:
        """``GET /frame`` — one-shot JPEG snapshot.

        404 only when no frame has EVER been published; see
        :func:`_latest_frame_snapshot` for the demand-then-cache wait.
        """
        _ = request
        data = await _latest_frame_snapshot(frame_bus)
        if data is None:
            return web.Response(status=404, text="no frame captured yet")
        return web.Response(
            body=data,
            content_type="image/jpeg",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    app.router.add_get("/watch", watch)
    app.router.add_get("/video", video)
    app.router.add_get("/frame", frame)


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


async def _latest_frame_snapshot(bus: FrameBusProtocol) -> bytes | None:
    """Wait briefly for a fresh frame, falling back to the cached one.

    Subscribing is itself the demand signal: the tick loop sees a
    non-zero subscriber count and starts the screencast at the next
    2 s tick boundary, so the wait window
    (:data:`_FRAME_SNAPSHOT_TIMEOUT_SECONDS`) usually ends with a live
    frame. On timeout (no session running, or the first frame still in
    flight) the bus's cached frame is served instead.

    Args:
        bus: Shared frame bus to snapshot from.

    Returns:
        JPEG bytes, or ``None`` when no frame has ever been published.
    """
    subscriber = bus.subscribe()
    try:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(
            None, subscriber.next_frame, _FRAME_SNAPSHOT_TIMEOUT_SECONDS
        )
    finally:
        bus.unsubscribe(subscriber)
    if data is None:
        return bus.latest()
    return data


async def _drain_frame_bus_to_response(
    bus: FrameBusProtocol,
    response: SSEResponseProtocol,
) -> None:
    """Subscribe to ``bus`` and pump JPEG frames into an MJPEG response.

    Mirrors :func:`_drain_status_bus_to_response`: owns the subscribe /
    unsubscribe pair, waits on the sync subscriber inside
    ``run_in_executor`` so the event loop stays responsive, and cleans
    the subscriber off the bus on any exception (client disconnect,
    teardown). The subscription itself is load-bearing beyond delivery:
    the tick loop reads the bus's subscriber count as the screencast
    demand signal.

    Keepalive: on a wait timeout the LAST frame is re-sent (MJPEG has
    no comment channel). Before any frame has arrived a timeout just
    loops — an idle-service viewer holds a silent open connection until
    a session starts publishing.

    Args:
        bus: Shared frame bus subscribed for the lifetime of the
            connection.
        response: aiohttp response whose write side receives multipart
            JPEG parts.
    """
    subscriber = bus.subscribe()
    try:
        loop = asyncio.get_running_loop()
        last: bytes | None = None
        while not subscriber.closed:
            frame = await loop.run_in_executor(
                None, subscriber.next_frame, _MJPEG_KEEPALIVE_SECONDS
            )
            if subscriber.closed:
                return
            if frame is None:
                if last is None:
                    continue
                frame = last
            last = frame
            header = (
                f"--{_MJPEG_BOUNDARY}\r\n"
                f"Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(frame)}\r\n\r\n"
            ).encode()
            await response.write(header + frame + b"\r\n")
    finally:
        bus.unsubscribe(subscriber)


__all__ = [
    "SessionRunnerHTTPProtocol",
    "make_app",
]
