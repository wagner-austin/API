"""Fleet HTTP transport: the aiohttp surface over the registry.

Request parsing, response encoding, and status-code mapping, grouped
into observation, telemetry, and lifecycle route registrars. The
registry itself is :mod:`tankpit_bot.service.fleet_manager`.
"""

from __future__ import annotations

from aiohttp import web
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_bytes,
    narrow_json_to_dict,
    narrow_json_to_int,
    narrow_json_to_str,
)
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.runtime_artifacts import bot_run_dir
from tankpit_bot.service.demo_routes import add_demo_routes
from tankpit_bot.service.fleet_config import (
    configured_accounts,
    engagement_doctrines,
    lobby_rooms,
    tank_registry,
    troop_colors,
)
from tankpit_bot.service.fleet_error import FleetError
from tankpit_bot.service.fleet_manager import FleetManager
from tankpit_bot.service.fleet_page import FLEET_PAGE_HTML
from tankpit_bot.service.fleet_wire import (
    FleetSnapshotDict,
    SpawnRequestDict,
    encode_fleet_bot,
    encode_fleet_snapshot,
)
from tankpit_bot.service.video_files import instance_video_response

log = get_logger(__name__)


def parse_spawn_request(body: bytes) -> SpawnRequestDict:
    """Parse the ``POST /bots`` body.

    Args:
        body: Raw request body.

    Returns:
        The parsed request. Every selector defaults to ``""`` and
        every bound to ``0``, which each resolver reads as "keep the
        default" rather than as a value.

    Raises:
        JSONTypeError: Malformed JSON or wrong field types.
    """
    data = narrow_json_to_dict(load_json_bytes(body))
    return SpawnRequestDict(
        instance=narrow_json_to_str(data.get("instance", "")),
        account=narrow_json_to_str(data.get("account", "")),
        kills=narrow_json_to_int(data.get("kills", 0)),
        seconds=narrow_json_to_int(data.get("seconds", 0)),
        role=narrow_json_to_str(data.get("role", "")),
        room=narrow_json_to_str(data.get("room", "")),
        troop=narrow_json_to_str(data.get("troop", "")),
        doctrine=narrow_json_to_str(data.get("doctrine", "")),
    )


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


def _add_observation_routes(app: web.Application, manager: FleetManager) -> None:
    """Wire the read-only routes: the page, the lists, the stats.

    Args:
        app: Application under construction.
        manager: The fleet registry the routes read.
    """

    async def control_page(request: web.Request) -> web.Response:
        """``GET /`` — the human control page (a skin over the API)."""
        _ = request
        return web.Response(text=FLEET_PAGE_HTML, content_type="text/html")

    async def list_bots(request: web.Request) -> web.Response:
        """``GET /bots`` — every instance's current state.

        Carries the manager's ``boot`` identity and whether it is
        ``draining``. Both are for the page rather than the registry:
        a boot id it does not recognise means every name it is holding
        belongs to a manager that no longer exists, and a draining
        manager is one whose bots are on their way out.
        """
        _ = request
        return _json_response(
            encode_fleet_snapshot(
                FleetSnapshotDict(
                    boot=manager.boot_id,
                    draining=manager.draining(),
                    bots=manager.report(),
                )
            )
        )

    async def list_accounts(request: web.Request) -> web.Response:
        """``GET /accounts`` — configured usernames, first is default."""
        _ = request
        names: list[JSONValue] = list(configured_accounts())
        return _json_response({"accounts": names})

    async def list_rooms(request: web.Request) -> web.Response:
        """``GET /rooms`` — room selectors, first is the default."""
        _ = request
        names: list[JSONValue] = list(lobby_rooms())
        return _json_response({"rooms": names})

    async def list_doctrines(request: web.Request) -> web.Response:
        """``GET /doctrines`` — engagement doctrines, default first."""
        _ = request
        names: list[JSONValue] = list(engagement_doctrines())
        return _json_response({"doctrines": names})

    async def list_troops(request: web.Request) -> web.Response:
        """``GET /troops`` — tank colors, in wire team-id order."""
        _ = request
        names: list[JSONValue] = list(troop_colors())
        return _json_response({"troops": names})

    async def list_tanks(request: web.Request) -> web.Response:
        """``GET /tanks`` — measured rank per account, world and colour."""
        _ = request
        return _json_response({"tanks": tank_registry()})

    app.router.add_get("/", control_page)
    app.router.add_get("/accounts", list_accounts)
    app.router.add_get("/rooms", list_rooms)
    app.router.add_get("/troops", list_troops)
    app.router.add_get("/doctrines", list_doctrines)
    app.router.add_get("/tanks", list_tanks)
    app.router.add_get("/bots", list_bots)


def _add_telemetry_routes(app: web.Application, manager: FleetManager) -> None:
    """Wire the per-instance telemetry routes: stats, hud, activity.

    Args:
        app: Application under construction.
        manager: The fleet registry the routes read.
    """

    async def bot_stats(request: web.Request) -> web.Response:
        """``GET /bots/{instance}/stats`` — latest-run digest summary."""
        try:
            summary = manager.stats(request.match_info["instance"])
        except FleetError as error:
            log.warning("Fleet: refused stats (404): %s", error)
            return web.Response(status=404, text=str(error))
        return _json_response(summary)

    async def bot_hud(request: web.Request) -> web.Response:
        """``GET /bots/{instance}/hud`` — this tick's in-page HUD payload.

        The bot mirrors the exact payload the ``make run`` overlay
        renders to ``runs/bot/<instance>/hud.json`` every tick; the
        fleet serves it verbatim so the page can draw the same card.
        """
        instance = request.match_info["instance"]
        try:
            manager.stats_gate(instance)
        except FleetError as error:
            log.warning("Fleet: refused hud (404): %s", error)
            return web.Response(status=404, text=str(error))
        try:
            raw = top_hooks.read_text(bot_run_dir(instance) / "hud.json")
        except OSError as error:
            # A run that has not written a HUD frame yet is the normal
            # case mid-boot, not a fault: the page polls until it lands.
            log.info("Fleet: no hud for %r yet: %s", instance, error)
            return _json_response({"available": False})
        return web.Response(text=raw, content_type="application/json")

    async def bot_activity(request: web.Request) -> web.Response:
        """``GET /bots/{instance}/activity`` — live tail of the run."""
        try:
            tail = manager.activity(request.match_info["instance"])
        except FleetError as error:
            log.warning("Fleet: refused activity (404): %s", error)
            return web.Response(status=404, text=str(error))
        return _json_response(tail)

    app.router.add_get("/bots/{instance}/stats", bot_stats)
    app.router.add_get("/bots/{instance}/hud", bot_hud)
    app.router.add_get("/bots/{instance}/activity", bot_activity)


def _add_video_routes(app: web.Application, manager: FleetManager) -> None:
    """Wire the per-instance video file serving.

    Its own group so the split keeps each wiring function inside the
    complexity budget, and because this is the one route family whose
    payload comes off the shared ``runs/`` tree rather than out of the
    registry.

    Args:
        app: Application under construction.
        manager: The fleet registry liveness is resolved through.
    """

    async def bot_video(request: web.Request) -> web.Response:
        """``GET /bots/{instance}/video/{file}`` — one HLS file of one child.

        Any REGISTERED instance: this is the operator surface, and an
        operator who may stop a bot may watch it. The public demo
        route narrows the same serving to demo slots
        (:mod:`tankpit_bot.service.demo_routes`).
        """
        return instance_video_response(
            manager, request.match_info["instance"], request.match_info["file"]
        )

    app.router.add_get("/bots/{instance}/video/{file}", bot_video)


def _add_lifecycle_routes(app: web.Application, manager: FleetManager) -> None:
    """Wire the mutating routes: spawn, stop, restart, remove.

    Args:
        app: Application under construction.
        manager: The fleet registry the routes operate on.
    """

    async def restart_bot(request: web.Request) -> web.Response:
        """``POST /bots/{instance}/restart`` — respawn a finished bot."""
        try:
            row = manager.restart(request.match_info["instance"])
        except FleetError as error:
            status = 409 if "still running" in str(error) else 404
            log.warning("Fleet: refused restart (%d): %s", status, error)
            return web.Response(status=status, text=str(error))
        return _json_response(encode_fleet_bot(row), status=201)

    async def spawn_bot(request: web.Request) -> web.Response:
        """``POST /bots`` — spawn one instance."""
        try:
            spawn_request = parse_spawn_request(await request.read())
            row = manager.spawn(
                instance=spawn_request["instance"],
                account=spawn_request["account"],
                kills=spawn_request["kills"],
                seconds=spawn_request["seconds"],
                role=spawn_request["role"],
                room=spawn_request["room"],
                troop=spawn_request["troop"],
                doctrine=spawn_request["doctrine"],
            )
        except (JSONTypeError, ValueError) as error:
            log.warning("Fleet: rejected spawn request (400): %s", error)
            return web.Response(status=400, text=f"bad spawn request: {error}")
        except FleetError as error:
            log.warning("Fleet: refused spawn (409): %s", error)
            return web.Response(status=409, text=str(error))
        return _json_response(encode_fleet_bot(row), status=201)

    async def stop_bot(request: web.Request) -> web.Response:
        """``POST /bots/{instance}/stop`` — graceful stop via sentinel."""
        try:
            row = manager.stop(request.match_info["instance"])
        except FleetError as error:
            log.warning("Fleet: refused stop (404): %s", error)
            return web.Response(status=404, text=str(error))
        return _json_response(encode_fleet_bot(row))

    async def remove_bot(request: web.Request) -> web.Response:
        """``DELETE /bots/{instance}`` — drop a finished instance."""
        try:
            row = manager.remove(request.match_info["instance"])
        except FleetError as error:
            status = 409 if "still running" in str(error) else 404
            log.warning("Fleet: refused remove (%d): %s", status, error)
            return web.Response(status=status, text=str(error))
        return _json_response(encode_fleet_bot(row))

    app.router.add_post("/bots", spawn_bot)
    app.router.add_post("/bots/{instance}/stop", stop_bot)
    app.router.add_post("/bots/{instance}/restart", restart_bot)
    app.router.add_delete("/bots/{instance}", remove_bot)


def _add_shutdown_route(app: web.Application, manager: FleetManager) -> None:
    """Wire the manager's own shutdown.

    Its own registrar rather than one more branch of the lifecycle
    group: those routes act on ONE instance, this one acts on the
    manager and everything it holds.

    Args:
        app: Application under construction.
        manager: The fleet registry to drain.
    """

    async def shutdown_fleet(request: web.Request) -> web.Response:
        """``POST /shutdown`` — drain every bot, then exit.

        Returns 202 immediately, naming the bots asked to stop. The
        manager does NOT exit here: it stays up, supervising, until
        the last child has finished its own teardown. That is what
        makes an interrupted ``make down`` harmless — the client
        walking away does not orphan anything, because the manager is
        still the one holding the drain.
        """
        _ = request
        draining: list[JSONValue] = list(manager.request_drain())
        return _json_response({"draining": draining}, status=202)

    app.router.add_post("/shutdown", shutdown_fleet)


def make_fleet_app(manager: FleetManager) -> web.Application:
    """Build the fleet manager's aiohttp application.

    Args:
        manager: The fleet registry the routes operate on.

    Returns:
        The configured ``aiohttp.web.Application``.
    """
    app = web.Application()
    _add_observation_routes(app, manager)
    _add_telemetry_routes(app, manager)
    _add_video_routes(app, manager)
    _add_lifecycle_routes(app, manager)
    _add_shutdown_route(app, manager)
    add_demo_routes(app, manager)
    return app


__all__ = [
    "log",
    "make_fleet_app",
    "parse_spawn_request",
]
