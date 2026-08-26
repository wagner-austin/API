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
from tankpit_bot.service.fleet_manager import (
    FleetBotDict,
    FleetError,
    FleetManager,
)
from tankpit_bot.service.fleet_page import FLEET_PAGE_HTML

log = get_logger(__name__)


def encode_fleet_bot(bot: FleetBotDict) -> JSONObject:
    """Encode one report row for the HTTP surface.

    Args:
        bot: The report row.

    Returns:
        JSON-serializable object.
    """
    return {
        "instance": bot["instance"],
        "account": bot["account"],
        "role": bot["role"],
        "room": bot["room"],
        "pid": bot["pid"],
        "alive": bot["alive"],
        "returncode": bot["returncode"],
        "kills": bot["kills"],
        "seconds": bot["seconds"],
        "started_ms": bot["started_ms"],
    }


def parse_spawn_request(body: bytes) -> tuple[str, str, int, int, str, str]:
    """Parse the ``POST /bots`` body.

    Args:
        body: Raw request body.

    Returns:
        ``(instance, account, kills, seconds, role, room)``.

    Raises:
        JSONTypeError: Malformed JSON or wrong field types.
    """
    data = narrow_json_to_dict(load_json_bytes(body))
    instance = narrow_json_to_str(data.get("instance", ""))
    account = narrow_json_to_str(data.get("account", ""))
    kills = narrow_json_to_int(data.get("kills", 0))
    seconds = narrow_json_to_int(data.get("seconds", 0))
    role = narrow_json_to_str(data.get("role", ""))
    room = narrow_json_to_str(data.get("room", ""))
    return instance, account, kills, seconds, role, room


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
    """Wire the read-only routes: the page, the list, the stats.

    Args:
        app: Application under construction.
        manager: The fleet registry the routes read.
    """

    async def control_page(request: web.Request) -> web.Response:
        """``GET /`` — the human control page (a skin over the API)."""
        _ = request
        return web.Response(text=FLEET_PAGE_HTML, content_type="text/html")

    async def list_bots(request: web.Request) -> web.Response:
        """``GET /bots`` — every instance's current state."""
        _ = request
        rows: list[JSONValue] = [encode_fleet_bot(bot) for bot in manager.report()]
        return _json_response({"bots": rows})

    async def list_accounts(request: web.Request) -> web.Response:
        """``GET /accounts`` — configured usernames, first is default."""
        _ = request
        names: list[JSONValue] = list(manager.accounts())
        return _json_response({"accounts": names})

    app.router.add_get("/", control_page)
    app.router.add_get("/accounts", list_accounts)
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
            instance, account, kills, seconds, role, room = parse_spawn_request(
                await request.read()
            )
            row = manager.spawn(
                instance=instance,
                account=account,
                kills=kills,
                seconds=seconds,
                role=role,
                room=room,
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
    _add_lifecycle_routes(app, manager)
    return app


__all__ = [
    "encode_fleet_bot",
    "log",
    "make_fleet_app",
    "parse_spawn_request",
]
