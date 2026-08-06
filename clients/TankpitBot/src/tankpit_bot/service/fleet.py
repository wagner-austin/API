"""AI-operated bot fleet: spawn, observe, and stop instance processes.

Built 2026-08-06 (user: "the goal is the ai can spin up and maintain
and see the bots, not the spa method"). The single-bot service and its
phone watch page stay what they are; THIS surface is for the operating
AI. It runs in a terminal the operator owns, so an orchestration
harness dying can never kill a live tank (the 41-kill session died at
46 minutes exactly that way), and the AI drives it with plain HTTP:

* ``GET  /bots`` — every instance, its pid, liveness, and bounds.
* ``POST /bots`` — spawn one bot process:
  ``{"instance": "alpha", "account": "...", "kills": 30,
  "seconds": 2700}`` (account and bounds optional).
* ``POST /bots/{instance}/stop`` — graceful: writes the instance's
  stop sentinel; the tick loop exits at the next boundary with a full
  teardown (scorecard, capture, archive).
* ``DELETE /bots/{instance}`` — remove a finished instance from the
  registry (refuses while the process lives).

Each bot is a CHILD PROCESS of this manager running the existing
``tankpit_bot.bot.entry`` main — the per-instance isolation
(``TANKPIT_BOT_INSTANCE`` artifact namespace, instance stop sentinel,
``TANKPIT_ACCOUNT`` selection) all lands via the child's environment.
Game-state observation stays on disk where it already lives: the AI
reads ``runs/bot/<instance>/latest.log`` and runs ``tankpit-run-digest``
against the instance's events; this surface owns lifecycle, not
telemetry.
"""

from __future__ import annotations

from aiohttp import web
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_bytes,
    narrow_json_to_dict,
    narrow_json_to_int,
    narrow_json_to_str,
)
from platform_core.logging import get_logger
from typing_extensions import TypedDict

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.runtime_artifacts import _INSTANCE_NAME, bot_run_dir
from tankpit_bot.service import _test_hooks as service_hooks

log = get_logger(__name__)

FLEET_PORT_DEFAULT = 27300
"""Default TCP port for the fleet manager.

Well below the Windows dynamic-reservation range for the same reason
as ``SERVICE_PORT`` (see :mod:`tankpit_bot.service.constants`), and
distinct from it so a sole-bot service and the fleet can coexist.
"""


def resolve_fleet_port() -> int:
    """Resolve the fleet manager's port from the environment.

    Returns:
        ``TANKPIT_FLEET_PORT`` when set, else :data:`FLEET_PORT_DEFAULT`.

    Raises:
        ValueError: If the value is not an integer in [1024, 65535].
    """
    raw = top_hooks.get_env("TANKPIT_FLEET_PORT")
    if raw is None or raw == "":
        return FLEET_PORT_DEFAULT
    port = int(raw)
    if not 1024 <= port <= 65535:
        raise ValueError(f"TANKPIT_FLEET_PORT {port} outside [1024, 65535]")
    return port


class FleetBotDict(TypedDict):
    """One managed bot instance, as reported by ``GET /bots``.

    Attributes:
        instance: Validated instance name (artifact namespace).
        account: ``TANKPIT_ACCOUNT`` the child was spawned with
            (empty means the accounts.json default).
        pid: Child process id.
        alive: Whether the process is still running at report time.
        returncode: Exit code once dead; ``None`` while alive.
        kills: Kill bound the child was spawned with (0 unbounded).
        seconds: Seconds bound the child was spawned with (0 unbounded).
        started_ms: Wall-clock spawn time.
    """

    instance: str
    account: str
    pid: int
    alive: bool
    returncode: int | None
    kills: int
    seconds: int
    started_ms: int


class _ManagedBot:
    """Registry entry pairing spawn metadata with the live process."""

    def __init__(
        self,
        *,
        instance: str,
        account: str,
        kills: int,
        seconds: int,
        started_ms: int,
        process: service_hooks.SpawnedProcessProtocol,
    ) -> None:
        """Bind one spawned bot to its metadata.

        Args:
            instance: Validated instance name.
            account: Account selector the child received.
            kills: Kill bound the child received.
            seconds: Seconds bound the child received.
            started_ms: Wall-clock spawn time.
            process: The spawned child process handle.
        """
        self.instance = instance
        self.account = account
        self.kills = kills
        self.seconds = seconds
        self.started_ms = started_ms
        self.process = process

    def report(self) -> FleetBotDict:
        """Return the instance's current state for ``GET /bots``.

        Returns:
            The typed report row.
        """
        returncode = self.process.poll()
        return FleetBotDict(
            instance=self.instance,
            account=self.account,
            pid=self.process.pid,
            alive=returncode is None,
            returncode=returncode,
            kills=self.kills,
            seconds=self.seconds,
            started_ms=self.started_ms,
        )


class FleetError(RuntimeError):
    """A fleet operation the HTTP layer maps to a 4xx response."""


class FleetManager:
    """Spawn and track one bot process per instance name."""

    def __init__(self) -> None:
        """Start with an empty registry."""
        self._bots: dict[str, _ManagedBot] = {}

    def spawn(self, *, instance: str, account: str, kills: int, seconds: int) -> FleetBotDict:
        """Spawn one bot child process under an instance namespace.

        Args:
            instance: Instance name; validated against the same
                pattern as ``resolve_bot_instance`` so a bad name is
                rejected here, not by a crashed child.
            account: ``TANKPIT_ACCOUNT`` selector; empty uses the
                accounts.json default.
            kills: Kill bound (0 unbounded).
            seconds: Seconds bound (0 unbounded).

        Returns:
            The spawned instance's report row.

        Raises:
            FleetError: If the name is invalid, already registered and
                alive, or the bounds are negative.
        """
        if not _INSTANCE_NAME.match(instance):
            raise FleetError(
                f"instance {instance!r} is not a valid instance name "
                "(lowercase alphanumeric plus -_, max 32 chars)"
            )
        if kills < 0 or seconds < 0:
            raise FleetError("bounds must be non-negative")
        existing = self._bots.get(instance)
        if existing is not None and existing.process.poll() is None:
            raise FleetError(
                f"instance {instance!r} is already running (pid {existing.process.pid})"
            )
        env = {
            "TANKPIT_BOT_INSTANCE": instance,
            "TANKPIT_BOT_SESSION_KILLS": str(kills),
            "TANKPIT_BOT_SESSION_SECONDS": str(seconds),
        }
        if account:
            env["TANKPIT_ACCOUNT"] = account
        process = service_hooks.spawn_bot_process(env)
        bot = _ManagedBot(
            instance=instance,
            account=account,
            kills=kills,
            seconds=seconds,
            started_ms=top_hooks.get_current_time_ms(),
            process=process,
        )
        self._bots[instance] = bot
        log.info(
            "Fleet: spawned instance %r pid %d (kills=%d seconds=%d)",
            instance,
            process.pid,
            kills,
            seconds,
        )
        return bot.report()

    def report(self) -> list[FleetBotDict]:
        """Return every registered instance's current state.

        Returns:
            Report rows sorted by instance name.
        """
        return [self._bots[name].report() for name in sorted(self._bots)]

    def stop(self, instance: str) -> FleetBotDict:
        """Request a graceful stop: write the instance's stop sentinel.

        The bot's tick loop polls the sentinel and exits at the next
        boundary with a full teardown — scorecard, capture save, and
        archive all happen, exactly as a bounded session ends.

        Args:
            instance: Registered instance name.

        Returns:
            The instance's report row after the request.

        Raises:
            FleetError: If the instance is not registered.
        """
        bot = self._bots.get(instance)
        if bot is None:
            raise FleetError(f"unknown instance {instance!r}")
        sentinel = bot_run_dir(instance) / "STOP"
        top_hooks.write_text(sentinel, "")
        log.info("Fleet: stop requested for %r (sentinel %s)", instance, sentinel)
        return bot.report()

    def remove(self, instance: str) -> FleetBotDict:
        """Drop a finished instance from the registry.

        Args:
            instance: Registered instance name.

        Returns:
            The removed instance's final report row.

        Raises:
            FleetError: If the instance is unknown or still alive —
                the fleet never silently kills; stop it first.
        """
        bot = self._bots.get(instance)
        if bot is None:
            raise FleetError(f"unknown instance {instance!r}")
        if bot.process.poll() is None:
            raise FleetError(f"instance {instance!r} is still running; stop it first")
        del self._bots[instance]
        return bot.report()


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
        "pid": bot["pid"],
        "alive": bot["alive"],
        "returncode": bot["returncode"],
        "kills": bot["kills"],
        "seconds": bot["seconds"],
        "started_ms": bot["started_ms"],
    }


def parse_spawn_request(body: bytes) -> tuple[str, str, int, int]:
    """Parse the ``POST /bots`` body.

    Args:
        body: Raw request body.

    Returns:
        ``(instance, account, kills, seconds)``.

    Raises:
        JSONTypeError: Malformed JSON or wrong field types.
    """
    data = narrow_json_to_dict(load_json_bytes(body))
    instance = narrow_json_to_str(data.get("instance", ""))
    account = narrow_json_to_str(data.get("account", ""))
    kills = narrow_json_to_int(data.get("kills", 0))
    seconds = narrow_json_to_int(data.get("seconds", 0))
    return instance, account, kills, seconds


def make_fleet_app(manager: FleetManager) -> web.Application:
    """Build the fleet manager's aiohttp application.

    Args:
        manager: The fleet registry the routes operate on.

    Returns:
        The configured ``aiohttp.web.Application``.
    """

    async def list_bots(request: web.Request) -> web.Response:
        """``GET /bots`` — every instance's current state."""
        _ = request
        rows = [encode_fleet_bot(bot) for bot in manager.report()]
        return web.Response(
            text=dump_json_str({"bots": rows}, indent=1),
            content_type="application/json",
        )

    async def spawn_bot(request: web.Request) -> web.Response:
        """``POST /bots`` — spawn one instance."""
        try:
            instance, account, kills, seconds = parse_spawn_request(await request.read())
            row = manager.spawn(instance=instance, account=account, kills=kills, seconds=seconds)
        except (JSONTypeError, ValueError) as error:
            log.warning("Fleet: rejected spawn request (400): %s", error)
            return web.Response(status=400, text=f"bad spawn request: {error}")
        except FleetError as error:
            log.warning("Fleet: refused spawn (409): %s", error)
            return web.Response(status=409, text=str(error))
        return web.Response(
            status=201,
            text=dump_json_str(encode_fleet_bot(row), indent=1),
            content_type="application/json",
        )

    async def stop_bot(request: web.Request) -> web.Response:
        """``POST /bots/{instance}/stop`` — graceful stop via sentinel."""
        try:
            row = manager.stop(request.match_info["instance"])
        except FleetError as error:
            log.warning("Fleet: refused stop (404): %s", error)
            return web.Response(status=404, text=str(error))
        return web.Response(
            text=dump_json_str(encode_fleet_bot(row), indent=1),
            content_type="application/json",
        )

    async def remove_bot(request: web.Request) -> web.Response:
        """``DELETE /bots/{instance}`` — drop a finished instance."""
        try:
            row = manager.remove(request.match_info["instance"])
        except FleetError as error:
            status = 409 if "still running" in str(error) else 404
            log.warning("Fleet: refused remove (%d): %s", status, error)
            return web.Response(status=status, text=str(error))
        return web.Response(
            text=dump_json_str(encode_fleet_bot(row), indent=1),
            content_type="application/json",
        )

    app = web.Application()
    app.router.add_get("/bots", list_bots)
    app.router.add_post("/bots", spawn_bot)
    app.router.add_post("/bots/{instance}/stop", stop_bot)
    app.router.add_delete("/bots/{instance}", remove_bot)
    return app


def main() -> None:
    """Run the ``tankpit-fleet`` manager until interrupted."""
    service_hooks.load_dotenv()
    port = resolve_fleet_port()
    manager = FleetManager()
    app = make_fleet_app(manager)
    log.info("tankpit-fleet listening on 127.0.0.1:%d", port)
    service_hooks.run_web_app(app, host="127.0.0.1", port=port)


__all__ = [
    "FLEET_PORT_DEFAULT",
    "FleetBotDict",
    "FleetError",
    "FleetManager",
    "encode_fleet_bot",
    "main",
    "make_fleet_app",
    "parse_spawn_request",
    "resolve_fleet_port",
]
