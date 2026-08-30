"""Fleet domain: the instance registry behind the HTTP surface.

Owns spawn/stop/restart/remove/stats over one bot child process per
instance name, and nothing about HTTP. Each bot is a CHILD PROCESS, so
an orchestrator dying can never kill a live tank.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
)
from platform_core.logging import get_logger
from typing_extensions import TypedDict

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.browser.accounts import _ACCOUNTS_PATH, load_accounts
from tankpit_bot.fleetshare.types import FLEET_ROLES, FleetRole
from tankpit_bot.runtime_artifacts import _INSTANCE_NAME, bot_run_dir
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service.fleet_telemetry import FleetTelemetry
from tankpit_bot.types.constants import TROOP_COLOR_NAMES
from tankpit_bot.types.rooms import LOBBY_ROOMS

log = get_logger(__name__)

FLEET_PORT_DEFAULT = 27300


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
        role: Resolved :data:`~tankpit_bot.fleetshare.types.FleetRole`
            the child was spawned with ([[fleet-coordination]]).
        room: ``TANKPIT_ROOM`` the child was spawned with (empty means
            the default Practice room).
        troop: Tank color name the child was spawned with (empty means
            the account's own default for that map).
        pid: Child process id.
        alive: Whether the process is still running at report time.
        returncode: Exit code once dead; ``None`` while alive.
        kills: Kill bound the child was spawned with (0 unbounded).
        seconds: Seconds bound the child was spawned with (0 unbounded).
        started_ms: Wall-clock spawn time.
    """

    instance: str
    account: str
    role: FleetRole
    room: str
    troop: str
    pid: int
    alive: bool
    returncode: int | None
    kills: int
    seconds: int
    started_ms: int


def _child_environment(
    instance: str,
    kills: int,
    seconds: int,
    resolved_role: str,
    account: str,
    room: str,
    troop: str,
) -> dict[str, str]:
    """Build one child's spawn environment.

    ``TANKPIT_ROLE`` is always explicit: the child inherits the
    manager's whole environment, and a role lingering there must never
    silently re-role the entire fleet. Empty account, room and troop
    omit their selectors so the child keeps its defaults (accounts.json
    default; the Practice room; the account's own tank color for that
    map).

    ``TANKPIT_TROOP`` goes over the wire as the team id, so the color
    NAME the operator picked is converted here — the index into
    :data:`~tankpit_bot.types.constants.TROOP_COLOR_NAMES` IS that id.

    Args:
        instance: Validated instance name.
        kills: Kill bound.
        seconds: Seconds bound.
        resolved_role: Resolved fleet role.
        account: Account selector ("" = default).
        room: Room selector ("" = default).
        troop: Tank color name ("" = the account's default).

    Returns:
        Environment overrides for the spawned child.
    """
    env = {
        "TANKPIT_BOT_INSTANCE": instance,
        "TANKPIT_BOT_SESSION_KILLS": str(kills),
        "TANKPIT_BOT_SESSION_SECONDS": str(seconds),
        "TANKPIT_ROLE": resolved_role,
    }
    if account:
        env["TANKPIT_ACCOUNT"] = account
    if room:
        env["TANKPIT_ROOM"] = room
    if troop:
        env["TANKPIT_TROOP"] = str(TROOP_COLOR_NAMES.index(troop))
    return env


class _ManagedBot:
    """Registry entry pairing spawn metadata with the live process."""

    def __init__(
        self,
        *,
        instance: str,
        account: str,
        role: FleetRole,
        room: str,
        troop: str,
        kills: int,
        seconds: int,
        started_ms: int,
        process: service_hooks.SpawnedProcessProtocol,
    ) -> None:
        """Bind one spawned bot to its metadata.

        Args:
            instance: Validated instance name.
            account: Account selector the child received.
            role: Resolved fleet role the child received.
            room: Room selector the child received ("" = default).
            troop: Tank color name the child received ("" = default).
            kills: Kill bound the child received.
            seconds: Seconds bound the child received.
            started_ms: Wall-clock spawn time.
            process: The spawned child process handle.
        """
        self.instance = instance
        self.account = account
        self.role = role
        self.room = room
        self.troop = troop
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
            role=self.role,
            room=self.room,
            troop=self.troop,
            pid=self.process.pid,
            alive=returncode is None,
            returncode=returncode,
            kills=self.kills,
            seconds=self.seconds,
            started_ms=self.started_ms,
        )


class FleetError(RuntimeError):
    """A fleet operation the HTTP layer maps to a 4xx response."""


def _resolve_role(role: str) -> FleetRole:
    """Resolve a spawn request's role selector to a fleet role.

    Args:
        role: Role selector; empty means fighter — the full doctrine
            is the primary configuration, a gatherer is an explicit
            operator choice ([[fleet-coordination]]).

    Returns:
        The resolved role.

    Raises:
        FleetError: If the selector is not a fleet role.
    """
    candidate = role or "fighter"
    for known in FLEET_ROLES:
        if candidate == known:
            return known
    known_roles = ", ".join(FLEET_ROLES)
    raise FleetError(f"role {role!r} is not a fleet role (one of: {known_roles})")


def _resolve_troop(troop: str) -> str:
    """Resolve a spawn request's color selector to a tank color name.

    Args:
        troop: Color name, or ``""`` to keep the account's own default
            tank color for the map it joins.

    Returns:
        The validated color name, or ``""``.

    Raises:
        FleetError: If the selector is not a tank color.
    """
    if troop == "":
        return ""
    for known in TROOP_COLOR_NAMES:
        if troop == known:
            return known
    known_colors = ", ".join(TROOP_COLOR_NAMES)
    raise FleetError(f"troop {troop!r} is not a tank color (one of: {known_colors})")


class FleetManager:
    """Spawn and track one bot process per instance name."""

    def __init__(self) -> None:
        """Start with an empty registry."""
        self._bots: dict[str, _ManagedBot] = {}
        self._telemetry = FleetTelemetry()

    def accounts(self) -> list[str]:
        """Return the configured account usernames.

        Accounts are CONFIG (``accounts.json``), never free text — the
        spawn surface only accepts a selector from this list, and the
        control page renders it as a dropdown. Usernames only;
        passwords never leave the file.

        Returns:
            Usernames in file order (the first is the default), empty
            when no accounts file exists.
        """
        if not top_hooks.path_exists(_ACCOUNTS_PATH):
            return []
        return [account["username"] for account in load_accounts(_ACCOUNTS_PATH)]

    def rooms(self) -> list[str]:
        """Return the room selectors the control page offers.

        The lobby lists two rooms, and the world's display name
        carries the current map, so the page offers the durable
        PREFIXES the join resolver matches on
        ([[game-rules]], :mod:`tankpit_bot.types.rooms`) rather than
        asking a human to type a name that rotates. Spawn still
        accepts any selector: this list is what the dropdown shows,
        not a closed set.

        Returns:
            Room selectors in lobby order; the first is the default.
        """
        return list(LOBBY_ROOMS)

    def troops(self) -> list[str]:
        """Return the tank colors the control page offers.

        Four colors, in TEAM ID order — the index is the wire's team
        id, so the list doubles as the name->id table the spawn
        environment converts through. An account holds FOUR TANKS PER
        WORLD, one per color, each with its own RANK, inventory, fuel
        and points (awards alone are shared) — so picking a color
        picks WHICH TANK plays, not a skin, and a fresh color starts
        that world from scratch. The worlds are independent: four on
        the main world plus four on Practice. Switching is throttled
        per world — 5 minutes between exiting a world and re-entering
        it on a different color ([[game-rules]]).

        Returns:
            Color names in team-id order.
        """
        return list(TROOP_COLOR_NAMES)

    def derive_instance(self, account: str) -> str:
        """Derive the instance name from the account — programmatic, reliable.

        One account can hold at most one live tank (the game refuses a
        second login), so the account IS the natural bot identity: the
        instance is its username lowered and sanitized to the
        namespace grammar. No account configured falls back to
        ``bot``. Callers may still name instances explicitly through
        the API; the control page never asks a human to invent one.

        Args:
            account: Selected account username, empty for the default.

        Returns:
            A valid instance name.
        """
        configured = self.accounts()
        source = account or (configured[0] if configured else "bot")
        cleaned = "".join(
            ch if ch.isascii() and (ch.isalnum() or ch in "-_") else "-" for ch in source.lower()
        )[:32]
        if not cleaned or not (cleaned[0].isascii() and cleaned[0].isalnum()):
            cleaned = f"b{cleaned}"[:32]
        return cleaned

    def spawn(
        self,
        *,
        instance: str,
        account: str,
        kills: int,
        seconds: int,
        role: str = "",
        room: str = "",
        troop: str = "",
    ) -> FleetBotDict:
        """Spawn one bot child process under an instance namespace.

        Args:
            instance: Instance name; validated against the same
                pattern as ``resolve_bot_instance`` so a bad name is
                rejected here, not by a crashed child.
            account: ``TANKPIT_ACCOUNT`` selector; empty uses the
                accounts.json default.
            kills: Kill bound (0 unbounded).
            seconds: Seconds bound (0 unbounded).
            room: ``TANKPIT_ROOM`` selector; empty keeps the child's
                default (Practice). Cross-room fleets stay safe: the
                knowledge exchange merges same-room reports only
                (2026-08-26).
            role: Fleet role selector; empty means fighter — the full
                doctrine is the primary configuration, a gatherer is
                an explicit operator choice ([[fleet-coordination]]).
            troop: Tank color name; empty keeps the account's own
                default tank for the map. Accounts hold one tank per
                color, so this picks which tank plays.

        Returns:
            The spawned instance's report row.

        Raises:
            FleetError: If the name is invalid, already registered and
                alive, the bounds are negative, or the role or troop
                is not a known one.
        """
        if not instance:
            instance = self.derive_instance(account)
        if not _INSTANCE_NAME.match(instance):
            raise FleetError(
                f"instance {instance!r} is not a valid instance name "
                "(lowercase alphanumeric plus -_, max 32 chars)"
            )
        if kills < 0 or seconds < 0:
            raise FleetError("bounds must be non-negative")
        resolved_role = _resolve_role(role)
        resolved_troop = _resolve_troop(troop)
        if account:
            configured = self.accounts()
            if account not in configured:
                known = ", ".join(configured) or "none configured"
                raise FleetError(
                    f"account {account!r} is not in accounts.json (accounts are "
                    f"config, not free text; configured: {known})"
                )
        configured = self.accounts()
        resolved_account = account or (configured[0] if configured else "")
        for other in self._bots.values():
            other_account = other.account or (configured[0] if configured else "")
            if (
                resolved_account
                and other_account == resolved_account
                and other.process.poll() is None
            ):
                raise FleetError(
                    f"account {resolved_account or 'default'!r} already has a live bot "
                    f"({other.instance!r}, pid {other.process.pid}) - the game refuses "
                    "a second login on the same account"
                )
        existing = self._bots.get(instance)
        if existing is not None and existing.process.poll() is None:
            raise FleetError(
                f"instance {instance!r} is already running (pid {existing.process.pid})"
            )
        process = service_hooks.spawn_bot_process(
            _child_environment(
                instance, kills, seconds, resolved_role, account, room, resolved_troop
            )
        )
        bot = _ManagedBot(
            instance=instance,
            account=account,
            role=resolved_role,
            room=room,
            troop=resolved_troop,
            kills=kills,
            seconds=seconds,
            started_ms=top_hooks.get_current_time_ms(),
            process=process,
        )
        self._bots[instance] = bot
        log.info(
            "Fleet: spawned instance %r pid %d (role=%s kills=%d seconds=%d)",
            instance,
            process.pid,
            resolved_role,
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

    def restart(self, instance: str) -> FleetBotDict:
        """Respawn a finished instance with the parameters it had.

        ALL of them: account, bounds, role, room, and troop. The room was
        missing here from the day it was added to spawn (2026-08-26)
        until 2026-08-28, so a restart silently relocated the bot to
        the default Practice room — the row said ``World``, the child
        joined Practice, and only the run log disagreed.

        The fleet never silently kills: restarting a LIVE instance is
        refused — stop it first, let the teardown run, then restart.

        Args:
            instance: Registered instance name.

        Returns:
            The respawned instance's report row.

        Raises:
            FleetError: If the instance is unknown or still alive.
        """
        bot = self._bots.get(instance)
        if bot is None:
            raise FleetError(f"unknown instance {instance!r}")
        if bot.process.poll() is None:
            raise FleetError(f"instance {instance!r} is still running; stop it first")
        return self.spawn(
            instance=instance,
            account=bot.account,
            kills=bot.kills,
            seconds=bot.seconds,
            role=bot.role,
            room=bot.room,
            troop=bot.troop,
        )

    def stats_gate(self, instance: str) -> None:
        """Refuse telemetry reads for unregistered instances.

        Args:
            instance: Candidate instance name.

        Raises:
            FleetError: If the instance is not registered.
        """
        if instance not in self._bots:
            raise FleetError(f"unknown instance {instance!r}")

    def stats(self, instance: str) -> JSONObject:
        """Summarize a registered instance's latest run from its events.

        The digest reduction (:mod:`fleet_telemetry`) — the same truth
        table ``make digest`` prints, reduced to the fields the control
        page shows, cached so 1 s page polling costs one events parse
        per cache window. Works on live runs and on crashed ones.

        Args:
            instance: Registered instance name.

        Returns:
            ``{"available": False}`` when the instance has produced no
            events yet, else the summary with ``"available": True``.

        Raises:
            FleetError: If the instance is not registered.
        """
        if instance not in self._bots:
            raise FleetError(f"unknown instance {instance!r}")
        return self._telemetry.stats(instance)

    def activity(self, instance: str) -> JSONObject:
        """Return the live tail of a registered instance's run.

        Current bot state, tick, fuel, and the last few AI/WORLD/STATE
        lines (:mod:`fleet_telemetry`) — what the bot is doing right
        now, for the control page's activity feed.

        Args:
            instance: Registered instance name.

        Returns:
            ``{"available": False}`` before the first events, else the
            tail with ``"available": True``.

        Raises:
            FleetError: If the instance is not registered.
        """
        if instance not in self._bots:
            raise FleetError(f"unknown instance {instance!r}")
        return self._telemetry.activity(instance)

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


__all__ = [
    "FLEET_PORT_DEFAULT",
    "FleetBotDict",
    "FleetError",
    "FleetManager",
    "log",
    "resolve_fleet_port",
]
