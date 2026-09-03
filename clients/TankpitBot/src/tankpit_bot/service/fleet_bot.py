"""One managed bot child: its report row and its spawn environment.

Split from :mod:`tankpit_bot.service.fleet_manager` 2026-08-29 at the
600-line ceiling. The cut is by role, not by size: this module is
everything about ONE child process -- the row the HTTP surface reports,
the environment it was launched with, and the handle it is polled
through -- while ``fleet_manager`` keeps the registry that owns many of
them and the lifecycle operations over that registry.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from tankpit_bot.fleetshare.types import FleetRole
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.types.constants import TROOP_COLOR_NAMES


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
        doctrine: Engagement doctrine the child was spawned with
            (empty means the child's own default, skirmish). Reported
            so an operator can SEE what a running bot is fighting
            under -- a selector with no readback is a setting you
            cannot confirm took.
        troop: Tank color name the child was spawned with (empty means
            the account's own default for that map).
        pid: Child process id.
        alive: Whether the process is still running at report time.
        returncode: Exit code once dead; ``None`` while alive.
        kills: Kill bound the child was spawned with (0 unbounded).
        seconds: Seconds bound the child was spawned with (0 unbounded).
        started_ms: Wall-clock spawn time.
        service_port: Port this child's own service listens on, INSIDE
            the manager's container. Reported so the manager can reach
            it to relay ``/video``; never published, and not something a
            caller outside the container can dial.
    """

    instance: str
    account: str
    role: FleetRole
    room: str
    troop: str
    doctrine: str
    pid: int
    alive: bool
    returncode: int | None
    kills: int
    seconds: int
    started_ms: int
    service_port: int


def _child_environment(
    *,
    instance: str,
    kills: int,
    seconds: int,
    resolved_role: str,
    account: str,
    room: str,
    troop: str,
    doctrine: str,
    human_min_rank: int,
    service_port: int,
) -> dict[str, str]:
    """Build one child's spawn environment.

    ``TANKPIT_ROLE`` and ``TANKPIT_BOT_HUMAN_MIN_RANK`` are always
    explicit: the child inherits the manager's whole environment, and
    a value lingering there must never silently re-role the fleet or
    quietly lower the rank floor a room's policy set. The floor in
    particular is a POLICY the fleet owns per room
    (:func:`~tankpit_bot.service.fleet_config.resolve_human_min_rank`),
    so a global ``TANKPIT_BOT_HUMAN_MIN_RANK`` in the operator's
    ``.env`` does NOT reach a fleet-spawned bot -- it cannot, because
    one global cannot say "lieutenant on World, recruit on Practice".
    Bots started outside the fleet (``make run``, the probes) still
    read that variable normally. Empty account, room, troop and
    doctrine omit their selectors so the child keeps its defaults
    (accounts.json default; the Practice room; the account's own tank
    color for that map; skirmish).

    KEYWORD-ONLY on purpose: five of these are strings and four are
    adjacent selectors, so a positional call that transposes two is
    silent -- the bot spawns, joins somewhere, and fights under a
    doctrine nobody chose.

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
        doctrine: Engagement doctrine ("" = skirmish, the unset
            behaviour).
        human_min_rank: Lowest human rank this bot may open a fight
            on, resolved from the room it is joining.
        service_port: Port this child's own service binds, unique
            across live children.

    Returns:
        Environment overrides for the spawned child.
    """
    env = {
        "TANKPIT_BOT_INSTANCE": instance,
        "TANKPIT_BOT_SESSION_KILLS": str(kills),
        "TANKPIT_BOT_SESSION_SECONDS": str(seconds),
        "TANKPIT_ROLE": resolved_role,
        "TANKPIT_BOT_HUMAN_MIN_RANK": str(human_min_rank),
        # A fleet child runs the SERVICE, not the bare bot, so it
        # serves its own /video and /frame off the same tick loop the
        # HUD already rides. The port is explicit because two children
        # sharing one would serve each other's video; the session
        # starts on its own because that is what a fleet child is for.
        "TANKPIT_BOT_SERVICE_PORT": str(service_port),
    }
    if account:
        env["TANKPIT_ACCOUNT"] = account
    if room:
        env["TANKPIT_ROOM"] = room
    if troop:
        env["TANKPIT_TROOP"] = str(TROOP_COLOR_NAMES.index(troop))
    if doctrine:
        env["TANKPIT_DOCTRINE"] = doctrine
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
        doctrine: str,
        kills: int,
        seconds: int,
        started_ms: int,
        service_port: int,
        process: service_hooks.SpawnedProcessProtocol,
    ) -> None:
        """Bind one spawned bot to its metadata.

        Args:
            instance: Validated instance name.
            account: Account selector the child received.
            role: Resolved fleet role the child received.
            room: Room selector the child received ("" = default).
            troop: Tank color name the child received ("" = default).
            doctrine: Engagement doctrine the child received.
            kills: Kill bound the child received.
            seconds: Seconds bound the child received.
            started_ms: Wall-clock spawn time.
            service_port: Port this child's service bound.
            process: The spawned child process handle.
        """
        self.instance = instance
        self.account = account
        self.role = role
        self.room = room
        self.troop = troop
        self.doctrine = doctrine
        self.kills = kills
        self.seconds = seconds
        self.started_ms = started_ms
        self.service_port = service_port
        self.process = process

    def report(self) -> FleetBotDict:
        """Return the instance's current state for ``GET /bots``.

        Returns:
            The typed report row.
        """
        return FleetBotDict(
            instance=self.instance,
            account=self.account,
            role=self.role,
            room=self.room,
            troop=self.troop,
            doctrine=self.doctrine,
            pid=self.process.pid,
            alive=self.process.is_running(),
            returncode=self.process.exit_code(),
            kills=self.kills,
            seconds=self.seconds,
            started_ms=self.started_ms,
            service_port=self.service_port,
        )


__all__ = [
    "FleetBotDict",
    "_ManagedBot",
    "_child_environment",
]
