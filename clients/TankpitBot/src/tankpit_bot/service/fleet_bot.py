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
    NAME the operator picked is converted here â€” the index into
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
        return FleetBotDict(
            instance=self.instance,
            account=self.account,
            role=self.role,
            room=self.room,
            troop=self.troop,
            pid=self.process.pid,
            alive=self.process.is_running(),
            returncode=self.process.exit_code(),
            kills=self.kills,
            seconds=self.seconds,
            started_ms=self.started_ms,
        )


__all__ = [
    "FleetBotDict",
    "_ManagedBot",
    "_child_environment",
]
