"""What a stranger may ask the fleet for, and nothing else.

``austinwagner.org/tankpit`` is open: no login, no key, no account of
the visitor's own. Everything the operator surface offers is therefore
off the table, and this module is the whole of what is left — start one
bounded Practice bot, see how many are up, watch one play.

The narrowing is by CONSTRUCTION, not by validation of what was asked.
A demo spawn reads no request body at all, so there is no field for a
caller to put an account, a room, a colour, a doctrine or an unbounded
session into; the slot name is generated here, so a caller cannot name
an instance belonging to the operator's own fleet. Reads narrow the
same way: every demo route resolves its id through
:func:`demo_slot_or_refuse` before it touches the registry, so the
public video relay cannot be pointed at a private bot by guessing.

What a demo row SAYS is likewise the whole list: a slot name, whether
it is alive, and how long it has been up. Operator instance names are
derived from account usernames
(:func:`~tankpit_bot.service.fleet_config.derive_instance`), so
reporting one publicly would publish the username; a demo slot is a
number and says nothing. Pids, ports, exit codes, accounts, rooms and
colours all stay on the operator surface where a reader is known.

There is an encoder here and deliberately no decoder. The only consumer
of this contract is the demo page's JavaScript; a Python decoder would
have no caller, and a decoder whose only exercise is its own test
proves nothing about the wire.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject, JSONValue
from typing_extensions import TypedDict

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.service.fleet_bot import FleetBotDict
from tankpit_bot.service.fleet_config import configured_accounts
from tankpit_bot.service.fleet_error import FleetError
from tankpit_bot.service.fleet_manager import FleetManager
from tankpit_bot.types.rooms import DEFAULT_LOBBY_ROOM

DEMO_MAX_BOTS = 5
"""Most bots the public demo will ever have in play at once.

A visitor's button is not a fleet-sizing decision, and five tanks is
already a busy screen. The real ceiling is usually lower — an account
can hold only one live tank, so :func:`demo_capacity` takes whichever
of this number and the configured account count is smaller.
"""

DEMO_SESSION_SECONDS = 900
"""How long a demo bot plays before ending itself.

Bounded on purpose and not configurable from outside: an unbounded
session started by an anonymous click has nobody to stop it. Fifteen
minutes is long enough to watch a fight and short enough that a visitor
who wanders off returns the account to the pool on their behalf.
"""

DEMO_SLOT_PREFIX = "demo-"
"""Namespace every demo instance name is built from.

The prefix is what makes the public surface's reads decidable: a name
either matches ``demo-<n>`` for ``n`` in 1..:data:`DEMO_MAX_BOTS` or it
is not the demo's to serve.
"""


class DemoBotDict(TypedDict):
    """One demo bot, as a stranger may see it.

    Attributes:
        slot: The public slot name (``demo-1`` .. ``demo-N``). Not the
            account-derived instance name the operator surface uses.
        alive: Whether the bot is still playing.
        uptime_seconds: Whole seconds since it was spawned.
    """

    slot: str
    alive: bool
    uptime_seconds: int


class DemoFleetDict(TypedDict):
    """The public demo's whole state.

    Attributes:
        running: How many demo bots are alive right now.
        capacity: How many may be alive at once, from
            :func:`demo_capacity`. The page renders ``running`` of
            ``capacity`` and disables its button when they are equal.
        draining: Whether the manager is shutting down. A draining
            fleet accepts no new bots, and a page that cannot say so
            shows a button that refuses every press without explaining
            itself.
        bots: The live demo bots, in slot order.
    """

    running: int
    capacity: int
    draining: bool
    bots: list[DemoBotDict]


def demo_capacity() -> int:
    """Return how many demo bots may run at once.

    Returns:
        The smaller of :data:`DEMO_MAX_BOTS` and the number of
        configured accounts — one account can hold one live tank, so a
        two-account machine has a capacity of two however high the
        ceiling is set. Zero when nothing is configured, which is the
        honest answer for a machine that cannot log in at all.
    """
    return min(DEMO_MAX_BOTS, len(configured_accounts()))


def demo_slots() -> list[str]:
    """Return the slot names this machine can fill, in order.

    Returns:
        ``demo-1`` .. ``demo-N`` for N = :func:`demo_capacity`.
    """
    return [f"{DEMO_SLOT_PREFIX}{index + 1}" for index in range(demo_capacity())]


def demo_slot_or_refuse(candidate: str) -> str:
    """Return ``candidate`` when it names a demo slot, else refuse.

    Matched against the slot GRAMMAR rather than against
    :func:`demo_slots`, so the answer does not move when accounts.json
    does: a bot spawned into ``demo-5`` stays watchable after the
    account pool shrinks and the capacity with it. Shrinking config
    must not make a live bot unreadable — it only stops the next spawn.

    Args:
        candidate: The id a caller supplied.

    Returns:
        The candidate, unchanged.

    Raises:
        FleetError: If the candidate is not a demo slot name. The
            operator's instances are named after accounts and are
            reachable only from the operator surface; this is the check
            that keeps the public routes off them.
    """
    for index in range(DEMO_MAX_BOTS):
        if candidate == f"{DEMO_SLOT_PREFIX}{index + 1}":
            return candidate
    raise FleetError(
        f"{candidate!r} is not a demo slot "
        f"({DEMO_SLOT_PREFIX}1 .. {DEMO_SLOT_PREFIX}{DEMO_MAX_BOTS})"
    )


def _public_row(row: FleetBotDict) -> DemoBotDict:
    """Reduce one report row to what the public may read.

    Args:
        row: The manager's full report row.

    Returns:
        The public projection. Every field the operator surface carries
        and this one does not — account, room, troop, doctrine, role,
        pid, service port, exit code, bounds — is dropped by being
        absent from the construction, not by being blanked afterwards.
    """
    elapsed_ms = top_hooks.get_current_time_ms() - row["started_ms"]
    return DemoBotDict(
        slot=row["instance"],
        alive=row["alive"],
        uptime_seconds=elapsed_ms // 1000,
    )


def demo_fleet(manager: FleetManager) -> DemoFleetDict:
    """Report the public demo's state.

    Live demo bots only. A finished slot is noise on a page whose job is
    to show tanks playing, and the operator surface is where a run's
    ending is read.

    Args:
        manager: The fleet registry to read.

    Returns:
        The public snapshot.
    """
    rows = [
        _public_row(row)
        for row in manager.report()
        if row["alive"] and row["instance"].startswith(DEMO_SLOT_PREFIX)
    ]
    return DemoFleetDict(
        running=len(rows),
        capacity=demo_capacity(),
        draining=manager.draining(),
        bots=rows,
    )


def demo_spawn(manager: FleetManager) -> DemoBotDict:
    """Start one bounded Practice bot in the next free demo slot.

    Takes no parameters, by design: see the module docstring. The room
    is stated rather than left to the child's default so the rank floor
    it implies is stated too — Practice is consequence-free, and a demo
    that silently inherited a World default would put anonymous clicks
    behind real tanks.

    Args:
        manager: The fleet registry to spawn into.

    Returns:
        The new bot's public row.

    Raises:
        FleetError: If the fleet is draining, no accounts are
            configured, every account already holds a live tank, or
            every slot is full.
    """
    if manager.draining():
        raise FleetError("the fleet is shutting down; nothing new starts now")
    if demo_capacity() == 0:
        raise FleetError("no accounts are configured on this machine; the demo has none to play")
    held = manager.live_accounts()
    account = next((name for name in configured_accounts() if name not in held), "")
    if not account:
        raise FleetError(
            f"every configured account already has a tank in play ({len(held)} running)"
        )
    running = set(manager.live_instances())
    slot = next((name for name in demo_slots() if name not in running), "")
    if not slot:
        raise FleetError(f"the demo is full ({demo_capacity()} bots); wait for one to finish")
    return _public_row(
        manager.spawn(
            instance=slot,
            account=account,
            kills=0,
            seconds=DEMO_SESSION_SECONDS,
            role="",
            room=DEFAULT_LOBBY_ROOM,
            troop="",
            doctrine="",
        )
    )


def encode_demo_bot(bot: DemoBotDict) -> JSONObject:
    """Encode one public row for the wire.

    Args:
        bot: The public row.

    Returns:
        JSON-serializable object.
    """
    return {
        "slot": bot["slot"],
        "alive": bot["alive"],
        "uptime_seconds": bot["uptime_seconds"],
    }


def encode_demo_fleet(snapshot: DemoFleetDict) -> JSONObject:
    """Encode the whole public snapshot for the wire.

    Args:
        snapshot: The snapshot to encode.

    Returns:
        JSON-serializable object.
    """
    rows: list[JSONValue] = [encode_demo_bot(bot) for bot in snapshot["bots"]]
    return {
        "running": snapshot["running"],
        "capacity": snapshot["capacity"],
        "draining": snapshot["draining"],
        "bots": rows,
    }


__all__ = [
    "DEMO_MAX_BOTS",
    "DEMO_SESSION_SECONDS",
    "DEMO_SLOT_PREFIX",
    "DemoBotDict",
    "DemoFleetDict",
    "demo_capacity",
    "demo_fleet",
    "demo_slot_or_refuse",
    "demo_slots",
    "demo_spawn",
    "encode_demo_bot",
    "encode_demo_fleet",
]
