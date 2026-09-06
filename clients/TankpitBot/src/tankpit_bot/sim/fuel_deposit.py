"""The fuel-deposit law: what one ``0x44`` command does to the world.

The byte sat in the archive for months as ``CMD_UNMODELLED_COMBAT`` —
"type 6, shoot-shaped, four distinct payloads, no law". It was neither
unmodellable nor combat. The client class is ``Wb`` (code ``'D'``,
:mod:`js-source-map`), its serializer writes
``[type][0x44][amount_lo][amount_hi][x][y]``, and reading the AMOUNT
where the shot keeps its coordinates is the whole reason the payloads
looked arbitrary: ``06446400aaae`` is not four random bytes, it is
"deposit 100 at (170, 174)".

The wire choreography was byte-mined a month before any of this, on
2026-08-03, from the same six windows ([[fuel-system]]) — what was
missing was only the COMMAND that causes it, which is why the sim
could see deposits in the archive and not perform one.

The law itself is a transfer with a floor. A tank moves ``amount``
units of its own fuel into a tile's cache, and the server never lets
it strand itself: at most ``fuel - DEPOSIT_FLOOR`` leaves the tank.
That floor is not inferred from this command — it is the same
:data:`~tankpit_bot.physics.capacity.DEPOSIT_FLOOR` measured at four
ranks in July 2026 from the user's own max deposits, and the archive
window that requested 294 while holding 294 (194 landed, 100 stayed)
is a fifth reading of it.

This module RESOLVES: it owns the mutation, and
:func:`tankpit_bot.sim.narrate.resources.narrate_fuel_deposit` is a
pure function of the outcome it returns
([[physics-module-roadmap]]).

No codec: the outcome never leaves the process. It is built inside one
command's routing and consumed by the narrator in the same call, so an
encode/decode pair would have exactly one caller — its own round-trip
test. The codec rule binds at serialization boundaries
([[coding-standards]]).
"""

from __future__ import annotations

from typing import TypedDict

from tankpit_bot.physics.capacity import DEPOSIT_FLOOR
from tankpit_bot.sim.world import SimContainerDict, SimWorldDict


class FuelDepositOutcomeDict(TypedDict):
    """The resolved facts of one fuel deposit.

    Attributes:
        tank_id: The depositing tank.
        x: The destination tile's X.
        y: The destination tile's Y.
        requested: The amount the client asked to deposit, verbatim.
        deposited: The amount that actually left the tank, after the
            floor. Equal to ``requested`` in five of the six archived
            deposits and short of it in the sixth.
        fuel_total: The tank's absolute fuel AFTER the transfer — the
            0x64 payload, captured rather than re-read.
        tile_volume: The tile's absolute volume after the transfer —
            the container record's ``remaining_volume``. Absolute, not
            a delta: production overwrites its belief about the
            container with whatever the record says.
    """

    tank_id: int
    x: int
    y: int
    requested: int
    deposited: int
    fuel_total: int
    tile_volume: int


def _container_at(world: SimWorldDict, x: int, y: int) -> SimContainerDict | None:
    """Find the container record on a tile.

    Args:
        world: Simulated world. Read only.
        x: Tile X.
        y: Tile Y.

    Returns:
        The container record, or ``None`` when the tile is bare
        ground. Bare ground is the ordinary case for a deposit — all
        six archived deposits landed on tiles with no prior volume.
    """
    for container in world["containers"]:
        if (container["x"], container["y"]) == (x, y):
            return container
    return None


def resolve_fuel_deposit(
    world: SimWorldDict,
    tank_id: int,
    x: int,
    y: int,
    requested: int,
) -> FuelDepositOutcomeDict:
    """Move fuel from a tank into a tile's cache. MUTATES the world.

    The transfer is ``min(requested, fuel - DEPOSIT_FLOOR)``, floored
    at zero so a tank at or below the floor deposits nothing rather
    than a negative amount. A deposit onto bare ground CREATES the
    container record: the tile now holds fuel, and the pickup law
    reads the same list.

    A new record starts undotted. ``dotted`` is exposure memory, set
    the first time radar reveals a container holding
    ``MAP_DOT_MIN_VOLUME`` or more — depositing is not a reveal, and
    nothing in the archive shows a deposit joining the map atlas.

    Args:
        world: Simulated world, positioned at the destination tile.
            MUTATED: the tank loses fuel and the tile gains it.
        tank_id: The depositing tank.
        x: The destination tile's X.
        y: The destination tile's Y.
        requested: The amount the client asked to deposit.

    Returns:
        The snapshot the deposit narrator is a pure function of.
    """
    tank = world["tanks"][tank_id]
    deposited = max(0, min(requested, tank["fuel"] - DEPOSIT_FLOOR))
    tank["fuel"] -= deposited
    container = _container_at(world, x, y)
    if container is None:
        container = SimContainerDict(x=x, y=y, volume=0, dotted=False)
        world["containers"].append(container)
    container["volume"] += deposited
    return FuelDepositOutcomeDict(
        tank_id=tank_id,
        x=x,
        y=y,
        requested=requested,
        deposited=deposited,
        fuel_total=tank["fuel"],
        tile_volume=container["volume"],
    )


__all__ = [
    "FuelDepositOutcomeDict",
    "resolve_fuel_deposit",
]
