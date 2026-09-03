"""The resolved facts of one explicit fuel-pickup command.

The pickup choreography is the one emission that read MUTABLE world
state while narrating: it looked up the container's remaining volume
and the tank's fuel at the moment it built the wire messages. That is
harmless with one connection and wrong with several, because narration
then depends on when it runs rather than on what happened
([[physics-module-roadmap]]).

This module snapshots those facts once, at resolution, so the narrator
is a pure function of the snapshot. Nothing here mutates the world —
the transfer itself is applied by
:func:`tankpit_bot.sim.movement.process_move`, whose arrival pickup
does the work.

No codec: the snapshot never leaves the process. It is built inside
one command's routing and consumed by the narrator in the same call,
so an encode/decode pair here would have had exactly one caller — its
own round-trip test. The codec rule binds at serialization boundaries,
where the guard chain already makes an unvalidated crossing
unwritable ([[coding-standards]], resolved 2026-09-02).
"""

from __future__ import annotations

from typing import TypedDict

from tankpit_bot.sim.world import SimWorldDict


class FuelPickupOutcomeDict(TypedDict):
    """Everything the pickup choreography needs, captured post-move.

    Attributes:
        tank_id: The picking tank.
        x: The clicked container tile's X.
        y: The clicked container tile's Y.
        volume_before: The container's volume BEFORE the command, read
            before the walk resolved.
        remaining: The container's volume after the transfer. Zero
            covers both a drained container and a tile whose container
            record is gone.
        walked: Whether the command's walk covered any tiles. An
            own-tile or adjacent click transfers without walking, and
            the measured choreography differs
            ([[fuel-system]]).
        fuel_total: The tank's absolute fuel after the transfer — the
            0x44 payload, captured rather than re-read.
    """

    tank_id: int
    x: int
    y: int
    volume_before: int
    remaining: int
    walked: bool
    fuel_total: int


def resolve_fuel_pickup(
    world: SimWorldDict,
    tank_id: int,
    x: int,
    y: int,
    *,
    volume_before: int,
    walked: bool,
) -> FuelPickupOutcomeDict:
    """Capture the post-move facts of one fuel-pickup command.

    Args:
        world: Simulated world, AFTER the move resolved. Read only.
        tank_id: The picking tank.
        x: The clicked container tile's X.
        y: The clicked container tile's Y.
        volume_before: The container's volume before the command.
        walked: Whether the walk covered any tiles.

    Returns:
        The snapshot the pickup narrator is a pure function of.
    """
    remaining = 0
    for container in world["containers"]:
        if (container["x"], container["y"]) == (x, y):
            remaining = container["volume"]
            break
    return FuelPickupOutcomeDict(
        tank_id=tank_id,
        x=x,
        y=y,
        volume_before=volume_before,
        remaining=remaining,
        walked=walked,
        fuel_total=world["tanks"][tank_id]["fuel"],
    )


__all__ = [
    "FuelPickupOutcomeDict",
    "resolve_fuel_pickup",
]
