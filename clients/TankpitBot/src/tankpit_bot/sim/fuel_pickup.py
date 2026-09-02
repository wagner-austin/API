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
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    require_bool,
    require_int,
)

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


def encode_fuel_pickup_outcome(outcome: FuelPickupOutcomeDict) -> JSONObject:
    """Encode one pickup snapshot to a JSON-serializable dict.

    Args:
        outcome: The snapshot to encode.

    Returns:
        JSON object with every snapshot field.
    """
    return {
        "tank_id": outcome["tank_id"],
        "x": outcome["x"],
        "y": outcome["y"],
        "volume_before": outcome["volume_before"],
        "remaining": outcome["remaining"],
        "walked": outcome["walked"],
        "fuel_total": outcome["fuel_total"],
    }


def decode_fuel_pickup_outcome(data: JSONObject) -> FuelPickupOutcomeDict:
    """Decode one pickup snapshot from JSON with validation.

    Args:
        data: JSON object carrying the snapshot fields.

    Returns:
        Validated snapshot.

    Raises:
        JSONTypeError: If a field has the wrong type.
        KeyError: If a field is missing.
    """
    return FuelPickupOutcomeDict(
        tank_id=require_int(data, "tank_id"),
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        volume_before=require_int(data, "volume_before"),
        remaining=require_int(data, "remaining"),
        walked=require_bool(data, "walked"),
        fuel_total=require_int(data, "fuel_total"),
    )


__all__ = [
    "FuelPickupOutcomeDict",
    "decode_fuel_pickup_outcome",
    "encode_fuel_pickup_outcome",
    "resolve_fuel_pickup",
]
