"""Pure wire-statement builders for the sim server.

Each function builds one decoded ``BinaryMessage`` from world state
alone — no server state, no side effects. The server and its
emission helpers compose these into per-tick batches.
"""

from __future__ import annotations

from tankpit_bot.physics.capacity import damage_tier
from tankpit_bot.protocol.types import (
    MovementDict,
    MovementResponseDict,
    TankInfoDict,
    TankStatusSyncDict,
)
from tankpit_bot.sim.commands import ClientCommandDict
from tankpit_bot.sim.movement import MoveOutcomeDict
from tankpit_bot.sim.world import SimWorldDict


def queued_tank_id(entry: tuple[int, ClientCommandDict]) -> int:
    """The round-order sort key: the queued command's tank id.

    Args:
        entry: One ``(tank_id, command)`` queue entry.

    Returns:
        The tank id — within-round resolution is ascending tank id.
    """
    return entry[0]


def movement_echo(world: SimWorldDict, outcome: MoveOutcomeDict) -> MovementDict:
    """Build the 0x47 echo for one processed move.

    Args:
        world: Simulated world (post-move).
        outcome: The move's outcome.

    Returns:
        The movement echo carrying the full routed path.
    """
    tank = world["tanks"][outcome["tank_id"]]
    path = outcome["path"]
    x, y = tank["x"], tank["y"]
    return MovementDict(
        msg_type=0x47,
        tank_id=tank["tank_id"],
        start_x=outcome["start_x"],
        start_y=outcome["start_y"],
        direction=0,
        damage_state=damage_tier(tank["fuel"], tank["rank"]),
        lb_score=0,
        rank=tank["rank"],
        flag=1,
        is_carrying=False,
        waypoints=[(x, y)] if path else [],
        path_tiles=len(path),
        path=path,
    )


def status_sync(tank_id: int, world: SimWorldDict, include_fuel: bool) -> TankStatusSyncDict:
    """Build a 0x2E status sync for one tank.

    The real wire carries the fuel field ONLY on the recipient's own
    tank (per-recipient long form); other tanks sync short-form. The
    production dispatcher treats any fuel-bearing 0x2E as self fuel,
    so emitting long-form for a victim would corrupt the client's own
    belief — caught by the step-(c) wire integration.

    Args:
        tank_id: The synced tank.
        world: Simulated world.
        include_fuel: True only for the connected client's tank.

    Returns:
        The status sync (long form with fuel, or short form).
    """
    tank = world["tanks"][tank_id]
    return TankStatusSyncDict(
        msg_type=0x2E,
        subtype=tank["team"],
        tank_id=tank_id,
        damage_state=damage_tier(tank["fuel"], tank["rank"]),
        rank=tank["rank"],
        lb_score=0,
        promo_state=0,
        fuel=tank["fuel"] if include_fuel else None,
    )


def identity_statement(world: SimWorldDict, tank_id: int) -> TankInfoDict:
    """Build the 0x21 identity broadcast for one tank.

    Args:
        world: Simulated world.
        tank_id: The announced tank.

    Returns:
        The identity message.
    """
    tank = world["tanks"][tank_id]
    return TankInfoDict(
        msg_type=0x21,
        tank_id=tank_id,
        team=tank["team"],
        decoration_state=bytes(4),
        persistent_tank_id=0,
        # The tank's seeded wire name. The default practice shape
        # (``red-<id>``, set by ``make_sim_tank``) keeps sim opponents
        # farmable; a human-shaped seeding puts the tank behind the
        # human-consent combat gate (2026-07-30) so sim sessions can
        # exercise the human-fight contracts (2026-07-31).
        name=tank["name"],
    )


def position_statement(world: SimWorldDict, tank_id: int) -> MovementResponseDict:
    """Build a 0x3D position statement for one tank.

    Args:
        world: Simulated world.
        tank_id: The positioned tank.

    Returns:
        The movement response carrying the tank's current tile.
    """
    tank = world["tanks"][tank_id]
    return MovementResponseDict(
        msg_type=0x3D,
        team=tank["team"],
        tank_id=tank_id,
        x=tank["x"],
        y=tank["y"],
        direction=0,
        damage_state=damage_tier(tank["fuel"], tank["rank"]),
        rank=tank["rank"],
        lb_score=0,
        carrying=0,
    )


__all__ = [
    "identity_statement",
    "movement_echo",
    "position_statement",
    "queued_tank_id",
    "status_sync",
]
