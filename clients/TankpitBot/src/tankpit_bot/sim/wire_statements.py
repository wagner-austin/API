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
    TankStatusDict,
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
        # The promotion bar rides with the fuel field or not at all.
        # Lit is the overwhelming wire majority (70,313 of 70,532
        # long-form bodies); the sim never darkens it because nothing
        # is known about what darkens it on the real server.
        promo_bar_lit=True if include_fuel else None,
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


def full_status_statement(world: SimWorldDict, tank_id: int) -> TankStatusDict:
    """Build the 0x3E full status the join burst carries for own tank.

    Every one of the 285 archived real sessions opens with the same
    five received frames and then ``0x21 identity, 0x3E full status,
    0x5A viewport, 0x3D position, 0x2E sync`` — the 0x3E always names
    the player's own tank. The sim's handshake had no 0x3E at all, so
    286 archived frames of this family had no sim counterpart
    ([[session-state-deglobalisation]]).

    Leaderboard score, leaderboard position and decorations are zero
    for the same reason ``identity_statement`` zeroes them: the sim
    keeps no cross-session standings, and inventing a rank would be a
    number the bot could read and believe.

    Args:
        world: Simulated world.
        tank_id: The described tank.

    Returns:
        The full status message.
    """
    tank = world["tanks"][tank_id]
    return TankStatusDict(
        msg_type=0x3E,
        team=tank["team"],
        rank=tank["rank"],
        damage_state=damage_tier(tank["fuel"], tank["rank"]),
        tank_id=tank_id,
        decoration_state=bytes(4),
        leaderboard_score=0,
        leaderboard_position=0,
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
    "full_status_statement",
    "identity_statement",
    "movement_echo",
    "position_statement",
    "queued_tank_id",
    "status_sync",
]
