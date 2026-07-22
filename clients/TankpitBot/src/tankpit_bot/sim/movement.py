"""Law 2 — instant server movement (wiki [[walk-mechanics]]).

One move command resolves entirely at its processing tick: the router
finds the path, the tank relocates to the destination, the full
per-tile cost is billed, and destination-tile effects (container
pickup, enemy-mine detonation) happen in the same tick. There is no
walking over time — the on-screen walk is client animation.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.physics.costs import WALK_COST_PER_TILE
from tankpit_bot.physics.damage import SINGLE_HIT_VICTIM_COST
from tankpit_bot.sim.pathfind import route
from tankpit_bot.sim.world import SimWorldDict

# Walking into an enemy mine costs the mine's 45 (wiki [[game-economy]],
# same magnitude as a single hit).
MINE_WALK_COST = SINGLE_HIT_VICTIM_COST


class PickupRecordDict(TypedDict):
    """One container drained by an arrival: tile and remaining volume."""

    x: int
    y: int
    remaining_volume: int


class MoveOutcomeDict(TypedDict):
    """Everything one processed move command changed.

    ``kind`` is ``moved`` on success; ``cant_go`` when no route exists
    (impassable destination or fully blocked paths);
    ``insufficient_fuel`` when the routed cost exceeds the tank's
    fuel. ``mine_positions`` lists enemy mines detonated by the
    arrival (at most one — the destination tile).
    """

    kind: Literal["moved", "cant_go", "insufficient_fuel"]
    tank_id: int
    start_x: int
    start_y: int
    path: str
    pickups: list[PickupRecordDict]
    mine_positions: list[tuple[int, int]]


def _blocked_by_world(world: SimWorldDict, mover_id: int, team: int, x: int, y: int) -> bool:
    """Report whether world entities block the mover from a tile.

    Args:
        world: Simulated world.
        mover_id: The moving tank's id (it never blocks itself).
        team: The mover's team — own-color mines are walkable
            (user contract 2026-07-21), enemy mines are routed around.
        x: Tile X.
        y: Tile Y.

    Returns:
        True when another living tank or an enemy mine occupies the tile.
    """
    for tank in world["tanks"].values():
        if tank["alive"] and tank["tank_id"] != mover_id and (tank["x"], tank["y"]) == (x, y):
            return True
    return any(mine["team"] != team and (mine["x"], mine["y"]) == (x, y) for mine in world["mines"])


def resolve_pickup(world: SimWorldDict, tank_id: int, pickups: list[PickupRecordDict]) -> None:
    """Drain the container under a tank into it, capacity-clamped.

    Shared by walk arrivals and teleport landings (both auto-pick-up
    on the destination tile).

    Args:
        world: Simulated world (mutated).
        tank_id: The arriving tank.
        pickups: Pickup records accumulator (appended).
    """
    tank = world["tanks"][tank_id]
    x, y = tank["x"], tank["y"]
    capacity = fuel_capacity(tank["rank"])
    for container in world["containers"]:
        if (container["x"], container["y"]) == (x, y) and container["volume"] > 0:
            transfer = min(container["volume"], capacity - tank["fuel"])
            container["volume"] -= transfer
            tank["fuel"] += transfer
            pickups.append(PickupRecordDict(x=x, y=y, remaining_volume=container["volume"]))


def _resolve_arrival(world: SimWorldDict, tank_id: int, outcome: MoveOutcomeDict) -> None:
    """Apply destination-tile effects: mine detonation and pickup.

    Args:
        world: Simulated world (mutated).
        tank_id: The arriving tank.
        outcome: Outcome being built (mutated).
    """
    tank = world["tanks"][tank_id]
    x, y = tank["x"], tank["y"]
    for mine in list(world["mines"]):
        if mine["team"] != tank["team"] and (mine["x"], mine["y"]) == (x, y):
            world["mines"].remove(mine)
            tank["fuel"] = max(0, tank["fuel"] - MINE_WALK_COST)
            outcome["mine_positions"].append((x, y))
    resolve_pickup(world, tank_id, outcome["pickups"])


def process_move(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    tank_id: int,
    dest_x: int,
    dest_y: int,
) -> MoveOutcomeDict:
    """Process one destination-click move command at the current tick.

    Args:
        world: Simulated world (mutated on success).
        terrain: Static terrain of the world's field.
        tank_id: The commanding tank (must exist and be alive).
        dest_x: Clicked destination X.
        dest_y: Clicked destination Y.

    Returns:
        The typed outcome; the world reflects it on ``moved``.
    """
    tank = world["tanks"][tank_id]
    start_x, start_y = tank["x"], tank["y"]
    outcome = MoveOutcomeDict(
        kind="moved",
        tank_id=tank_id,
        start_x=start_x,
        start_y=start_y,
        path="",
        pickups=[],
        mine_positions=[],
    )

    def passable(x: int, y: int) -> bool:
        """Interior tiles avoid entities; the destination may hold a mine."""
        if not terrain.is_passable(x, y):
            return False
        if (x, y) == (dest_x, dest_y):
            return not any(
                other["alive"]
                and other["tank_id"] != tank_id
                and (other["x"], other["y"]) == (x, y)
                for other in world["tanks"].values()
            )
        return not _blocked_by_world(world, tank_id, tank["team"], x, y)

    path = route(passable, start_x, start_y, dest_x, dest_y)
    if path is None:
        outcome["kind"] = "cant_go"
        return outcome
    cost = WALK_COST_PER_TILE * len(path)
    if cost > tank["fuel"]:
        outcome["kind"] = "insufficient_fuel"
        return outcome
    tank["x"] = dest_x
    tank["y"] = dest_y
    tank["fuel"] -= cost
    outcome["path"] = path
    _resolve_arrival(world, tank_id, outcome)
    return outcome


__all__ = [
    "MINE_WALK_COST",
    "MoveOutcomeDict",
    "PickupRecordDict",
    "process_move",
    "resolve_pickup",
]
