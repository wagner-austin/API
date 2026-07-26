"""Laws 2 and 2b — instant server movement and ferry surface routing
(wiki [[walk-mechanics]], [[ferry-mechanics]]).

One move command resolves entirely at its processing tick: the router
finds the path, the tank relocates, the per-tile cost is billed, and
destination-tile effects (container pickup, enemy-mine detonation)
happen in the same tick. There is no walking over time — the
on-screen walk is client animation.

Ferry law (user contract 2026-07-19, single-command routing): one
command NEVER chains surfaces. On land, water is unreachable
(cant_go) but a ferry tile boards; while riding, water is open sea
and the ferry moves with the tank. The FIRST queue-consuming surface
transition — stepping onto a ferry (boarding) or from water/ferry
onto land (disembarking) — STOPS the move at that tile; the billed
cost and the echoed path cover only the tiles actually walked.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.physics.costs import WALK_COST_PER_TILE
from tankpit_bot.physics.damage import SINGLE_HIT_VICTIM_COST
from tankpit_bot.sim.blocks import BLOCK_BRIDGE, block_tile_value
from tankpit_bot.sim.pathfind import route
from tankpit_bot.sim.world import SimFerryDict, SimWorldDict

# Walking into an enemy mine costs the mine's 45 (wiki [[game-economy]],
# same magnitude as a single hit).
MINE_WALK_COST = SINGLE_HIT_VICTIM_COST

_STEP_DELTAS: dict[str, tuple[int, int]] = {
    "n": (0, -1),
    "s": (0, 1),
    "e": (1, 0),
    "w": (-1, 0),
}

Surface = Literal["land", "water", "ferry"]


def ferry_at(world: SimWorldDict, x: int, y: int) -> SimFerryDict | None:
    """Return the ferry occupying a tile, if any.

    Args:
        world: Simulated world.
        x: Tile X.
        y: Tile Y.

    Returns:
        The ferry at (x, y), or None.
    """
    for ferry in world["ferries"]:
        if (ferry["x"], ferry["y"]) == (x, y):
            return ferry
    return None


def tile_surface(
    world: SimWorldDict, terrain: TerrainMapProtocol, x: int, y: int
) -> Surface | None:
    """Classify one tile's movement surface.

    Args:
        world: Simulated world (ferries override the static map).
        terrain: Static terrain of the world's field.
        x: Tile X.
        y: Tile Y.

    Returns:
        ``ferry`` for a live ferry tile, ``land`` for passable static
        ground (including a walkable water-block bridge —
        [[movable-blocks]]: walking on a bridge is ordinary movement),
        ``water`` for static water, and None for rock, land-block
        obstacles, and stacked blocks (impassable on every surface).
    """
    if ferry_at(world, x, y) is not None:
        return "ferry"
    block_value = block_tile_value(world, terrain, x, y)
    if block_value != 0:
        return "land" if block_value == BLOCK_BRIDGE else None
    if terrain.is_passable(x, y):
        return "land"
    if terrain.get_terrain(x, y) == terrain.WATER:
        return "water"
    return None


class PickupRecordDict(TypedDict):
    """One container drained by an arrival: tile and remaining volume."""

    x: int
    y: int
    remaining_volume: int


class MoveOutcomeDict(TypedDict):
    """Everything one processed move command changed.

    ``kind`` is ``moved`` on success; ``cant_go`` when no route exists
    (impassable destination or fully blocked paths). Fuel never
    rejects a walk: the debit clamps to remaining fuel, and fuel-0
    walks execute in full (density runs 2-3, 2026-07-25 — repeated
    accepted multi-tile walks at fuel 0; wiki [[game-economy]]:
    "walking and scanning are both free at fuel 0").
    ``mine_positions`` lists enemy mines detonated by the arrival
    (at most one — the destination tile).
    """

    kind: Literal["moved", "cant_go"]
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


def _truncate_at_transition(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    start_x: int,
    start_y: int,
    start_surface: Surface,
    path: str,
) -> tuple[str, int, int]:
    """Cut a routed path at the first queue-consuming surface transition.

    Boarding a ferry from land and stepping from water/ferry onto land
    each consume the whole action: the server stops the tank ON the
    transition tile and a fresh command continues from there.

    Args:
        world: Simulated world.
        terrain: Static terrain of the world's field.
        start_x: Route start X.
        start_y: Route start Y.
        start_surface: Surface the tank stands on at route start.
        path: The routed step string.

    Returns:
        ``(walked_path, final_x, final_y)`` — the possibly shortened
        step string and the tile the tank actually stops on.
    """
    x, y = start_x, start_y
    previous = start_surface
    walked = 0
    for step in path:
        dx, dy = _STEP_DELTAS[step]
        x, y = x + dx, y + dy
        walked += 1
        surface = tile_surface(world, terrain, x, y)
        if previous == "land" and surface == "ferry":
            break
        if previous in ("water", "ferry") and surface == "land":
            break
        if surface is not None:
            previous = surface
    return path[:walked], x, y


def process_move(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    tank_id: int,
    dest_x: int,
    dest_y: int,
) -> MoveOutcomeDict:
    """Process one destination-click move command at the current tick.

    Routing is single-surface (the ferry law): on land the router
    opens ground and ferry tiles only — a water click is cant_go;
    while riding it opens water, ferry, and land tiles, with the
    disembark stop truncating at the first land step. The ferry
    carries its rider: while the rider stays afloat the ferry ends
    the move under the tank; a disembark leaves it on the last water
    tile.

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
    riding = ferry_at(world, start_x, start_y)
    start_surface: Surface = "ferry" if riding is not None else "land"
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
        """Surface-gated routing; the destination tile may hold a mine."""
        surface = tile_surface(world, terrain, x, y)
        if surface is None:
            return False
        if riding is None and surface == "water":
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
    walked, final_x, final_y = _truncate_at_transition(
        world, terrain, start_x, start_y, start_surface, path
    )
    # The debit clamps to remaining fuel (radar-analog law): the walk
    # itself never rejects for fuel — fuel-0 walks executed live.
    cost = min(WALK_COST_PER_TILE * len(walked), tank["fuel"])
    tank["x"] = final_x
    tank["y"] = final_y
    tank["fuel"] -= cost
    outcome["path"] = walked
    if riding is not None:
        if tile_surface(world, terrain, final_x, final_y) == "land":
            last_x, last_y = start_x, start_y
            for step in walked[:-1]:
                dx, dy = _STEP_DELTAS[step]
                last_x, last_y = last_x + dx, last_y + dy
            riding["x"], riding["y"] = last_x, last_y
        else:
            riding["x"], riding["y"] = final_x, final_y
    _resolve_arrival(world, tank_id, outcome)
    return outcome


__all__ = [
    "MINE_WALK_COST",
    "MoveOutcomeDict",
    "PickupRecordDict",
    "Surface",
    "ferry_at",
    "process_move",
    "resolve_pickup",
    "tile_surface",
]
