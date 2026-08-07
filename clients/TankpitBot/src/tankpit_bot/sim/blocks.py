"""Law 6b — movable concrete blocks (wiki [[movable-blocks]]).

Wire-cracked 2026-07-20: ONE client command ('b', 98) serves pickup
and drop — the server decides from carry state. The wire value enum
is shared by 0x42 / 0x4A / 0x5A: one block over static water is a
walkable bridge (1), a block on land is an obstacle (2), two blocks
on a water tile are stacked terrain (3), 0 clears. Block operations
are FREE; placing on a mined land tile destroys ANY team's mine
wire-silently; containers under blocks survive; teleport while
towing is refused with 0x52 code 0.

Sim assumptions (documented): pickup and drop both require CARDINAL
adjacency (the measured out-of-reach press drew code 1; the exact
reach is unmeasured); stacking beyond two and drops onto rock or
occupied land-block tiles refuse with code 1; the transient towed
0x4A pairs along the walk path are not modeled.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.sim.world import SimBlockDict, SimWorldDict
from tankpit_bot.state.scan_coverage import tile_key

BLOCK_BRIDGE = 1
BLOCK_LAND = 2
BLOCK_STACKED = 3
BLOCK_CLEARED = 0

_PICKUP_DIRECTIONS: dict[tuple[int, int], str] = {
    (1, 0): "e",
    (-1, 0): "w",
    (0, 1): "s",
    (0, -1): "n",
}


class BlockOutcomeDict(TypedDict):
    """One resolved block press.

    ``direction`` is the ASCII compass code on a pickup (which side
    of the tank the block attached from) and 0 on a drop.
    ``tile_value`` is the target tile's wire value AFTER the action
    (the 0x42 ``obstacle_type`` and the 0x4A state).
    """

    kind: Literal["picked_up", "dropped", "out_of_reach", "refused"]
    x: int
    y: int
    direction: int
    tile_value: int


def blocks_at(world: SimWorldDict, x: int, y: int) -> int:
    """Count resting blocks on a tile.

    Args:
        world: Simulated world.
        x: Tile X.
        y: Tile Y.

    Returns:
        The number of blocks stacked on (x, y) — 0, 1, or 2.
    """
    return sum(1 for block in world["blocks"] if (block["x"], block["y"]) == (x, y))


def block_tile_value(world: SimWorldDict, terrain: TerrainMapProtocol, x: int, y: int) -> int:
    """Derive a tile's wire block value from context.

    Args:
        world: Simulated world.
        terrain: Static terrain of the world's field.
        x: Tile X.
        y: Tile Y.

    Returns:
        The shared-enum value: 0 (no blocks), 1 (bridge), 2 (land
        obstacle), or 3 (stacked).
    """
    count = blocks_at(world, x, y)
    if count == 0:
        return BLOCK_CLEARED
    if terrain.get_terrain(x, y) == terrain.WATER:
        return BLOCK_STACKED if count >= 2 else BLOCK_BRIDGE
    return BLOCK_LAND


def process_block_press(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    tank_id: int,
    target_x: int,
    target_y: int,
) -> BlockOutcomeDict:
    """Resolve one 'b' press: pickup when empty-handed, drop when towing.

    Args:
        world: Simulated world (mutated on success).
        terrain: Static terrain of the world's field.
        tank_id: The pressing tank.
        target_x: Clicked tile X.
        target_y: Clicked tile Y.

    Returns:
        The typed outcome; the world reflects it on success.
    """
    tank = world["tanks"][tank_id]
    offset = (target_x - tank["x"], target_y - tank["y"])
    direction_letter = _PICKUP_DIRECTIONS.get(offset)
    outcome = BlockOutcomeDict(
        kind="out_of_reach", x=target_x, y=target_y, direction=0, tile_value=0
    )
    if direction_letter is None:
        return outcome
    if not tank["carrying"]:
        picked: SimBlockDict | None = None
        for block in world["blocks"]:
            if (block["x"], block["y"]) == (target_x, target_y):
                picked = block
                break
        if picked is None:
            return outcome
        world["blocks"].remove(picked)
        tank["carrying"] = True
        outcome["kind"] = "picked_up"
        outcome["direction"] = ord(direction_letter)
        outcome["tile_value"] = block_tile_value(world, terrain, target_x, target_y)
        return outcome
    ground = terrain.get_terrain(target_x, target_y)
    resting = blocks_at(world, target_x, target_y)
    water = ground == terrain.WATER
    if ground == terrain.ROCK or (water and resting >= 2) or (not water and resting >= 1):
        outcome["kind"] = "refused"
        return outcome
    world["blocks"].append(SimBlockDict(x=target_x, y=target_y))
    tank["carrying"] = False
    if not water:
        world["mines"].pop(tile_key(target_x, target_y), None)
    outcome["kind"] = "dropped"
    outcome["tile_value"] = block_tile_value(world, terrain, target_x, target_y)
    return outcome


__all__ = [
    "BLOCK_BRIDGE",
    "BLOCK_CLEARED",
    "BLOCK_LAND",
    "BLOCK_STACKED",
    "BlockOutcomeDict",
    "block_tile_value",
    "blocks_at",
    "process_block_press",
]
