"""Law 2c — autonomous ferry drift and its 0x4A wire pair."""

from __future__ import annotations

from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.ferries import (
    MAP_SPAN,
    WIRE_TERRAIN_CLEARED,
    WIRE_TERRAIN_FERRY,
    drift_ferries,
)
from tankpit_bot.sim.world import SimFerryDict, SimWorldDict, make_sim_tank, make_sim_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _lake() -> InMemoryTerrainMap:
    """A 9x9 pond of water centred on (100, 100), land everywhere else."""
    return InMemoryTerrainMap(
        terrain_data={(x, y): "W" for x in range(96, 105) for y in range(96, 105)}
    )


def _world_with_ferry(x: int, y: int) -> SimWorldDict:
    """A world holding exactly one ferry and no tanks on it."""
    world = make_sim_world("field01_r.gif")
    world["ferries"].append(SimFerryDict(x=x, y=y))
    return world


def _terrain_updates(messages: list[BinaryMessage]) -> list[list[tuple[int, int, int]]]:
    """The `updates` payload of every 0x4A in a batch."""
    return [m["updates"] for m in messages if m["msg_type"] == 0x4A]


def test_a_drifting_ferry_clears_its_old_tile_and_sets_its_new_one() -> None:
    """The measured wire shape: one frame, two updates, 0 then 5.

    All 205 archived ferry moves carry exactly this pair — the vacated
    tile reverts to the wire's "nothing here" value so the static map
    (water) shows through, and the occupied tile becomes terrain 5
    ([[ferry-mechanics]]).
    """
    world = _world_with_ferry(100, 100)
    messages: list[BinaryMessage] = []

    drift_ferries(world, _lake(), messages)

    updates = _terrain_updates(messages)
    assert len(updates) == 1
    (from_x, from_y, from_value), (to_x, to_y, to_value) = updates[0]
    assert (from_x, from_y, from_value) == (100, 100, WIRE_TERRAIN_CLEARED)
    assert to_value == WIRE_TERRAIN_FERRY
    assert max(abs(to_x - from_x), abs(to_y - from_y)) == 1
    assert (world["ferries"][0]["x"], world["ferries"][0]["y"]) == (to_x, to_y)


def test_drift_is_one_tile_per_tick() -> None:
    """Each tick moves a ferry by exactly one tile, never further.

    The chained archive steps land a median 2003 ms apart — one tick —
    and the modal step is a single axial or diagonal tile.
    """
    world = _world_with_ferry(100, 100)
    terrain = _lake()
    previous = (100, 100)
    for _ in range(6):
        world["tick"] += 1
        messages: list[BinaryMessage] = []
        drift_ferries(world, terrain, messages)
        current = (world["ferries"][0]["x"], world["ferries"][0]["y"])
        assert max(abs(current[0] - previous[0]), abs(current[1] - previous[1])) == 1
        previous = current


def test_a_ridden_ferry_does_not_drift() -> None:
    """The rider drives it; drifting too would double the step.

    ``movement._update_ridden_ferry`` already carried the ferry with
    its rider this tick.
    """
    world = _world_with_ferry(100, 100)
    world["tanks"][9] = make_sim_tank(9, 2, 1, 100, 100, 1100)
    messages: list[BinaryMessage] = []

    drift_ferries(world, _lake(), messages)

    assert messages == []
    assert (world["ferries"][0]["x"], world["ferries"][0]["y"]) == (100, 100)


def test_a_ferry_against_the_shore_idles() -> None:
    """A heading onto non-water is not a move, and emits nothing.

    This is where the long observed gaps come from: the archive's
    terrain-update spacing runs to a p75 of 8005 ms, four ticks.
    """
    world = _world_with_ferry(100, 100)
    landlocked = InMemoryTerrainMap(terrain_data={(100, 100): "W"})
    messages: list[BinaryMessage] = []

    drift_ferries(world, landlocked, messages)

    assert messages == []
    assert (world["ferries"][0]["x"], world["ferries"][0]["y"]) == (100, 100)


def test_a_ferry_never_drifts_onto_another_ferry() -> None:
    """Two ferries do not stack: the blocked one idles this tick."""
    world = _world_with_ferry(100, 100)
    terrain = _lake()
    messages: list[BinaryMessage] = []
    drift_ferries(world, terrain, messages)
    landed = (world["ferries"][0]["x"], world["ferries"][0]["y"])

    crowded = _world_with_ferry(100, 100)
    crowded["ferries"].append(SimFerryDict(x=landed[0], y=landed[1]))
    crowded_messages: list[BinaryMessage] = []
    drift_ferries(crowded, terrain, crowded_messages)

    assert (crowded["ferries"][0]["x"], crowded["ferries"][0]["y"]) == (100, 100)
    assert (100, 100) not in [(u[0][0], u[0][1]) for u in _terrain_updates(crowded_messages)]


def test_a_heading_off_the_map_edge_is_not_a_move() -> None:
    """Tile coordinates stop at 0 and 255; the edge is a shore."""
    world = _world_with_ferry(0, 0)
    edge = InMemoryTerrainMap(terrain_data={(x, y): "W" for x in range(0, 3) for y in range(0, 3)})
    for tick in range(8):  # every heading in the cycle, from the corner
        world["tick"] = tick
        world["ferries"][0]["x"], world["ferries"][0]["y"] = 0, 0
        messages: list[BinaryMessage] = []
        drift_ferries(world, edge, messages)
        assert 0 <= world["ferries"][0]["x"] < MAP_SPAN
        assert 0 <= world["ferries"][0]["y"] < MAP_SPAN
