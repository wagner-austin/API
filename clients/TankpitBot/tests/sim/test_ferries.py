"""Ferries are furniture until a rider walks them.

Law, operator ruling 2026-09-02: "ferries only move when someone is
riding them and moving them. they do not move if they are unattended.
they do not move if you stand on them. someone may use them when you
arent looking i suppose, and move them, but that requires a person."

Standing on one is not enough — the rider has to WALK. That matches
what the archive already said in [[ferry-mechanics]] since 2026-08-04
("No ferry drift law exists in the archive": 148 moves across 312
captures, 136 rider-attributed, the 12 residuals under-observed
riders) and what live emits: 0.91 0x4A per session across 341
sessions.

The sim carried an autonomous drift law from 2026-08-06 until
2026-09-02, moving all 33 seeded ferries every tick and emitting
2,336-4,460 0x4A per session — roughly 3,600x live. It is deleted;
carrying a ferry with its rider lives in ``movement`` and is
untouched. The first test below is the pin: it fails the moment
anything moves an unattended ferry again.
"""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.resources import data_directory
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.world import SimFerryDict, SimWorldDict, make_sim_tank, make_sim_world
from tankpit_bot.sim.world_seed_mines import seed_ferries
from tests.in_memory_terrain_map import InMemoryTerrainMap

_FIELD = "field01_r.gif"


def _lake() -> InMemoryTerrainMap:
    """A 9x9 pond of water centred on (100, 100), land everywhere else."""
    return InMemoryTerrainMap(
        terrain_data={(x, y): "W" for x in range(96, 105) for y in range(96, 105)}
    )


def _terrain_updates(messages: list[BinaryMessage]) -> list[list[tuple[int, int, int]]]:
    """The ``updates`` payload of every 0x4A in a batch."""
    return [m["updates"] for m in messages if m["msg_type"] == 0x4A]


def _real_terrain() -> TerrainMapProtocol:
    """Load the committed field01 terrain the seeder floats ferries on."""
    return _test_hooks.load_terrain_map(data_directory() / _FIELD)


def _world_with_ferry(x: int, y: int) -> SimWorldDict:
    """A world holding exactly one ferry, with the client well clear."""
    world = make_sim_world(_FIELD)
    world["tanks"][9] = make_sim_tank(9, 0, 1, 90, 90, 1000)
    world["ferries"].append(SimFerryDict(x=x, y=y))
    return world


def test_an_unattended_ferry_never_moves_and_never_says_anything() -> None:
    """THE PIN. An idle ferry emits no 0x4A, tick after tick.

    Deleting the autonomous drift law is only durable if re-adding it
    breaks something, so this asserts the operator's ruling directly:
    with nobody aboard, the ferry is furniture. Thirty ticks is far
    past the old law's one-move-per-tick cadence, which would have
    produced thirty 0x4A pairs here.
    """
    world = _world_with_ferry(100, 100)
    server = SimServer(world, _lake(), client_id=9)

    emitted: list[BinaryMessage] = []
    for _ in range(30):
        emitted.extend(server.advance_tick())

    assert _terrain_updates(emitted) == []
    assert (world["ferries"][0]["x"], world["ferries"][0]["y"]) == (100, 100)


def test_a_ferry_under_a_standing_tank_still_does_not_move() -> None:
    """Standing on one is not riding it — the rider must WALK.

    The operator's ruling is explicit that occupancy alone moves
    nothing, and the deleted drift law had it backwards: it treated an
    occupied tile as "ridden, skip" and drifted every OTHER ferry,
    so a boarded ferry was the only still one on the map.
    """
    world = _world_with_ferry(100, 100)
    world["tanks"][9]["x"] = 100
    world["tanks"][9]["y"] = 100
    server = SimServer(world, _lake(), client_id=9)

    emitted: list[BinaryMessage] = []
    for _ in range(10):
        emitted.extend(server.advance_tick())

    assert _terrain_updates(emitted) == []
    assert (world["ferries"][0]["x"], world["ferries"][0]["y"]) == (100, 100)


def test_the_room_floats_ferries_on_its_water() -> None:
    """Every scenario gets ferries, not just the one that named a tile.

    Seeding is unaffected by the drift deletion: a room still has its
    ferries as terrain the bot can board, they simply do not wander.
    """
    terrain = _real_terrain()
    world = make_sim_world(_FIELD)

    afloat = seed_ferries(world, terrain)

    assert afloat == len(world["ferries"])
    assert afloat > 0
    for ferry in world["ferries"]:
        assert terrain.get_terrain(ferry["x"], ferry["y"]) == terrain.WATER


def test_seeded_ferries_never_stack() -> None:
    """One ferry per tile, so a rider always boards exactly one."""
    world = make_sim_world(_FIELD)
    seed_ferries(world, _real_terrain())

    tiles = [(ferry["x"], ferry["y"]) for ferry in world["ferries"]]
    assert len(set(tiles)) == len(tiles)


def test_a_scenarios_own_ferry_survives_the_seeding() -> None:
    """A world that placed its own keeps it — the ferry scenario does."""
    world = make_sim_world(_FIELD)
    world["ferries"].append(SimFerryDict(x=118, y=112))

    seed_ferries(world, _real_terrain())

    assert (118, 112) in [(ferry["x"], ferry["y"]) for ferry in world["ferries"]]
