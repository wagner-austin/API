"""Law 9 — room churn: visitors arriving and leaving mid-session."""

from __future__ import annotations

from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.visitors import (
    VISITOR_ARRIVAL_PERIOD_TICKS,
    VISITOR_ENTRY_X,
    VISITOR_ENTRY_Y,
    VISITOR_ID_BASE,
    VISITOR_STAY_TICKS,
    RoomChurn,
)
from tankpit_bot.sim.world import SimWorldDict, make_sim_tank, make_sim_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _world() -> SimWorldDict:
    """A world holding only the client tank."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 2, 1, 100, 100, 1100)
    return world


def _run_to(churn: RoomChurn, world: SimWorldDict, tick: int) -> list[BinaryMessage]:
    """Advance the churn clock to a tick, collecting everything emitted."""
    terrain = InMemoryTerrainMap()
    emitted: list[BinaryMessage] = []
    while world["tick"] < tick:
        world["tick"] += 1
        churn.advance(world, terrain, emitted)
    return emitted


def test_a_visitor_arrives_on_the_measured_period() -> None:
    """Sixteen entries over 285 sessions is one per ~1,700 ticks.

    Churn is rare on the real wire and rare here; a soak sees one when
    it runs long enough to deserve one
    ([[session-state-deglobalisation]]).
    """
    world = _world()
    churn = RoomChurn()

    quiet = _run_to(churn, world, VISITOR_ARRIVAL_PERIOD_TICKS - 1)
    assert quiet == []

    arrival = _run_to(churn, world, VISITOR_ARRIVAL_PERIOD_TICKS)
    assert [message["msg_type"] for message in arrival] == [0x28]
    assert churn.visitor_id == VISITOR_ID_BASE


def test_the_entry_reports_no_position() -> None:
    """15 of 16 archived entries carry (0, 0): the tank is not in view."""
    world = _world()
    churn = RoomChurn()

    arrival = _run_to(churn, world, VISITOR_ARRIVAL_PERIOD_TICKS)

    entry = arrival[0]
    assert (entry["x"], entry["y"]) == (VISITOR_ENTRY_X, VISITOR_ENTRY_Y)
    assert entry["tank_id"] == VISITOR_ID_BASE
    assert entry["damage_state"] == 3
    assert entry["rank"] in (0, 1)


def test_the_visitor_really_joins_the_world() -> None:
    """An announced arrival is a tank the room can actually see."""
    world = _world()
    churn = RoomChurn()

    _run_to(churn, world, VISITOR_ARRIVAL_PERIOD_TICKS)

    visitor = world["tanks"][VISITOR_ID_BASE]
    assert visitor["alive"] is True
    assert (visitor["x"], visitor["y"]) != (VISITOR_ENTRY_X, VISITOR_ENTRY_Y)


def test_the_visitor_leaves_after_the_measured_stay() -> None:
    """Median visit is 59 ticks, and the exit is a plain departure.

    Every one of the ten paired archived exits is ``was_silent=False,
    was_eliminated=False``: the player left, and the client is told.
    """
    world = _world()
    churn = RoomChurn()
    _run_to(churn, world, VISITOR_ARRIVAL_PERIOD_TICKS)

    departure = _run_to(churn, world, VISITOR_ARRIVAL_PERIOD_TICKS + VISITOR_STAY_TICKS)

    assert [message["msg_type"] for message in departure] == [0x29]
    assert departure[0] == {
        "msg_type": 0x29,
        # The first entry of the archive's four-colour cycle.
        "team": 0,
        "tank_id": VISITOR_ID_BASE,
        "was_silent": False,
        "was_eliminated": False,
    }
    assert churn.visitor_id is None
    assert VISITOR_ID_BASE not in world["tanks"]


def test_only_one_visitor_at_a_time() -> None:
    """A second arrival waits for the first to leave."""
    world = _world()
    churn = RoomChurn()
    _run_to(churn, world, VISITOR_ARRIVAL_PERIOD_TICKS)

    # Past the next arrival period, but the first visitor has not left.
    world["tick"] = VISITOR_ARRIVAL_PERIOD_TICKS * 2 - 1
    churn.arrived_tick = world["tick"]
    extra = _run_to(churn, world, VISITOR_ARRIVAL_PERIOD_TICKS * 2)

    assert extra == []
    assert churn.visitor_id == VISITOR_ID_BASE


def test_successive_visitors_take_fresh_ids() -> None:
    """A second visitor is a different tank, not the first returning."""
    world = _world()
    churn = RoomChurn()
    _run_to(churn, world, VISITOR_ARRIVAL_PERIOD_TICKS)
    _run_to(churn, world, VISITOR_ARRIVAL_PERIOD_TICKS + VISITOR_STAY_TICKS)

    second = _run_to(churn, world, VISITOR_ARRIVAL_PERIOD_TICKS * 2)

    assert [message["tank_id"] for message in second] == [VISITOR_ID_BASE + 1]


def test_a_sealed_map_admits_nobody() -> None:
    """No open tile, no arrival — and no half-announced visitor."""
    world = _world()
    churn = RoomChurn()
    sealed = InMemoryTerrainMap(terrain_data={(x, y): "#" for x in range(256) for y in range(256)})
    world["tick"] = VISITOR_ARRIVAL_PERIOD_TICKS
    emitted: list[BinaryMessage] = []

    churn.advance(world, sealed, emitted)

    assert emitted == []
    assert churn.visitor_id is None


def test_the_server_churns_its_room() -> None:
    """The tick processor drives the churn, not just the unit."""
    from tankpit_bot.sim.server import SimServer

    world = _world()
    server = SimServer(world, InMemoryTerrainMap(), client_id=9)
    world["tick"] = VISITOR_ARRIVAL_PERIOD_TICKS - 1

    batch = server.advance_tick()

    assert 0x28 in [message["msg_type"] for message in batch]
