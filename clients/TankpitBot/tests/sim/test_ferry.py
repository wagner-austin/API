"""Law 2b — ferry surface routing (wiki [[ferry-mechanics]]).

The user contract (2026-07-19): one command never chains surfaces.
On land, water is unreachable but a ferry tile boards; riding opens
the water and the ferry moves with the tank; the first
queue-consuming transition (boarding, or stepping onto land) stops
the move on the transition tile.
"""

from __future__ import annotations

from tankpit_bot.bot.tick_body import _tick_once
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.movement import (
    _execute_walk,
    ferry_at,
    process_move,
    tile_surface,
)
from tankpit_bot.sim.world import (
    SimContainerDict,
    SimFerryDict,
    SimWorldDict,
    make_sim_tank,
    make_sim_world,
)
from tankpit_bot.sniffer.world_state import get_world_service
from tests.in_memory_terrain_map import InMemoryTerrainMap
from tests.sim.seam import boot_seam

_ROCK = "#"
_WATER = "W"


def _channel_map() -> InMemoryTerrainMap:
    """Land for x <= 12, a water channel for 13 <= x <= 20, land beyond."""
    water = {(x, y): _WATER for x in range(13, 21) for y in range(0, 40)}
    return InMemoryTerrainMap(terrain_data=water)


def _world(tank_x: int, tank_y: int) -> SimWorldDict:
    """Tank 9 at the given tile with a ferry docked at (13, 10)."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, tank_x, tank_y, 1000)
    world["ferries"].append(SimFerryDict(x=13, y=10))
    return world


def test_tile_surfaces_classify_ferry_water_land_and_rock() -> None:
    """The surface classifier reads ferries over the static map."""
    world = _world(10, 10)
    terrain = InMemoryTerrainMap(terrain_data={(5, 5): _WATER, (6, 6): _ROCK})
    assert tile_surface(world, terrain, 13, 10) == "ferry"
    assert tile_surface(world, terrain, 5, 5) == "water"
    assert tile_surface(world, terrain, 6, 6) is None
    assert tile_surface(world, terrain, 2, 2) == "land"


def test_water_click_from_land_is_cant_go() -> None:
    """The server never auto-routes to a ferry — open water refuses."""
    world = _world(10, 10)
    outcome = process_move(world, _channel_map(), 9, 16, 10)
    assert outcome["kind"] == "cant_go"
    assert (world["tanks"][9]["x"], world["tanks"][9]["y"]) == (10, 10)


def test_boarding_stops_on_the_ferry_tile() -> None:
    """A click past the docked ferry still stops ON the ferry (boarding
    consumes the whole action)."""
    world = _world(10, 10)
    outcome = process_move(world, _channel_map(), 9, 13, 10)
    assert outcome["kind"] == "moved"
    assert (world["tanks"][9]["x"], world["tanks"][9]["y"]) == (13, 10)
    assert world["tanks"][9]["fuel"] == 1000 - len(outcome["path"])
    assert tile_surface(world, _channel_map(), 13, 10) == "ferry"


def test_riding_opens_the_water_and_the_ferry_follows() -> None:
    """From the ferry, a water click sails there and carries the boat."""
    world = _world(13, 10)
    outcome = process_move(world, _channel_map(), 9, 18, 14)
    assert outcome["kind"] == "moved"
    assert (world["tanks"][9]["x"], world["tanks"][9]["y"]) == (18, 14)
    assert world["ferries"] == [SimFerryDict(x=18, y=14)]
    assert ferry_at(world, 13, 10) is None


def test_disembark_stops_one_step_onto_land() -> None:
    """A land click from the water stops at the FIRST land tile; the
    ferry stays on the last water tile."""
    world = _world(13, 10)
    world["ferries"][0] = SimFerryDict(x=13, y=10)
    outcome = process_move(world, _channel_map(), 9, 25, 10)
    assert outcome["kind"] == "moved"
    assert (world["tanks"][9]["x"], world["tanks"][9]["y"]) == (21, 10)
    ferry = world["ferries"][0]
    assert (ferry["x"], ferry["y"]) == (20, 10)
    assert len(outcome["path"]) == 8


def test_overshoot_disembark_closes_with_cant_go() -> None:
    """A transition stop SHORT of the click gets the code-1 close.

    Live 2026-08-03 cluster A: the bot rode the ferry afloat on
    (59,28) water and collected at land targets inland — every
    command echoed the one-step disembark and then the 0x52. The
    receipt says "your command did not finish", not "your walk was
    refused".
    """
    from tankpit_bot.protocol.constants import SUPERVISOR_ERROR_CANT_GO
    from tankpit_bot.sim.emissions import emit_move

    world = _world(13, 10)
    outcome = process_move(world, _channel_map(), 9, 25, 10)
    assert outcome["kind"] == "moved"
    assert outcome["stop_reason"] == "transition"
    assert outcome["dest_reached"] is False
    messages: list[BinaryMessage] = []
    emit_move(world, 9, outcome, messages)
    assert messages[0]["msg_type"] == 0x47
    closes = [m for m in messages if m["msg_type"] == 0x52]
    assert [m["error_code"] for m in closes] == [SUPERVISOR_ERROR_CANT_GO]


def test_transition_stop_on_the_clicked_tile_closes_silently() -> None:
    """Boarding the ferry tile the click named is a FINISHED command."""
    from tankpit_bot.sim.emissions import emit_move

    world = _world(10, 10)
    outcome = process_move(world, _channel_map(), 9, 13, 10)
    assert outcome["stop_reason"] == "transition"
    assert outcome["dest_reached"] is True
    messages: list[BinaryMessage] = []
    emit_move(world, 9, outcome, messages)
    assert messages[0]["msg_type"] == 0x47
    assert [m for m in messages if m["msg_type"] == 0x52] == []


def test_mine_walk_over_arrest_closes_silently() -> None:
    """The hidden-mine arrest emits 0x45, never a code 1.

    Archive 2026-07-29/30: 18 detonations, zero paired code-1s.
    """
    from tankpit_bot.sim.emissions import emit_move
    from tankpit_bot.sim.world import place_mine

    world = _world(10, 10)
    place_mine(world, 11, 10, 2)
    outcome = process_move(world, _channel_map(), 9, 12, 10)
    assert outcome["kind"] == "moved"
    assert outcome["stop_reason"] == "mine"
    assert outcome["dest_reached"] is False
    messages: list[BinaryMessage] = []
    emit_move(world, 9, outcome, messages)
    assert messages[0]["msg_type"] == 0x47
    assert any(m["msg_type"] == 0x45 for m in messages)
    assert [m for m in messages if m["msg_type"] == 0x52] == []


def test_floating_container_picks_up_while_riding() -> None:
    """A container on water drains normally from the ferry."""
    world = _world(13, 10)
    world["containers"].append(SimContainerDict(x=17, y=12, volume=200, dotted=True))
    world["tanks"][9]["fuel"] = 500
    outcome = process_move(world, _channel_map(), 9, 17, 12)
    assert outcome["kind"] == "moved"
    assert [p["remaining_volume"] for p in outcome["pickups"]] == [0]
    assert world["tanks"][9]["fuel"] == 500 - len(outcome["path"]) + 200


def test_viewport_patches_carry_ferry_tiles_and_reverts() -> None:
    """0x5A enumerates in-window ferries and reverts vacated tiles."""
    from tankpit_bot.sim.commands import ClientCommandDict
    from tankpit_bot.sim.server import SimServer

    world = _world(10, 10)
    server = SimServer(world, _channel_map(), client_id=9)
    burst = server.handshake()
    patches = [m for m in burst if m["msg_type"] == 0x5A]
    assert len(patches) == 1
    left = patches[0]["viewport_left"]
    entities = patches[0]["entities"]
    assert [(e["terrain_type"], e["col"] + left - 1) for e in entities] == [(5, 13)]
    world["ferries"][0] = SimFerryDict(x=13, y=12)
    server.queue_command(
        9,
        ClientCommandDict(
            kind="move", command=112, x=11, y=10, target_id=0, slot=0, message_id=0, direction=0
        ),
    )
    batch = server.advance_tick()
    patch = next(m for m in batch if m["msg_type"] == 0x5A)
    coded = sorted(
        (
            e["terrain_type"],
            e["col"] + patch["viewport_left"] - 1,
            e["row"] + patch["viewport_top"] - 1,
        )
        for e in patch["entities"]
    )
    assert coded == [(0, 13, 10), (5, 13, 12)]


def test_truncation_carries_the_surface_across_unclassified_tiles() -> None:
    """A hand-built path over rock keeps the last known surface.

    The router never routes across rock, but the execution walk is
    total over arbitrary step strings: an unclassifiable tile leaves
    ``previous`` unchanged instead of guessing.
    """
    world = _world(10, 10)
    terrain = InMemoryTerrainMap(terrain_data={(11, 10): _ROCK})
    walked, x, y, reason = _execute_walk(
        world, terrain, world["tanks"][9], "land", "ee", stop_on_contact=False
    )
    assert walked == "ee"
    assert (x, y) == (12, 10)
    assert reason == "exhausted"


def test_out_of_window_ferry_tiles_wait_for_the_window() -> None:
    """Patches cover only the 18x18 grid; far tiles defer their revert."""
    from tankpit_bot.sim.commands import ClientCommandDict
    from tankpit_bot.sim.server import SimServer

    world = _world(10, 10)
    server = SimServer(world, _channel_map(), client_id=9)
    server.handshake()
    assert (13, 10) in server._viewport._patched_dynamic_tiles
    world["ferries"][0] = SimFerryDict(x=13, y=12)
    server.queue_command(
        9,
        ClientCommandDict(
            kind="teleport", command=116, x=40, y=10, target_id=0, slot=0, message_id=0, direction=0
        ),
    )
    batch = server.advance_tick()
    patch = next(m for m in batch if m["msg_type"] == 0x5A)
    assert patch["entities"] == []
    assert (13, 10) in server._viewport._patched_dynamic_tiles


def test_production_world_learns_the_ferry_over_the_seam() -> None:
    """The real ingestion composes the sim's 0x5A ferry into terrain."""
    bot, _server, _link, _table = boot_seam(ferries=((104, 100),))
    _tick_once(bot)
    terrain = get_world_service().world_state["terrain"]
    tile = terrain.get("104,100")
    if tile is None:
        raise AssertionError("the seam never delivered the ferry tile")
    assert tile["terrain_type"] == 5


def test_teleport_lands_on_a_ferry_tile() -> None:
    """Boarding by teleport is legal — the F5 doctrine's core move.

    The ferry floats on water the static map calls impassable; the
    landing law reads the live ferry over the map (user,
    [[ferry-mechanics]]: "you generally will need to teleport to the
    ferry since many times it will be on its own area in the water").
    """
    from tankpit_bot.sim.actions import process_teleport

    world = _world(10, 10)
    outcome = process_teleport(world, _channel_map(), 9, 13, 10)
    tank = world["tanks"][9]
    assert outcome["kind"] == "landed"
    assert (tank["x"], tank["y"]) == (13, 10)
    assert ferry_at(world, 13, 10) == world["ferries"][0]


def test_riding_west_emits_an_ascending_viewport_patch() -> None:
    """The 0x5A skip-RLE cursor only walks forward.

    A ferry carrying its rider one tile west produces BOTH a fresh
    patch (the new tile, earlier in the walk) and a revert (the
    vacated tile, later) — the tracker must order them ascending or
    the wire encoder's delta goes negative (found by the first
    ferry-scenario soak, 2026-08-01).
    """
    from tankpit_bot.sim.viewport_window import ViewportTracker

    world = _world(13, 10)  # client standing ON the ferry at (13, 10)
    terrain = _channel_map()
    tracker = ViewportTracker(world, terrain, client_id=9)
    first = tracker.build_update()
    assert [(e["col"], e["row"], e["terrain_type"]) for e in first["entities"]] == [(9, 9, 5)]
    world["ferries"][0]["x"] = 12
    world["tanks"][9]["x"] = 12
    second = tracker.build_update()
    coded = [(e["col"], e["row"], e["terrain_type"]) for e in second["entities"]]
    assert coded == [(8, 9, 5), (9, 9, 0)]
