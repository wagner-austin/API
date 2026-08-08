"""Law 6b — movable concrete blocks (wiki [[movable-blocks]])."""

from __future__ import annotations

from tankpit_bot.bot.tick_body import _tick_once
from tankpit_bot.sim.blocks import (
    BLOCK_BRIDGE,
    BLOCK_LAND,
    BLOCK_STACKED,
    block_tile_value,
    process_block_press,
)
from tankpit_bot.sim.combat import WEAPON_SINGLE, process_shot
from tankpit_bot.sim.commands import ClientCommandDict
from tankpit_bot.sim.movement import process_move, tile_surface
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.world import (
    SimBlockDict,
    SimWorldDict,
    make_sim_tank,
    make_sim_world,
    place_mine,
)
from tankpit_bot.sniffer.world_service import WorldService
from tests.in_memory_terrain_map import InMemoryTerrainMap
from tests.sim.seam import boot_seam

_NOBODY: frozenset[int] = frozenset()
_WATER = "W"


def _world() -> SimWorldDict:
    """Tank 9 at (10, 10) beside a land block at (11, 10)."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 10, 10, 1000)
    world["blocks"].append(SimBlockDict(x=11, y=10))
    return world


def _map_with_pond() -> InMemoryTerrainMap:
    """Ground everywhere except a pond at x in 13..15, y in 8..12."""
    pond = {(x, y): _WATER for x in range(13, 16) for y in range(8, 13)}
    return InMemoryTerrainMap(terrain_data=pond)


def test_tile_values_derive_from_context() -> None:
    """One block: land=2, water=1; two on water: 3."""
    world = _world()
    terrain = _map_with_pond()
    assert block_tile_value(world, terrain, 11, 10) == BLOCK_LAND
    world["blocks"].append(SimBlockDict(x=13, y=10))
    assert block_tile_value(world, terrain, 13, 10) == BLOCK_BRIDGE
    world["blocks"].append(SimBlockDict(x=13, y=10))
    assert block_tile_value(world, terrain, 13, 10) == BLOCK_STACKED
    assert block_tile_value(world, terrain, 5, 5) == 0


def test_pickup_requires_cardinal_adjacency_and_reports_direction() -> None:
    """The 'e' pickup clears the tile and sets the carry flag; a far
    press is out of reach."""
    world = _world()
    far = process_block_press(world, _map_with_pond(), 9, 20, 20)
    assert far["kind"] == "out_of_reach"
    empty = process_block_press(world, _map_with_pond(), 9, 9, 10)
    assert empty["kind"] == "out_of_reach"
    picked = process_block_press(world, _map_with_pond(), 9, 11, 10)
    assert picked["kind"] == "picked_up"
    assert picked["direction"] == ord("e")
    assert picked["tile_value"] == 0
    assert world["blocks"] == []
    assert world["tanks"][9]["carrying"] is True
    placed = process_block_press(world, _map_with_pond(), 9, 10, 11)
    assert world["tanks"][9]["carrying"] is False
    assert placed["kind"] == "dropped"


def test_drop_on_land_kills_any_mine_silently() -> None:
    """A land drop destroys mines of EVERY team with no detonation."""
    world = _world()
    world["tanks"][9]["carrying"] = True
    world["blocks"] = []
    # One mine per tile now: the second placement replaces the first,
    # which is the invariant the drop then clears.
    place_mine(world, 10, 11, 0)
    place_mine(world, 10, 11, 1)
    dropped = process_block_press(world, _map_with_pond(), 9, 10, 11)
    assert dropped["kind"] == "dropped"
    assert dropped["tile_value"] == BLOCK_LAND
    assert world["mines"] == {}
    assert world["tanks"][9]["carrying"] is False


def test_water_drop_builds_a_bridge_and_stacks_once() -> None:
    """First water drop bridges (1); the second stacks (3); a third refuses."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 12, 10, 1000)
    terrain = _map_with_pond()
    for expected in (BLOCK_BRIDGE, BLOCK_STACKED):
        world["tanks"][9]["carrying"] = True
        dropped = process_block_press(world, terrain, 9, 13, 10)
        assert (dropped["kind"], dropped["tile_value"]) == ("dropped", expected)
    world["tanks"][9]["carrying"] = True
    assert process_block_press(world, terrain, 9, 13, 10)["kind"] == "refused"
    assert process_block_press(world, terrain, 9, 11, 10)["kind"] == "dropped"


def test_land_block_refuses_a_second_block_and_rock_refuses_all() -> None:
    """No stacking on land; rock never accepts a drop."""
    world = _world()
    world["tanks"][9]["carrying"] = True
    refused = process_block_press(world, _map_with_pond(), 9, 11, 10)
    assert refused["kind"] == "refused"
    rock = InMemoryTerrainMap(terrain_data={(10, 11): "#"})
    assert process_block_press(world, rock, 9, 10, 11)["kind"] == "refused"


def test_bridge_walks_and_obstacles_block_routing() -> None:
    """A bridge is ordinary ground; a land block is impassable."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 12, 10, 1000)
    terrain = _map_with_pond()
    world["blocks"].extend(
        [SimBlockDict(x=13, y=10), SimBlockDict(x=14, y=10), SimBlockDict(x=15, y=10)]
    )
    assert tile_surface(world, terrain, 13, 10) == "land"
    outcome = process_move(world, terrain, 9, 17, 10)
    assert outcome["kind"] == "moved"
    assert (world["tanks"][9]["x"], world["tanks"][9]["y"]) == (17, 10)
    world["blocks"].append(SimBlockDict(x=16, y=10))
    assert tile_surface(world, terrain, 16, 10) is None


def test_land_blocks_obstruct_shots_but_bridges_do_not() -> None:
    """A land block clips the ray; a flat bridge lets it pass."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 10, 10, 1000)
    world["tanks"][11] = make_sim_tank(11, 1, 1, 15, 10, 500)
    world["blocks"].append(SimBlockDict(x=12, y=10))
    outcome = process_shot(world, InMemoryTerrainMap(), 9, 15, 10, _NOBODY, 0, None)
    assert outcome["weapon"] == WEAPON_SINGLE
    assert (outcome["impact_x"], outcome["impact_y"]) == (12, 10)
    bridge_world = make_sim_world("field01_r.gif")
    bridge_world["tanks"][9] = make_sim_tank(9, 0, 1, 12, 10, 1000)
    bridge_world["tanks"][11] = make_sim_tank(11, 1, 1, 17, 10, 500)
    bridge_world["tanks"][9]["counts"][1] = 3
    bridge_world["blocks"].append(SimBlockDict(x=14, y=10))
    terrain = _map_with_pond()
    hit = process_shot(bridge_world, terrain, 9, 17, 10, _NOBODY, 0, None)
    assert hit["victim_id"] == 11


def test_server_emits_0x42_0x4a_and_refreshed_viewport() -> None:
    """A drop rides the batch as BuildPickup + TerrainUpdate + 0x5A."""
    world = _world()
    world["blocks"] = []
    world["tanks"][9]["carrying"] = True
    server = SimServer(world, _map_with_pond(), client_id=9)
    press = ClientCommandDict(
        kind="block", command=98, x=10, y=11, target_id=0, slot=0, message_id=0, direction=0
    )
    server.queue_command(9, press)
    messages = server.advance_tick()
    kinds = [m["msg_type"] for m in messages]
    assert kinds[:3] == [0x42, 0x4A, 0x5A]
    build = messages[0]
    assert build["msg_type"] == 0x42
    assert (build["drop_x"], build["drop_y"], build["obstacle_type"]) == (10, 11, BLOCK_LAND)
    tiles = messages[1]
    assert tiles["msg_type"] == 0x4A
    assert tiles["updates"] == [(10, 11, BLOCK_LAND)]
    patch = messages[2]
    assert patch["msg_type"] == 0x5A
    coded = [
        (
            e["terrain_type"],
            e["col"] + patch["viewport_left"] - 1,
            e["row"] + patch["viewport_top"] - 1,
        )
        for e in patch["entities"]
    ]
    assert coded == [(BLOCK_LAND, 10, 11)]


def test_out_of_reach_press_answers_code_1_for_the_client_only() -> None:
    """The measured 0x52 code 1 for a far press; enemies stay silent."""
    world = _world()
    world["tanks"][11] = make_sim_tank(11, 1, 1, 20, 20, 500)
    server = SimServer(world, _map_with_pond(), client_id=9)
    far = ClientCommandDict(
        kind="block", command=98, x=30, y=30, target_id=0, slot=0, message_id=0, direction=0
    )
    server.queue_command(9, far)
    own = server.advance_tick()
    assert [m["error_code"] for m in own if m["msg_type"] == 0x52] == [1]
    server.queue_command(11, far)
    other = server.advance_tick()
    assert [m for m in other if m["msg_type"] == 0x52] == []


def test_towing_refuses_teleport_with_code_0() -> None:
    """The measured three-for-three towing teleport refusal."""
    world = _world()
    world["tanks"][9]["carrying"] = True
    server = SimServer(world, _map_with_pond(), client_id=9)
    hop = ClientCommandDict(
        kind="teleport", command=116, x=40, y=40, target_id=0, slot=0, message_id=0, direction=0
    )
    server.queue_command(9, hop)
    messages = server.advance_tick()
    assert [m["error_code"] for m in messages if m["msg_type"] == 0x52] == [0]
    assert (world["tanks"][9]["x"], world["tanks"][9]["y"]) == (10, 10)


def test_pickup_walks_past_other_blocks_in_the_registry() -> None:
    """The pickup removes the block at the TARGET tile, not the first."""
    world = _world()
    world["blocks"].insert(0, SimBlockDict(x=30, y=30))
    picked = process_block_press(world, _map_with_pond(), 9, 11, 10)
    assert picked["kind"] == "picked_up"
    assert world["blocks"] == [SimBlockDict(x=30, y=30)]


def test_enemy_block_actions_stay_per_recipient() -> None:
    """An enemy's tow-refusal and drop never address the client."""
    world = _world()
    world["tanks"][11] = make_sim_tank(11, 1, 1, 20, 20, 500)
    world["tanks"][11]["carrying"] = True
    server = SimServer(world, _map_with_pond(), client_id=9)
    hop = ClientCommandDict(
        kind="teleport", command=116, x=40, y=40, target_id=0, slot=0, message_id=0, direction=0
    )
    server.queue_command(11, hop)
    towed = server.advance_tick()
    assert [m for m in towed if m["msg_type"] == 0x52] == []
    drop = ClientCommandDict(
        kind="block", command=98, x=20, y=21, target_id=0, slot=0, message_id=0, direction=0
    )
    server.queue_command(11, drop)
    dropped = server.advance_tick()
    kinds = [m["msg_type"] for m in dropped]
    assert 0x42 in kinds
    assert 0x4A in kinds
    assert 0x5A not in kinds


def test_blocks_refuse_teleport_landings_and_mine_placement() -> None:
    """A block tile rejects hop landings and is skipped by a press."""
    from tankpit_bot.sim.actions import process_mine_press, process_teleport

    world = _world()
    walls = {(30 + dx, 30 + dy): "#" for dx, dy in ((1, 0), (0, -1), (-1, 0), (0, 1))}
    world["blocks"].append(SimBlockDict(x=30, y=30))
    blocked = process_teleport(world, InMemoryTerrainMap(terrain_data=walls), 9, 30, 30)
    assert blocked["kind"] == "blocked"
    pressed = process_mine_press(world, _map_with_pond(), 9)
    assert (11, 10) not in pressed["placed"]
    assert len(pressed["placed"]) == 8


def test_seam_command_service_carries_the_block_press() -> None:
    """The real command bytes decode into the sim's block kind."""
    from tankpit_bot.protocol.commands import build_block_command

    bot, _server, link, _table = boot_seam(blocks=((101, 100),))
    assert bot._send_bytes(build_block_command(101, 100), "block") is True
    assert link.sent_commands == ["block"]


def test_production_world_learns_blocks_over_the_seam() -> None:
    """The real ingestion composes sim blocks into wire terrain."""
    bot, _server, _link, _table = boot_seam(blocks=((104, 101),))
    _tick_once(bot)
    tile = get_world_terrain(bot.world).get("104,101")
    if tile is None:
        raise AssertionError("the seam never delivered the block tile")
    assert tile["terrain_type"] == BLOCK_LAND


def get_world_terrain(ws: WorldService) -> dict[str, dict[str, int]]:
    """Return ``ws``'s wire-terrain registry."""

    terrain: dict[str, dict[str, int]] = {}
    for key, tile in ws.world_state["terrain"].items():
        terrain[key] = {"terrain_type": tile["terrain_type"]}
    return terrain
