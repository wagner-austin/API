"""Tests for sim ghost emissions."""

from __future__ import annotations

from platform_core.json_utils import (
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_int,
    narrow_json_to_list,
)

from tankpit_bot.sim.ghost import (
    GhostTracker,
    compile_ghost_spec,
)
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.world import (
    make_sim_tank,
    make_sim_world,
)
from tests.conftest import FakeFileSystem
from tests.in_memory_terrain_map import InMemoryTerrainMap
from tests.sim._ghost_fixtures import (
    _fight_capture,
    _rich_capture,
)


def test_tracker_waits_for_the_first_recorded_tile() -> None:
    """Rounds before any recorded position compare against nothing."""
    tracker = GhostTracker({5: (100, 100)})
    tracker.note_round(0, 10, 10)
    assert tracker.compared_ticks == 0
    tracker.note_round(5, 100, 100)
    assert tracker.compared_ticks == 1


def test_seeding_skips_rock_but_keeps_water(fake_fs: FakeFileSystem) -> None:
    """Wire-real reads on water seed; rock reads are skipped."""
    from tankpit_bot.sim.ghost import seed_ghost_world_population

    spec = compile_ghost_spec(_rich_capture())
    terrain = InMemoryTerrainMap(terrain_data={(140, 90): "W", (143, 90): "#", (141, 90): "#"})
    from tankpit_bot.sim.world import SimContainerDict, SimEquipmentDict

    containers: list[SimContainerDict] = []
    equipment: list[SimEquipmentDict] = []
    skipped = seed_ghost_world_population(containers, equipment, spec, terrain)
    tiles = {(c["x"], c["y"]) for c in containers}
    assert (140, 90) in tiles  # water: kept
    assert (143, 90) not in tiles  # rock: skipped
    assert equipment == []  # rock equipment skipped
    assert skipped == 2
    open_containers: list[SimContainerDict] = []
    open_equipment: list[SimEquipmentDict] = []
    assert (
        seed_ghost_world_population(open_containers, open_equipment, spec, InMemoryTerrainMap())
        == 0
    )
    assert [(e["x"], e["y"]) for e in open_equipment] == [(141, 90)]


def test_ghost_atlas_composition_underlays_the_mined_room(fake_fs: FakeFileSystem) -> None:
    """``--ghost --from-atlas``: the atlas fills unobserved tiles, the
    recording's own dot atlas is the exposed set, and the capture's
    per-tile reads override the underlay."""
    from pathlib import Path

    from tankpit_bot import _test_hooks
    from tankpit_bot.sim.atlas_seed import DEFAULT_ATLAS_PATH
    from tankpit_bot.sim.run import main
    from tankpit_bot.sim.scenarios import SIM_FIELD

    entry = {
        "observations": 4,
        "sessions": 2,
        "first_ms": 1_000,
        "last_ms": 2_000,
        "max_fuel": 0,
        "equipment_seen": False,
        "visible_seen": True,
    }
    atlas_tiles = {
        # overridden by the capture's own (140, 90) read
        "140,90": {**entry, "last_v": 999, "max_fuel": 999},
        # unobserved by the capture, in its dot atlas -> dotted
        "150,95": {**entry, "last_v": 300, "max_fuel": 300},
        # unobserved, NOT in the dot atlas -> hidden
        "60,60": {**entry, "last_v": 400, "max_fuel": 400},
        # drained, not in the dot atlas -> seeds nothing in ghost mode
        "61,60": {**entry, "last_v": 0},
    }
    fake_fs.write_text(DEFAULT_ATLAS_PATH, dump_json_str({"1|field01.gif": atlas_tiles}))
    fake_fs.write_text(Path(SIM_FIELD), "fake-gif-bytes")
    # A BOT-named ghost: the reactive-policy driver constructs over it
    # (the certified roster policy under the recorded timeline).
    fake_fs.write_text(Path("runs/ghost-input.capture_session.json"), _fight_capture("orange-2"))
    _test_hooks.load_terrain_map = lambda gif_path: InMemoryTerrainMap()
    exit_code = main(
        [
            "--ghost",
            "runs/ghost-input.capture_session.json",
            "--from-atlas",
            "--rounds",
            "3",
            "--stamp",
            "20260801-000005",
        ]
    )
    assert exit_code == 0
    files = fake_fs.get_written_files()
    world_path = next(path for path in files if "sim-20260801-000005.world.json" in path)
    world_doc = narrow_json_to_dict(load_json_str(files[world_path]))
    containers = [narrow_json_to_dict(c) for c in narrow_json_to_list(world_doc["containers"])]
    by_tile = {(narrow_json_to_int(c["x"]), narrow_json_to_int(c["y"])): c for c in containers}
    # dot-atlas tile: dotted from the recording's exposed set
    assert by_tile[(150, 95)]["dotted"] is True
    # off-atlas underlay tile: hidden
    assert by_tile[(60, 60)]["dotted"] is False
    assert (61, 60) not in by_tile
    # the capture's own drained-dot read at (140, 90) beats the atlas
    assert narrow_json_to_int(by_tile[(140, 90)]["volume"]) == 0


def test_seed_ghost_world_relocates_or_skips_blocked_spawns() -> None:
    """A ghost sighted on impassable ground spawns at the nearest open
    tile; one with no open ground nearby is skipped."""
    from tankpit_bot.sim.ghost import GhostSpecDict, GhostTankDict
    from tankpit_bot.sim.run_boot import _seed_ghost_world

    spec = GhostSpecDict(
        client_team=0,
        client_rank=1,
        client_x=10,
        client_y=10,
        client_fuel=800,
        client_counts=[25] * 5,
        ghosts=[
            GhostTankDict(tank_id=500, team=1, rank=1, name="rider", x=40, y=40),
            GhostTankDict(tank_id=501, team=1, rank=1, name="stuck", x=80, y=80),
        ],
        events=[],
        recorded_path={0: (10, 10)},
        containers=[],
        equipment=[],
        dot_atlas=[],
        ticks=1,
        unplaced_tanks=0,
    )
    rocks = {(40, 40): "#"}
    rocks.update({(80 + dx, 80 + dy): "#" for dx in range(-5, 6) for dy in range(-5, 6)})
    world = make_sim_world("field01_r.gif")
    _seed_ghost_world(world, InMemoryTerrainMap(terrain_data=rocks), spec, None)
    assert 500 in world["tanks"]
    spawned = world["tanks"][500]
    assert (spawned["x"], spawned["y"]) != (40, 40)
    assert abs(spawned["x"] - 40) <= 4 and abs(spawned["y"] - 40) <= 4
    assert 501 not in world["tanks"]


def test_dead_ghosts_skip_their_remaining_timeline() -> None:
    """Events of a corpse are dropped, not replayed."""
    from tankpit_bot.sim.ghost import GhostEventDict, GhostSpecDict
    from tankpit_bot.sim.run_boot import _queue_ghost_round

    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 10, 10, 1000)
    world["tanks"][500] = make_sim_tank(500, 1, 1, 20, 20, 600)
    world["tanks"][500]["alive"] = False
    server = SimServer(world, InMemoryTerrainMap(), client_id=9)
    spec = GhostSpecDict(
        client_team=0,
        client_rank=1,
        client_x=10,
        client_y=10,
        client_fuel=800,
        client_counts=[25] * 5,
        ghosts=[],
        events=[
            GhostEventDict(tick=0, tank_id=500, kind="place", x=22, y=20, message_id=0),
            GhostEventDict(tick=0, tank_id=999, kind="shoot", x=10, y=10, message_id=0),
        ],
        recorded_path={},
        containers=[],
        equipment=[],
        dot_atlas=[],
        ticks=1,
        unplaced_tanks=0,
    )
    _queue_ghost_round(server, spec, 0)
    assert (world["tanks"][500]["x"], world["tanks"][500]["y"]) == (20, 20)
    batch = server.advance_tick()
    # nothing was queued: the tick carries only the client's sync
    assert [m["msg_type"] for m in batch] == [0x2E]
