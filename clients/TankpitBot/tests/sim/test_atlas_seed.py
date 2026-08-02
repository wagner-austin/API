"""Atlas seeding — the mined real field as a sim world population."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str

from tankpit_bot.sim.atlas_seed import seed_atlas_population
from tankpit_bot.sim.world import SimWorldDict, make_sim_tank, make_sim_world
from tests.conftest import FakeFileSystem
from tests.in_memory_terrain_map import InMemoryTerrainMap

_ATLAS_PATH = Path("runs/analysis/container_atlas.json")


def _entry(last_v: int, visible: bool) -> dict[str, int | bool]:
    """One atlas tile entry with the fields the seeder reads."""
    return {
        "observations": 3,
        "sessions": 2,
        "first_ms": 1_000,
        "last_ms": 2_000,
        "last_v": last_v,
        "max_fuel": max(last_v, 0),
        "equipment_seen": last_v == -1,
        "visible_seen": visible,
    }


def _write_atlas(fake_fs: FakeFileSystem, tiles: dict[str, dict[str, int | bool]]) -> None:
    fake_fs.write_text(_ATLAS_PATH, dump_json_str({"1|field01.gif": tiles}))


def _world() -> SimWorldDict:
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 2, 1, 50, 50, 1100)
    return world


def test_atlas_tiles_classify_into_the_sim_vocabulary(fake_fs: FakeFileSystem) -> None:
    """Fuel keeps its mined volume and exposure; equipment and drained
    dots seed as themselves; radar-only empties seed nothing."""
    _write_atlas(
        fake_fs,
        {
            "10,10": _entry(700, visible=True),  # exposed fuel -> dotted
            "11,10": _entry(340, visible=False),  # radar-only fuel -> hidden
            "12,10": _entry(-1, visible=True),  # equipment
            "13,10": _entry(0, visible=True),  # drained dot (answers code 4)
            "14,10": _entry(0, visible=False),  # empty hidden tile: nothing
        },
    )
    world = _world()

    tally = seed_atlas_population(world, InMemoryTerrainMap(), _ATLAS_PATH)

    assert tally == {
        "fuel": 2,
        "drained_dots": 1,
        "equipment": 1,
        "water_tiles": 0,
        "rock_skipped": 0,
    }
    by_tile = {(c["x"], c["y"]): c for c in world["containers"]}
    assert by_tile[(10, 10)] == {"x": 10, "y": 10, "volume": 700, "dotted": True}
    assert by_tile[(11, 10)] == {"x": 11, "y": 10, "volume": 340, "dotted": False}
    assert by_tile[(13, 10)] == {"x": 13, "y": 10, "volume": 0, "dotted": True}
    assert (14, 10) not in by_tile
    assert [(e["x"], e["y"]) for e in world["equipment"]] == [(12, 10)]


def test_water_tiles_seed_and_rock_tiles_skip(fake_fs: FakeFileSystem) -> None:
    """The water-locked population is real; rock is a mining artifact."""
    _write_atlas(
        fake_fs,
        {
            "20,20": _entry(500, visible=False),  # water -> kept
            "21,20": _entry(500, visible=False),  # rock -> skipped
        },
    )
    world = _world()
    terrain = InMemoryTerrainMap(terrain_data={(20, 20): "W", (21, 20): "#"})

    tally = seed_atlas_population(world, terrain, _ATLAS_PATH)

    assert tally["fuel"] == 1
    assert tally["water_tiles"] == 1
    assert tally["rock_skipped"] == 1
    assert [(c["x"], c["y"]) for c in world["containers"]] == [(20, 20)]


def test_occupied_tank_tiles_are_left_open(fake_fs: FakeFileSystem) -> None:
    """A tile a living tank stands on seeds nothing."""
    _write_atlas(fake_fs, {"50,50": _entry(900, visible=True)})
    world = _world()

    tally = seed_atlas_population(world, InMemoryTerrainMap(), _ATLAS_PATH)

    assert tally["fuel"] == 0
    assert world["containers"] == []


def test_missing_field_entry_fails_loudly(fake_fs: FakeFileSystem) -> None:
    """An atlas without this field's room is a wrong-file error."""
    fake_fs.write_text(_ATLAS_PATH, dump_json_str({"2|field42.gif": {}}))
    with pytest.raises(RuntimeError, match="no entry for field"):
        seed_atlas_population(_world(), InMemoryTerrainMap(), _ATLAS_PATH)


def test_unreadable_atlas_fails_loudly(fake_fs: FakeFileSystem) -> None:
    """A missing or corrupt atlas file never seeds a silent empty world."""
    with pytest.raises(RuntimeError, match="unreadable"):
        seed_atlas_population(_world(), InMemoryTerrainMap(), _ATLAS_PATH)
    fake_fs.write_text(_ATLAS_PATH, "{not json")
    with pytest.raises(RuntimeError, match="unreadable"):
        seed_atlas_population(_world(), InMemoryTerrainMap(), _ATLAS_PATH)
