"""``tankpit-sim-run`` — the sim session CLI and its scenario laws."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict, narrow_json_to_list

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH
from tankpit_bot.sim.run import (
    SIM_FIELD,
    _require_seeds_passable,
    main,
    make_default_sim_world,
    run_sim_session,
)
from tankpit_bot.sim.world_seed import (
    DOTTED_FUEL_COUNT,
    HIDDEN_EQUIPMENT_COUNT,
    HIDDEN_FUEL_COUNT,
    select_practice_layout,
)
from tankpit_bot.types import decode_capture_session
from tests.conftest import FakeFileSystem
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _install_fake_terrain(fake_fs: FakeFileSystem) -> None:
    """Provide the field GIF and an all-passable terrain loader.

    Args:
        fake_fs: The installed fake file system.
    """
    fake_fs.write_text(Path(SIM_FIELD), "fake-gif-bytes")

    def load_fake_terrain(gif_path: Path) -> TerrainMapProtocol:
        """Return an open in-memory terrain for any requested field."""
        del gif_path
        return InMemoryTerrainMap()

    _test_hooks.load_terrain_map = load_fake_terrain


def test_default_scenario_seeds_are_passable_on_the_real_field() -> None:
    """The shipped arena must sit on genuinely open field01 ground.

    The first real-terrain run seeded the coastal (100, 100) region
    and drowned six containers — this pin keeps the default honest
    against the actual GIF (a git-tracked repo asset).
    """
    terrain = _test_hooks.load_terrain_map(Path(SIM_FIELD))
    _require_seeds_passable(make_default_sim_world(), terrain)


def test_seed_validation_rejects_impassable_tiles() -> None:
    """Seeds on rock — tank, container, or equipment — fail loudly."""
    world = make_default_sim_world()
    client = world["tanks"][9]
    container = world["containers"][0]
    equipment = world["equipment"][0]
    walls = {
        (client["x"], client["y"]): "#",
        (container["x"], container["y"]): "#",
        (equipment["x"], equipment["y"]): "#",
    }
    with pytest.raises(RuntimeError, match="fuel container"):
        _require_seeds_passable(world, InMemoryTerrainMap(terrain_data=walls))


def test_run_writes_capture_world_and_events(fake_fs: FakeFileSystem) -> None:
    """A short session archives all three artifacts in standard shapes."""
    _install_fake_terrain(fake_fs)
    result = run_sim_session(12, opponent=True, stamp="20260722-000004")
    assert result["stamp"] == "20260722-000004"
    assert 0 < result["rounds_played"] <= 12
    assert result["commands_sent"] > 0
    files = fake_fs.get_written_files()
    capture = decode_capture_session(
        narrow_json_to_dict(load_json_str(files[result["capture_path"]]))
    )
    assert capture["magic"] == "simmagic"
    assert capture["messages"] != []
    world_snapshot = narrow_json_to_dict(load_json_str(files[result["world_path"]]))
    assert world_snapshot["field"] == SIM_FIELD
    assert files[result["events_path"]] != ""


def test_run_records_a_production_session_exit(fake_fs: FakeFileSystem) -> None:
    """A long-enough session ends via the production exit path.

    On the open fake terrain the deterministic default scenario plays
    to a real ``SessionExitError`` well before 200 rounds — the run
    must record that reason instead of crashing.
    """
    _install_fake_terrain(fake_fs)
    result = run_sim_session(200, opponent=True, stamp="20260722-000005")
    assert result["rounds_played"] < 200
    assert result["exit_reason"] in (
        "no_viable_targets",
        "out_of_fuel",
        "no_productive_collect",
        "deactivated",
    )
    assert result["exit_detail"] != ""


def test_boot_refuses_missing_terrain_and_missing_key(fake_fs: FakeFileSystem) -> None:
    """Both boot preconditions fail loudly, never best-effort."""
    with pytest.raises(RuntimeError, match="terrain GIF"):
        run_sim_session(1, opponent=False, stamp="20260722-000006")
    _install_fake_terrain(fake_fs)
    fake_fs.remove(DEFAULT_STATIC_KEY_PATH)
    with pytest.raises(RuntimeError, match="XOR static key"):
        run_sim_session(1, opponent=False, stamp="20260722-000007")


def test_main_parses_arguments_and_reports(fake_fs: FakeFileSystem) -> None:
    """The manual argument loop drives one session and returns 0."""
    _install_fake_terrain(fake_fs)
    exit_code = main(
        ["--rounds", "6", "--no-opponent", "--stamp", "20260722-000008", "--mystery", "--rounds"]
    )
    assert exit_code == 0
    files = fake_fs.get_written_files()
    assert any("sim-20260722-000008.capture_session.json" in path for path in files)


def test_practice_session_seeds_the_mined_layout_and_population(
    fake_fs: FakeFileSystem,
) -> None:
    """Practice mode seeds the stamp-selected real layout: the full
    36-bot roster, the client at its mined join spawn, and the static
    container field (dotted atlas + hidden fuel + hidden equipment).
    The session archives normally."""
    _install_fake_terrain(fake_fs)
    result = run_sim_session(14, practice=True, stamp="20260725-000001")
    assert result["rounds_played"] >= 1
    layout = select_practice_layout("20260725-000001")
    world_doc = narrow_json_to_dict(load_json_str(fake_fs.read_text(Path(result["world_path"]))))
    raw_tanks = narrow_json_to_list(world_doc["tanks"])
    tank_ids = {narrow_json_to_dict(entry)["tank_id"] for entry in raw_tanks}
    for roster_id, _team, _rank, _x, _y in layout["roster"]:
        assert roster_id in tank_ids
    raw_containers = narrow_json_to_list(world_doc["containers"])
    dotted = sum(1 for entry in raw_containers if narrow_json_to_dict(entry)["dotted"] is True)
    hidden = sum(1 for entry in raw_containers if narrow_json_to_dict(entry)["dotted"] is False)
    # Conservation: exposure moves containers hidden -> dotted, never
    # creates or destroys them (the measured-density world is sparse
    # enough that a short session may dot nothing; the exposure law
    # itself is pinned by test_radar_exposure_dots_large_hidden_fuel).
    assert dotted + hidden == DOTTED_FUEL_COUNT + HIDDEN_FUEL_COUNT
    assert dotted >= DOTTED_FUEL_COUNT
    assert len(narrow_json_to_list(world_doc["equipment"])) >= HIDDEN_EQUIPMENT_COUNT


def test_main_practice_flag_drives_a_roster_session(fake_fs: FakeFileSystem) -> None:
    """`--practice` reaches run_sim_session and archives normally."""
    _install_fake_terrain(fake_fs)
    exit_code = main(["--rounds", "4", "--practice", "--stamp", "20260725-000002"])
    assert exit_code == 0
    files = fake_fs.get_written_files()
    assert any("sim-20260725-000002.capture_session.json" in path for path in files)
