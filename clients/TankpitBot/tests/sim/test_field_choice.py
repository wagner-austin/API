"""Playing the sim on another shipped field: naming it, refusing the field01-bound
scenarios, settling field01-placed seeds onto open ground, and the runs that use it."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks
from tankpit_bot.resources import data_directory
from tankpit_bot.sim.field_choice import (
    SETTLE_RADIUS,
    FieldChoiceError,
    require_field_scenario,
    resolve_field,
    settle_on_field,
)
from tankpit_bot.sim.field_run import run_field_session
from tankpit_bot.sim.run import main, run_sim_session
from tankpit_bot.sim.scenarios import SIM_CLIENT_ID, SIM_FIELD, make_default_sim_world
from tankpit_bot.sim.world import (
    SimContainerDict,
    SimEquipmentDict,
    decode_sim_world,
    make_sim_tank,
    make_sim_world,
)
from tests.conftest import FakeFileSystem
from tests.in_memory_terrain_map import InMemoryTerrainMap

_DESERT = "field05_r.gif"
_ARCHIVE = Path("runs") / "desert"


def test_a_field_resolves_to_its_shipped_terrain_with_or_without_the_suffix() -> None:
    """The server's name and the bare name both find the minimap."""
    assert resolve_field("field05") == _DESERT
    assert resolve_field("field05.gif") == _DESERT
    with pytest.raises(
        FieldChoiceError, match="SIM_FIELD_UNKNOWN: no shipped minimap for 'field99'"
    ):
        resolve_field("field99")


def test_field01_bound_scenarios_refuse_another_field_only() -> None:
    """Every scenario plays field01; the bound ones are named when refused."""
    require_field_scenario(SIM_FIELD, ferry=True, larder=True, atlas=True, ghost=True)
    require_field_scenario(_DESERT, ferry=False, larder=False, atlas=False, ghost=False)
    with pytest.raises(
        FieldChoiceError,
        match="SIM_FIELD_SCENARIO: --ferry, --ghost replay or rely on field01 and cannot play",
    ):
        require_field_scenario(_DESERT, ferry=True, larder=False, atlas=False, ghost=True)


def test_seeds_on_rock_move_to_the_nearest_open_tile() -> None:
    """Tanks, containers and equipment on rock move one ring out; open ones stay."""
    world = make_sim_world(_DESERT)
    world["tanks"][9] = make_sim_tank(9, 1, 1, 50, 50, 900)
    world["tanks"][10] = make_sim_tank(10, 2, 1, 80, 80, 900)
    world["containers"] = [SimContainerDict(x=60, y=60, volume=500, dotted=False)]
    world["equipment"] = [SimEquipmentDict(x=70, y=70)]
    terrain = InMemoryTerrainMap(terrain_data={(50, 50): "#", (60, 60): "#", (70, 70): "W"})
    assert settle_on_field(world, terrain) == 3
    for x, y in (
        (world["tanks"][9]["x"], world["tanks"][9]["y"]),
        (world["containers"][0]["x"], world["containers"][0]["y"]),
        (world["equipment"][0]["x"], world["equipment"][0]["y"]),
    ):
        assert terrain.is_passable(x, y)
    assert max(abs(world["tanks"][9]["x"] - 50), abs(world["tanks"][9]["y"] - 50)) == 1
    assert (world["tanks"][10]["x"], world["tanks"][10]["y"]) == (80, 80)


def test_a_seed_walled_in_beyond_one_viewport_is_refused() -> None:
    """Rock everywhere leaves nowhere to settle, and that is said, not guessed."""
    world = make_sim_world(_DESERT)
    world["tanks"][9] = make_sim_tank(9, 1, 1, 50, 50, 900)
    with pytest.raises(
        FieldChoiceError, match=f"SIM_FIELD_UNSETTLED: no open tile within {SETTLE_RADIUS} of"
    ):
        settle_on_field(world, InMemoryTerrainMap(default="#"))


@pytest.fixture()
def desert_fs(fake_fs: FakeFileSystem) -> Generator[FakeFileSystem, None, None]:
    """field05 shipped, with rock under the arena client's field01 spawn.

    Yields:
        The installed fake file system.
    """
    fake_fs.write_text(data_directory() / _DESERT, "fake-gif-bytes")
    fake_fs.write_text(data_directory() / SIM_FIELD, "fake-gif-bytes")
    client = make_default_sim_world()["tanks"][SIM_CLIENT_ID]
    rock = {(client["x"], client["y"]): "#"}
    original = _test_hooks.load_terrain_map

    def desert(gif_path: Path) -> InMemoryTerrainMap:
        """Rock at the client's arena spawn on field05, open ground elsewhere."""
        return InMemoryTerrainMap(terrain_data=rock if gif_path.name == _DESERT else {})

    _test_hooks.load_terrain_map = desert
    yield fake_fs
    _test_hooks.load_terrain_map = original


def test_a_session_plays_the_named_field_with_its_seeds_settled(
    desert_fs: FakeFileSystem,
) -> None:
    """The archived world names field05, and the client stands on open ground."""
    run = run_sim_session(
        2, archive_dir=_ARCHIVE, stamp="desert-1", layout="bot-20260706-223721", field="field05"
    )
    world = decode_sim_world(
        narrow_json_to_dict(load_json_str(desert_fs.read_text(Path(run["world_path"]))))
    )
    assert world["field"] == _DESERT
    arena_spawn = make_default_sim_world()["tanks"][SIM_CLIENT_ID]
    client = world["tanks"][SIM_CLIENT_ID]
    assert (client["x"], client["y"]) != (arena_spawn["x"], arena_spawn["y"])


def test_a_field01_bound_scenario_refuses_another_field_before_writing(
    desert_fs: FakeFileSystem,
) -> None:
    """The ferry lake is field01's; nothing plays and nothing is archived."""
    with pytest.raises(FieldChoiceError, match="SIM_FIELD_SCENARIO: --ferry"):
        run_sim_session(2, archive_dir=_ARCHIVE, ferry=True, field="field05")
    assert not any(path.startswith("runs") for path in desert_fs.get_written_files())


def test_a_field_session_plays_the_named_field(desert_fs: FakeFileSystem) -> None:
    """Several bots take the field the same way one does."""
    result = run_field_session(
        2,
        clients=2,
        archive_dir=_ARCHIVE,
        stamp="desert-2",
        layout="bot-20260706-223721",
        population_seed=7,
        field="field05",
    )
    world = decode_sim_world(
        narrow_json_to_dict(load_json_str(desert_fs.read_text(Path(result["world_path"]))))
    )
    assert world["field"] == _DESERT


def test_the_cli_passes_field_to_both_kinds_of_session(
    desert_fs: FakeFileSystem, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--field`` reaches a one-bot session and a field of bots alike."""
    common = ["--rounds", "1", "--layout", "bot-20260706-223721", "--population-seed", "7"]
    common += ["--out", str(_ARCHIVE), "--field", "field05"]
    assert main([*common, "--stamp", "cli-one"]) == 0
    assert main([*common, "--stamp", "cli-two", "--clients", "2"]) == 0
    capsys.readouterr()
    for name in ("sim-cli-one.world.json", "sim-cli-two.world.json"):
        text = desert_fs.read_text(_ARCHIVE / name)
        assert decode_sim_world(narrow_json_to_dict(load_json_str(text)))["field"] == _DESERT
