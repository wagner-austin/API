"""The stamp names a run; it does not choose what the run plays.

These are the pins that make a cluster sweep honest. Until 2026-09-02 a
run's stamp selected both the practice layout and the container-population
seed, so an array whose tasks stamp themselves varied the world along with
whatever parameter the sweep meant to vary. One published saturation table
was retracted over exactly that.

The laws asserted here: a named world is reproducible across stamps, two
named worlds differ under one stamp, an unknown name is refused rather
than defaulted, and artifact paths can be moved off the fixed root that N
array tasks would otherwise share.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_int,
    narrow_json_to_list,
)

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.runtime_artifacts import build_probe_run_artifacts
from tankpit_bot.sim.cli_args import (
    ARRAY_TASK_ENV_VAR,
    UnnamedWorldError,
    _parse_cli,
    require_named_world,
)
from tankpit_bot.sim.run import run_sim_session
from tankpit_bot.sim.scenarios import SIM_FIELD
from tankpit_bot.sim.world_seed import (
    PRACTICE_LAYOUT_PROVENANCES,
    PRACTICE_LAYOUTS,
    UnknownPracticeLayoutError,
    layout_by_provenance,
    population_seed_for_stamp,
    select_practice_layout,
)
from tests.conftest import FakeFileSystem
from tests.in_memory_terrain_map import InMemoryTerrainMap

_SIM_ARCHIVE = Path("runs") / "sim"


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


def _spawns(fake_fs: FakeFileSystem, world_path: str) -> set[tuple[int, int]]:
    """Read the roster spawn tiles out of an archived world.

    Args:
        fake_fs: The installed fake file system.
        world_path: Path the session wrote its world to.

    Returns:
        Every tank's (x, y), which is what a layout decides.
    """
    doc = narrow_json_to_dict(load_json_str(fake_fs.read_text(Path(world_path))))
    spawns: set[tuple[int, int]] = set()
    for entry in narrow_json_to_list(doc["tanks"]):
        tank = narrow_json_to_dict(entry)
        spawns.add((narrow_json_to_int(tank["x"]), narrow_json_to_int(tank["y"])))
    return spawns


def test_every_layout_is_reachable_by_name() -> None:
    """The provenance list tracks the table, so a new layout is selectable.

    Derived rather than hand-written precisely so adding a layout cannot
    leave it unnameable.
    """
    assert tuple(x["provenance"] for x in PRACTICE_LAYOUTS) == PRACTICE_LAYOUT_PROVENANCES
    for provenance in PRACTICE_LAYOUT_PROVENANCES:
        assert layout_by_provenance(provenance)["provenance"] == provenance


def test_an_unknown_layout_name_is_refused_not_defaulted() -> None:
    """A sweep that silently played a different world than its document
    names would produce numbers nobody could interpret."""
    with pytest.raises(UnknownPracticeLayoutError) as excinfo:
        layout_by_provenance("bot-99999999-000000")
    assert "bot-99999999-000000" in str(excinfo.value)


def test_the_stamp_derivations_are_deterministic_and_disagree() -> None:
    """Both stamp-derived values are stable, and they are different
    functions -- fixing only the layout would leave the larder moving."""
    assert select_practice_layout("s-a") is select_practice_layout("s-a")
    assert population_seed_for_stamp("s-a") == population_seed_for_stamp("s-a")
    seeds = {population_seed_for_stamp(f"s-{n}") for n in range(8)}
    assert len(seeds) > 1


def test_a_named_world_is_identical_across_two_different_stamps(
    fake_fs: FakeFileSystem,
) -> None:
    """THE LAW THIS WORK EXISTS FOR.

    Two sessions differing only in their stamp must play the same world
    once the world is named. Before the fix the stamp chose the layout
    and the container seed, so this pair differed on both and every
    number taken across them was confounded.
    """
    _install_fake_terrain(fake_fs)
    named = PRACTICE_LAYOUT_PROVENANCES[0]
    left = run_sim_session(
        6,
        archive_dir=_SIM_ARCHIVE,
        practice=True,
        stamp="20260902-000001",
        layout=named,
        population_seed=4242,
    )
    right = run_sim_session(
        6,
        archive_dir=_SIM_ARCHIVE,
        practice=True,
        stamp="20260902-999999",
        layout=named,
        population_seed=4242,
    )

    assert _spawns(fake_fs, left["world_path"]) == _spawns(fake_fs, right["world_path"])


def test_two_named_layouts_differ_under_one_stamp(fake_fs: FakeFileSystem) -> None:
    """The selector actually selects.

    The complement of the law above: holding the stamp still and naming
    two different layouts must produce two different worlds, or the
    parameter would be inert and the pin above would pass vacuously.

    THE TWO RUNS NEED SEPARATE ARCHIVE DIRECTORIES, and that is blocker 2
    in miniature rather than test hygiene. Holding the stamp still is
    exactly what a sweep does to stop it choosing the world -- and the
    archive path is stamp-derived, so both sessions write
    ``sim-20260902-000002.world.json``. Written to one directory the
    second silently overwrites the first and this assertion compares a
    file with itself. It did, on the first run of this test.
    """
    _install_fake_terrain(fake_fs)
    first = run_sim_session(
        6,
        archive_dir=_SIM_ARCHIVE / "task-1",
        practice=True,
        stamp="20260902-000002",
        layout=PRACTICE_LAYOUT_PROVENANCES[0],
        population_seed=4242,
    )
    second = run_sim_session(
        6,
        archive_dir=_SIM_ARCHIVE / "task-2",
        practice=True,
        stamp="20260902-000002",
        layout=PRACTICE_LAYOUT_PROVENANCES[1],
        population_seed=4242,
    )

    assert _spawns(fake_fs, first["world_path"]) != _spawns(fake_fs, second["world_path"])


def test_an_unnamed_world_still_follows_the_stamp(fake_fs: FakeFileSystem) -> None:
    """Interactive behaviour is unchanged.

    Absent an explicit layout the stamp still picks one, so existing
    soaks reproduce exactly. The hazard is not that the derivation
    exists -- it is that a sweep could not opt out of it.
    """
    _install_fake_terrain(fake_fs)
    result = run_sim_session(6, archive_dir=_SIM_ARCHIVE, practice=True, stamp="20260725-000001")
    expected = select_practice_layout("20260725-000001")
    spawns = _spawns(fake_fs, result["world_path"])
    for _id, _team, _rank, x, y in expected["roster"]:
        assert (x, y) in spawns


def test_the_cli_carries_the_three_new_flags() -> None:
    """A sweep member drives this through the CLI, so the flags must land."""
    parsed = _parse_cli(
        [
            "--layout",
            "bot-20260706-223721",
            "--population-seed",
            "77",
            "--runs-root",
            "runs/task-3",
        ]
    )

    assert parsed["layout"] == "bot-20260706-223721"
    assert parsed["population_seed"] == 77
    assert parsed["runs_root"] == "runs/task-3"


def test_the_cli_defaults_leave_the_world_stamp_derived() -> None:
    """None means "derive", which is what keeps interactive runs varying."""
    parsed = _parse_cli([])

    assert parsed["layout"] is None
    assert parsed["population_seed"] is None
    assert parsed["runs_root"] is None


def test_a_runs_root_moves_every_probe_artifact_off_the_shared_path() -> None:
    """N array tasks sharing a node must not share `latest.sim.log`.

    The fixed root is right on a workstation and wrong on a cluster:
    both the `latest.*` paths AND the stamped archive paths move, because
    a sweep that holds the stamp still to stop it choosing the world
    would otherwise collide on the archives too.
    """
    shared = build_probe_run_artifacts("sim", "20260902-000003")
    isolated = build_probe_run_artifacts("sim", "20260902-000003", Path("runs") / "task-7")

    assert shared["latest_log_path"] != isolated["latest_log_path"]
    assert shared["archive_events_path"] != isolated["archive_events_path"]
    for path in (
        isolated["latest_log_path"],
        isolated["archive_log_path"],
        isolated["latest_events_path"],
        isolated["archive_events_path"],
        isolated["log_dir"],
    ):
        assert path.startswith(str(Path("runs") / "task-7"))


def test_two_array_tasks_share_no_artifact_path() -> None:
    """The isolation is per task, which is the property an array needs."""
    one = build_probe_run_artifacts("sim", "same-stamp", Path("runs") / "task-1")
    two = build_probe_run_artifacts("sim", "same-stamp", Path("runs") / "task-2")

    assert set(one.values()).isdisjoint(set(two.values()) - {"sim"})


def _env(**values: str) -> Callable[[str], str | None]:
    """Build an environment reader stating exactly these variables.

    Args:
        **values: The variables to report as set.

    Returns:
        A reader returning None for everything else.
    """

    def read(key: str) -> str | None:
        return values.get(key)

    return read


def test_an_interactive_run_is_not_gated() -> None:
    """Omitting the flags locally stays legal.

    The derivation is a feature for a soak; the gate must not turn every
    interactive run into an error, or it would be removed.
    """
    require_named_world(_parse_cli([]), _env())


def test_declaring_a_sweep_without_naming_the_world_is_refused() -> None:
    """`--sweep` says these numbers will be compared, so the world must
    be stated rather than left to the run's name."""
    with pytest.raises(UnnamedWorldError) as excinfo:
        require_named_world(_parse_cli(["--sweep"]), _env())
    message = str(excinfo.value)
    assert "--layout" in message
    assert "--population-seed" in message


def test_a_slurm_array_task_is_gated_without_declaring_anything() -> None:
    """THE ARM THAT MATTERS.

    The failure being guarded is a FORGOTTEN flag, and a gate you must
    remember to arm does not guard against forgetting. Slurm setting the
    array-task variable betrays the intent to compare even when nobody
    declared it, so the refusal happens on the cluster whether or not the
    member document remembered `--sweep`.
    """
    with pytest.raises(UnnamedWorldError) as excinfo:
        require_named_world(_parse_cli([]), _env(**{ARRAY_TASK_ENV_VAR: "7"}))
    assert ARRAY_TASK_ENV_VAR in str(excinfo.value)


def test_a_half_named_world_is_still_refused() -> None:
    """Naming the layout and forgetting the seed still moves the larder.

    The population seed is the half that prints nowhere, so a gate that
    accepted a partially-named world would pass exactly the runs whose
    confound is hardest to spot afterwards.
    """
    with pytest.raises(UnnamedWorldError) as excinfo:
        require_named_world(_parse_cli(["--sweep", "--layout", "bot-20260706-223721"]), _env())
    message = str(excinfo.value)
    assert "--population-seed" in message
    assert "--layout" not in message


def test_a_fully_named_sweep_member_passes() -> None:
    """The gate lets through exactly what a sweep member must supply."""
    require_named_world(
        _parse_cli(["--sweep", "--layout", "bot-20260706-223721", "--population-seed", "5"]),
        _env(**{ARRAY_TASK_ENV_VAR: "7"}),
    )


def test_an_empty_array_task_variable_does_not_arm_the_gate() -> None:
    """Set-but-empty is not an array task.

    A shell that exports the name without a value would otherwise gate
    every local run, which is how a guard earns its way out of the tree.
    """
    require_named_world(_parse_cli([]), _env(**{ARRAY_TASK_ENV_VAR: ""}))
