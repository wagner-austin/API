"""Tests for scripts.build_sim_baseline.

The sim sessions here are REAL — the production ``_tick_once`` against
a real :class:`SimServer`, exactly as ``make sim-baseline`` runs them —
and land in the fake file system rather than the repo's ``runs`` tree.
That matters for the end-to-end test: the capture the sim just
generated is the capture the differ then reads, so the script is
checked against its own output rather than against a fixture standing
in for it.

Only the LIVE side is injected, because it is the one thing a test
cannot generate: it is the real 341-session archive, and letting it
resolve would make every assertion a function of the corpus.
"""

from __future__ import annotations

import runpy
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from scripts.build_sim_baseline import (
    BASELINE_ROOT,
    DEFAULT_ROUNDS,
    LIVE_DIRECTORIES,
    SCENARIOS,
    _int_flag,
    build_baseline,
    main,
)

from scripts import _test_hooks as script_hooks
from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.analysis import _test_hooks as analysis_hooks
from tankpit_bot.protocol.command_builders import build_teleport_command
from tankpit_bot.sim.scenarios import SIM_FIELD
from tests.analysis._capture_fixtures import (
    OWN_TANK,
    _command,
    _payload,
    _radar_result,
    _received,
    _sent,
    _session_json,
    _tank_info,
)
from tests.conftest import FakeFileSystem
from tests.in_memory_terrain_map import InMemoryTerrainMap

#: The ferry scenario's own water tile (``make_ferry_sim_world``).
_FERRY_TILE = (118, 112)

#: One live window whose teleport draws a 0x46 the sim never produces,
#: so the missing-law side of the diff has a row to report.
_LIVE_SESSION = _session_json(
    messages=[
        _received(_payload(_tank_info(OWN_TANK)), timestamp_ms=1000),
        _sent(_payload(_command(build_teleport_command(9, 9))), timestamp_ms=1100),
        _received(_payload(_radar_result()), timestamp_ms=1200),
    ]
)


@pytest.fixture()
def _sim_world(fake_fs: FakeFileSystem) -> Generator[FakeFileSystem, None, None]:
    """Give the sim a field GIF and an all-passable terrain loader.

    Yields:
        The installed fake file system, holding whatever the sessions
        wrote.
    """
    fake_fs.write_text(Path(SIM_FIELD), "fake-gif-bytes")
    real_terrain = _test_hooks.load_terrain_map

    def load_fake_terrain(gif_path: Path) -> TerrainMapProtocol:
        """Return an in-memory terrain the whole sweep can seed on.

        Open ground everywhere except the ferry scenario's own tile,
        which has to be WATER — the scenario floats a ferry there and
        the seed validator refuses to start a session whose furniture
        is on the wrong surface. On a real ``make sim-baseline`` the
        field GIF supplies that water; here the fixture must.

        Args:
            gif_path: Ignored.

        Returns:
            The terrain map every baseline scenario can seed on.
        """
        del gif_path
        return InMemoryTerrainMap(terrain_data={_FERRY_TILE: "W"})

    _test_hooks.load_terrain_map = load_fake_terrain
    try:
        yield fake_fs
    finally:
        _test_hooks.load_terrain_map = real_terrain


@pytest.fixture()
def _argv() -> Generator[None, None, None]:
    """Restore ``sys.argv`` after a test rewrites it."""
    old = sys.argv
    try:
        yield
    finally:
        sys.argv = old


def _silent(level: script_hooks.LogLevel) -> None:
    """Absorb the logging call without configuring handlers.

    Args:
        level: Ignored.
    """


def _absent(path: Path) -> bool:
    """Report every archive directory as missing.

    Args:
        path: Ignored.

    Returns:
        False, always.
    """
    del path
    return False


def _captures(fake_fs: FakeFileSystem, directory: Path) -> list[Path]:
    """Every capture the sessions archived under one directory.

    Args:
        fake_fs: The installed fake file system.
        directory: The baseline directory to list.

    Returns:
        The capture paths, sorted.
    """
    return sorted(
        Path(written)
        for written in fake_fs.get_written_files()
        if written.startswith(str(directory)) and written.endswith(".capture_session.json")
    )


def test_breadth_comes_from_scenarios_because_repetition_cannot() -> None:
    """The sim is deterministic, so every scenario must be distinct.

    Three sessions of one scenario produced byte-identical wire on
    2026-09-02. A baseline built by repetition would therefore be one
    sample wearing several filenames, which is exactly what makes a
    "the sim never does this" verdict unsafe. Each label appears once.
    """
    labels = [scenario["label"] for scenario in SCENARIOS]
    assert sorted(labels) == sorted(set(labels))
    assert len(SCENARIOS) >= 4
    assert Path("runs") / "sim-baseline" == BASELINE_ROOT


def test_the_scenarios_do_not_all_drive_the_same_command_vocabulary() -> None:
    """A sweep of five identically-flagged scenarios would be one sample.

    The flags are what make the production bot send different things,
    so at least the combat/forage and the roster/scripted splits have
    to be genuinely represented.
    """
    assert {scenario["opponent"] for scenario in SCENARIOS} == {True, False}
    assert {scenario["practice"] for scenario in SCENARIOS} == {True, False}
    assert {scenario["ferry"] for scenario in SCENARIOS} == {True, False}
    assert any(scenario["opponent_name"] for scenario in SCENARIOS)


def test_the_default_depth_reaches_past_the_bots_opening_moves() -> None:
    """Depth is the only other lever a deterministic sim has."""
    assert DEFAULT_ROUNDS >= 150


def test_the_baseline_lands_in_its_own_stamped_directory(_sim_world: FakeFileSystem) -> None:
    """Each generation gets a directory that has never held another.

    This is the whole point of the script: ``runs/sim`` accumulates
    every sim ever run, so a fidelity verdict taken over it describes
    their union rather than the current one.
    """
    archive_dir = build_baseline(3, "20260902-120000")

    assert archive_dir == BASELINE_ROOT / "20260902-120000"
    assert _captures(_sim_world, archive_dir) == sorted(
        archive_dir / f"sim-20260902-120000-{scenario['label']}.capture_session.json"
        for scenario in SCENARIOS
    )


def test_every_session_archives_its_world_beside_its_capture(
    _sim_world: FakeFileSystem,
) -> None:
    """The world snapshot rides along, as it does for any sim run."""
    archive_dir = build_baseline(3, "20260902-130000")

    written = _sim_world.get_written_files()
    for scenario in SCENARIOS:
        assert str(archive_dir / f"sim-20260902-130000-{scenario['label']}.world.json") in written


def test_an_absent_flag_takes_its_default() -> None:
    """No ``--sessions`` means the module default, not a parse error."""
    assert _int_flag([], "--rounds", 400) == 400
    assert _int_flag(["--other", "9"], "--rounds", 400) == 400


def test_a_named_flag_is_read() -> None:
    """The value following the flag is the value used."""
    assert _int_flag(["--rounds", "3"], "--rounds", 400) == 3


def test_a_mistyped_count_raises_rather_than_defaulting() -> None:
    """A flag the caller MEANT to set is never quietly ignored.

    Falling back here would play the default depth while the operator
    believed they had asked for something else.
    """
    with pytest.raises(ValueError, match="invalid literal"):
        _int_flag(["--rounds", "lots"], "--rounds", 400)


def test_main_refuses_a_missing_live_archive(
    capsys: pytest.CaptureFixture[str], _argv: None
) -> None:
    """Without the live side every sim shape reads as invented.

    That is not a fidelity verdict, it is an artefact of an empty
    comparison, so the run stops by name instead of generating a
    corpus it cannot judge.
    """
    real_exists = script_hooks.path_exists
    real_logging = script_hooks.setup_rich_logging
    script_hooks.path_exists = _absent
    script_hooks.setup_rich_logging = _silent
    sys.argv = ["build_sim_baseline"]
    try:
        with pytest.raises(SystemExit) as excinfo:
            main()
    finally:
        script_hooks.path_exists = real_exists
        script_hooks.setup_rich_logging = real_logging

    assert excinfo.value.code == 1
    assert capsys.readouterr().out == f"No such directory: {LIVE_DIRECTORIES[0]}\n"


def test_main_diffs_the_capture_it_just_generated(
    _sim_world: FakeFileSystem, capsys: pytest.CaptureFixture[str], _argv: None
) -> None:
    """End to end: generate, then diff exactly what was generated.

    The sim side of the diff is served straight out of the fake file
    system, so nothing stands in for the script's own output — if the
    sessions archived nothing, or archived it somewhere else, the sim
    session count below is zero and the assertion fails.
    """
    real_exists = script_hooks.path_exists
    real_logging = script_hooks.setup_rich_logging

    def _live_only(path: Path) -> bool:
        """Report the live archive directories as present.

        Args:
            path: Directory the script checked.

        Returns:
            True for the two live directories.
        """
        return path in LIVE_DIRECTORIES

    def _list(directory: Path) -> list[Path]:
        """Serve one live capture, or the generated baseline captures.

        Args:
            directory: Directory being scanned.

        Returns:
            The capture paths inside it.
        """
        if directory in LIVE_DIRECTORIES:
            return [directory / "injected.capture_session.json"]
        return _captures(_sim_world, directory)

    def _read(path: Path) -> str:
        """Serve the live fixture, or the sim's own generated capture.

        Args:
            path: Capture path being read.

        Returns:
            The session JSON for that side.
        """
        if path.parent in LIVE_DIRECTORIES:
            return _LIVE_SESSION
        return _sim_world.read_text(path)

    script_hooks.path_exists = _live_only
    script_hooks.setup_rich_logging = _silent
    analysis_hooks.set_analysis_hooks(read_text_fn=_read, list_session_paths_fn=_list)
    sys.argv = ["build_sim_baseline", "--rounds", "4"]
    try:
        main()
    finally:
        script_hooks.path_exists = real_exists
        script_hooks.setup_rich_logging = real_logging
        analysis_hooks.reset_analysis_hooks()

    output = capsys.readouterr().out
    assert "building baseline " in output
    assert output.count(" rounds, ") == len(SCENARIOS)
    assert f"\nlive_sessions=2 live_windows=2\nsim_sessions={len(SCENARIOS)} " in output
    assert "MISSING LAWS (real server does it, sim never does):" in output
    assert "re-read this baseline later with:" in output


def test_module_runs_as_a_script(capsys: pytest.CaptureFixture[str], _argv: None) -> None:
    """The ``__main__`` entry point refuses the same missing archive."""
    real_exists = script_hooks.path_exists
    real_logging = script_hooks.setup_rich_logging
    script_hooks.path_exists = _absent
    script_hooks.setup_rich_logging = _silent
    sys.argv = ["build_sim_baseline"]
    sys.modules.pop("scripts.build_sim_baseline", None)
    try:
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("scripts.build_sim_baseline", run_name="__main__")
    finally:
        script_hooks.path_exists = real_exists
        script_hooks.setup_rich_logging = real_logging

    assert excinfo.value.code == 1
    assert capsys.readouterr().out == f"No such directory: {LIVE_DIRECTORIES[0]}\n"
