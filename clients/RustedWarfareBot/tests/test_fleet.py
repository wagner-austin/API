"""The match fleet manager: registry, spawn parameters, transcripts.

The manager is exercised through injected process doubles; the real
spawn/kill implementations get their own live-child tests in
``test_fleet_http.py`` alongside the socket layer.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from pathlib import Path

import pytest

from rw_bot.harness import _test_hooks
from rw_bot.harness.fleet import FleetError, FleetManager, transcript_path


class FakeMatch:
    """Controllable child-process double."""

    def __init__(self, pid: int) -> None:
        """Create the double.

        Args:
            pid: Process id to report.
        """
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        """Return the scripted exit code.

        Returns:
            ``None`` while the fake is alive.
        """
        return self.returncode


class FakeSpawner:
    """Records spawn invocations and hands out process doubles."""

    def __init__(self) -> None:
        """Start with no spawns recorded."""
        self.argvs: list[list[str]] = []
        self.transcripts: list[Path] = []
        self.matches: list[FakeMatch] = []

    def __call__(self, argv: Sequence[str], transcript: Path) -> FakeMatch:
        """Record one spawn.

        Args:
            argv: Argument vector, program first.
            transcript: Transcript file the child would write.

        Returns:
            A fresh process double.
        """
        self.argvs.append(list(argv))
        self.transcripts.append(transcript)
        match = FakeMatch(pid=4001 + len(self.matches))
        self.matches.append(match)
        return match


def _swallow_line(text: str) -> None:
    """Drop one output line — tests assert on state, not chatter.

    Args:
        text: Ignored.
    """


@pytest.fixture()
def spawner() -> Generator[FakeSpawner, None, None]:
    """Install a recording spawner and a silent output line."""
    original_spawn = _test_hooks.spawn_match
    original_write = _test_hooks.write_line
    fake = FakeSpawner()
    _test_hooks.spawn_match = fake
    _test_hooks.write_line = _swallow_line
    yield fake
    _test_hooks.spawn_match = original_spawn
    _test_hooks.write_line = original_write


def _spawn_alpha(manager: FleetManager, **overrides: int | str) -> None:
    """Spawn the standard test instance.

    Args:
        manager: Manager under test.
        **overrides: Parameter overrides.
    """
    parameters: dict[str, int | str] = {
        "instance": "alpha",
        "seed": 7,
        "map_name": "",
        "opponents": 1,
        "difficulty": 2,
        "fastforward": 8,
        "tree": "",
    }
    parameters.update(overrides)
    manager.spawn(
        instance=str(parameters["instance"]),
        seed=int(parameters["seed"]),
        map_name=str(parameters["map_name"]),
        opponents=int(parameters["opponents"]),
        difficulty=int(parameters["difficulty"]),
        fastforward=int(parameters["fastforward"]),
        tree=str(parameters["tree"]),
    )


def test_spawn_composes_the_make_play_invocation(spawner: FakeSpawner) -> None:
    """Every knob rides the make command line; empty map/tree are omitted."""
    manager = FleetManager()
    _spawn_alpha(manager, map_name="maps/skirmish/x.tmx", tree="frozen/t1")

    assert spawner.argvs == [
        [
            "make",
            "play",
            "PLAY_SEED=7",
            "PLAY_OPPONENTS=1",
            "PLAY_DIFFICULTY=2",
            "PLAY_FASTFORWARD=8",
            "PLAY_LOG=runs/fleet/alpha.log",
            "PLAY_MAP=maps/skirmish/x.tmx",
            "PLAY_TREE=frozen/t1",
        ]
    ]
    assert spawner.transcripts == [Path("runs") / "fleet" / "alpha.out"]


def test_spawn_omits_empty_map_and_tree(spawner: FakeSpawner) -> None:
    """Blank map and tree keep the Makefile defaults, not empty overrides."""
    manager = FleetManager()
    _spawn_alpha(manager)

    argv = spawner.argvs[0]
    assert all(not part.startswith("PLAY_MAP") for part in argv)
    assert all(not part.startswith("PLAY_TREE") for part in argv)


def test_spawn_rejects_bad_names_and_negative_numbers(spawner: FakeSpawner) -> None:
    """Path characters, uppercase, and negatives never reach make."""
    manager = FleetManager()

    with pytest.raises(FleetError) as bad_name:
        _spawn_alpha(manager, instance="../escape")
    assert bad_name.value.code == "RW-FLEET-001"
    with pytest.raises(FleetError) as bad_seed:
        _spawn_alpha(manager, seed=-1)
    assert bad_seed.value.code == "RW-FLEET-002"
    assert spawner.argvs == []


def test_spawn_refuses_a_live_duplicate_but_replaces_a_dead_one(
    spawner: FakeSpawner,
) -> None:
    """One live match per instance; a finished one may be respawned."""
    manager = FleetManager()
    _spawn_alpha(manager)

    with pytest.raises(FleetError) as refused:
        _spawn_alpha(manager)
    assert refused.value.code == "RW-FLEET-003"

    spawner.matches[0].returncode = 0
    _spawn_alpha(manager)
    assert spawner.matches[1].pid == 4002


def test_report_sorts_and_reflects_liveness(spawner: FakeSpawner) -> None:
    """Rows come back sorted and track process exit."""
    manager = FleetManager()
    _spawn_alpha(manager, instance="bravo")
    _spawn_alpha(manager, instance="alpha")
    spawner.matches[0].returncode = 7

    rows = manager.report()

    assert [row["instance"] for row in rows] == ["alpha", "bravo"]
    assert rows[0]["alive"] is True
    assert rows[1]["alive"] is False
    assert rows[1]["returncode"] == 7
    assert rows[1]["fastforward"] == 8


def test_stats_reduces_the_transcript(spawner: FakeSpawner) -> None:
    """The verdict block is found and everything before it dropped."""
    lines: tuple[str, ...] = (
        "[play] fast-forward: 8x",
        "verdict        B (victory)",
        "plan           12/12 -- done: all built",
    )

    def fake_read(path: Path) -> tuple[str, ...]:
        return lines

    original_read = _test_hooks.read_text_lines
    _test_hooks.read_text_lines = fake_read
    try:
        manager = FleetManager()
        _spawn_alpha(manager)
        stats = manager.stats("alpha")
    finally:
        _test_hooks.read_text_lines = original_read

    assert stats["available"] is True
    assert stats["finished"] is True
    assert stats["verdict"] == "verdict        B (victory)"
    assert stats["report"] == list(lines[1:])


def test_stats_before_any_output_is_unavailable(spawner: FakeSpawner) -> None:
    """A missing transcript is a state, not an error; ghosts still 404."""

    def raise_missing(path: Path) -> tuple[str, ...]:
        raise OSError(f"no transcript at {path}")

    original_read = _test_hooks.read_text_lines
    _test_hooks.read_text_lines = raise_missing
    try:
        manager = FleetManager()
        _spawn_alpha(manager)
        stats = manager.stats("alpha")
        with pytest.raises(FleetError) as unknown:
            manager.stats("ghost")
    finally:
        _test_hooks.read_text_lines = original_read

    assert stats == {"available": False, "finished": False, "verdict": "", "report": []}
    assert unknown.value.code == "RW-FLEET-004"


def test_stats_running_match_is_available_but_unfinished(
    spawner: FakeSpawner,
) -> None:
    """Transcript lines without a verdict mean the match is mid-flight."""

    def fake_read(path: Path) -> tuple[str, ...]:
        return ("[play] launching game",)

    original_read = _test_hooks.read_text_lines
    _test_hooks.read_text_lines = fake_read
    try:
        manager = FleetManager()
        _spawn_alpha(manager)
        stats = manager.stats("alpha")
    finally:
        _test_hooks.read_text_lines = original_read

    assert stats["available"] is True
    assert stats["finished"] is False
    assert stats["report"] == []


def test_stop_kills_the_tree_and_refuses_finished_matches(
    spawner: FakeSpawner,
) -> None:
    """Stop fells the live tree by pid; a finished match cannot be stopped."""
    killed: list[int] = []

    def fake_kill(pid: int) -> None:
        killed.append(pid)

    original_kill = _test_hooks.kill_tree
    _test_hooks.kill_tree = fake_kill
    try:
        manager = FleetManager()
        _spawn_alpha(manager)
        manager.stop("alpha")
        spawner.matches[0].returncode = 1
        with pytest.raises(FleetError) as finished:
            manager.stop("alpha")
        with pytest.raises(FleetError) as unknown:
            manager.stop("ghost")
    finally:
        _test_hooks.kill_tree = original_kill

    assert killed == [4001]
    assert finished.value.code == "RW-FLEET-006"
    assert unknown.value.code == "RW-FLEET-004"


def test_restart_respawns_with_identical_parameters(spawner: FakeSpawner) -> None:
    """Restart replays the exact spawn; refuses while alive; 404 for ghosts."""
    manager = FleetManager()
    _spawn_alpha(manager, map_name="maps/skirmish/x.tmx", fastforward=4)

    with pytest.raises(FleetError) as alive:
        manager.restart("alpha")
    assert alive.value.code == "RW-FLEET-005"
    with pytest.raises(FleetError) as unknown:
        manager.restart("ghost")
    assert unknown.value.code == "RW-FLEET-004"

    spawner.matches[0].returncode = 0
    row = manager.restart("alpha")
    assert row["pid"] == 4002
    assert spawner.argvs[1] == spawner.argvs[0]


def test_remove_drops_only_finished_matches(spawner: FakeSpawner) -> None:
    """Remove refuses a live match and clears a finished one."""
    manager = FleetManager()
    _spawn_alpha(manager)

    with pytest.raises(FleetError) as alive:
        manager.remove("alpha")
    assert alive.value.code == "RW-FLEET-005"

    spawner.matches[0].returncode = 0
    row = manager.remove("alpha")
    assert row["alive"] is False
    assert manager.report() == []
    with pytest.raises(FleetError) as unknown:
        manager.remove("alpha")
    assert unknown.value.code == "RW-FLEET-004"


def test_transcript_path_is_the_instance_namespace() -> None:
    """Transcripts live under runs/fleet, one per instance."""
    assert transcript_path("alpha") == Path("runs") / "fleet" / "alpha.out"
