"""The reaper's tree walks, its gates and its kill accounting."""

from __future__ import annotations

from pathlib import Path

from maketools.reap import (
    MAX_DEPTH,
    ancestor_depth,
    descendants,
    in_project,
    is_test_process,
    kill_targets,
    reap_descendants,
    sweep_stale,
)
from tests.conftest import World, row


def test_is_test_process_matches_the_three_shapes_and_nothing_else() -> None:
    assert is_test_process(row(1, 0, "poetry.exe", "poetry run pytest -n auto"))
    assert is_test_process(row(1, 0, "pytest", "pytest tests"))
    assert is_test_process(row(1, 0, "python3.11", 'python -u -c "import sys;exec(eval(...))"'))
    assert not is_test_process(row(1, 0, "python.exe", "python -m http.server"))
    assert not is_test_process(row(1, 0, "node.exe", "node pytest-lookalike.js"))
    assert not is_test_process(row(1, 0, "python.exe", ""))


def test_descendants_are_listed_deepest_first_and_bounded() -> None:
    rows = [row(1, 0), row(2, 1), row(3, 2), row(4, 2), row(9, 8)]
    assert [r["pid"] for r in descendants(rows, 1)] == [3, 4, 2]
    assert descendants(rows, 1, depth=MAX_DEPTH + 1) == []
    # A row that names itself as its own parent is not its own descendant.
    assert descendants([row(5, 5)], 5) == []


def test_ancestor_depth_counts_the_chain_and_caps() -> None:
    rows = [row(1, 0), row(2, 1), row(3, 2)]
    by_pid = {r["pid"]: r for r in rows}
    assert ancestor_depth(by_pid, by_pid[3]) == 3
    cycle = {1: row(1, 2), 2: row(2, 1)}
    assert ancestor_depth(cycle, cycle[1]) == MAX_DEPTH


def test_in_project_resolves_membership_through_the_ancestry() -> None:
    launcher = row(1, 0, "poetry.exe", "poetry run pytest", executable="C:/x/procart/.venv/p.exe")
    worker = row(2, 1, "python.exe", 'python -u -c "exec(eval(sys.stdin.readline()))"')
    stranger = row(3, 0, "python.exe", "python -m pytest")
    by_pid = {1: launcher, 2: worker, 3: stranger}
    assert in_project(by_pid, worker, "procart")
    assert in_project(by_pid, launcher, "procart")
    assert not in_project(by_pid, stranger, "procart")
    cycle = {1: row(1, 2), 2: row(2, 1)}
    assert not in_project(cycle, cycle[1], "procart")


def test_kill_targets_counts_gone_as_killed_and_alive_refusals_as_failed(world: World) -> None:
    world.refuse_kill = {2, 3}
    world.alive = {3}
    report = kill_targets([row(1, 0, commit_mb=10), row(2, 0, commit_mb=20), row(3, 0)])
    assert report == {"targets": 3, "killed": 2, "failed": 1, "commit_mb": 130.0}
    assert world.killed == [1, 2, 3]
    assert world.errors == ["reap: could not kill pid=3 python.exe: refused to kill 3"]
    assert world.lines == ["reap: killed 2, failed 1, ~130 MB commit reclaimed"]


def test_reap_descendants_kills_only_test_processes_under_the_root(world: World) -> None:
    world.tables = [
        [
            row(4242, 1),
            row(10, 4242, "poetry.exe", "poetry run pytest"),
            row(11, 10),
            row(12, 10, "node.exe", "node"),
        ]
    ]
    report = reap_descendants(4242)
    assert report["targets"] == 2
    assert world.killed == [11, 10]


def test_reap_descendants_with_nothing_under_the_root_says_so(world: World) -> None:
    world.tables = [[row(1, 0)]]
    assert reap_descendants(4242)["targets"] == 0
    assert world.lines == ["reap: descendants of pid 4242", "reap: nothing to reap."]
    assert world.killed == []


def test_sweep_with_nothing_stale_reports_and_touches_nothing(world: World, tmp_path: Path) -> None:
    world.tables = [[row(1, 0, created_unix=world.now_value)]]
    report = sweep_stale(tmp_path / "procart")
    assert report["targets"] == 0
    assert world.lines[-1] == "reap: nothing stale."
    assert world.slept == []


def test_sweep_aborts_when_the_candidate_set_is_burning_cpu(world: World, tmp_path: Path) -> None:
    old = world.now_value - 61 * 60
    before = [
        row(5, 0, executable="/x/procart/.venv/bin/python", created_unix=old, cpu_seconds=1.0)
    ]
    after = [row(5, 0, executable="/x/procart/.venv/bin/python", created_unix=old, cpu_seconds=2.0)]
    world.tables = [before, after]
    report = sweep_stale(tmp_path / "procart")
    assert report["targets"] == 0
    assert world.slept == [5]
    assert "CPU delta over 5s = 1.000s" in world.lines[-2]
    assert world.lines[-1].startswith("reap: ABORTING")
    assert world.killed == []


def test_sweep_reaps_an_idle_stale_set_deepest_first(world: World, tmp_path: Path) -> None:
    old = world.now_value - 61 * 60
    launcher = row(
        5,
        1,
        "poetry",
        "poetry run pytest",
        executable="/x/procart/.venv/bin/python",
        created_unix=old,
    )
    controller = row(6, 5, "python3", "python -m pytest", created_unix=old)
    worker = row(7, 6, "python3", "python -u -c exec(eval(...))", created_unix=old, cpu_seconds=0.5)
    fresh = row(8, 6, "python3", "python -u -c exec(eval(...))", created_unix=world.now_value)
    world.tables = [
        [launcher, controller, worker, fresh],
        [launcher, controller, {**worker, "cpu_seconds": 0.55}, fresh],
    ]
    report = sweep_stale(tmp_path / "procart", older_than_minutes=60)
    assert report["targets"] == 3
    assert world.killed == [7, 6, 5]
    assert world.lines[-2] == "reap: 3 process(es) holding 300 MB commit"


def test_sweep_ignores_a_candidate_that_vanished_between_samples(
    world: World, tmp_path: Path
) -> None:
    old = world.now_value - 61 * 60
    stale = row(5, 0, executable="/x/procart/.venv/bin/python", created_unix=old)
    world.tables = [[stale], []]
    report = sweep_stale(tmp_path / "procart")
    assert "CPU delta over 5s = 0.000s" in world.lines[-3]
    assert report["killed"] == 1
