"""The launcher: job on Windows, session elsewhere, sweep, token, caps, teardown."""

from __future__ import annotations

from pathlib import Path

from maketools.test_run import (
    NO_WORKER_RESTART,
    PYTEST_ARGV,
    coverage_arguments,
    remove_run_files,
    run_tests,
)
from tests.conftest import FakeJobApi, World, row


def test_coverage_arguments_name_only_the_roots_that_exist(tmp_path: Path) -> None:
    assert coverage_arguments(tmp_path) == ["--cov-branch", "--cov-report=term-missing"]
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    assert coverage_arguments(tmp_path)[2:] == ["--cov=src", "--cov=scripts"]


def test_remove_run_files_takes_this_token_and_leaves_the_others(
    world: World, tmp_path: Path
) -> None:
    for name in (".coverage-aaaa", ".coverage-aaaa.host.1.x", ".coverage-bbbb"):
        (tmp_path / name).write_bytes(b"")
    assert remove_run_files(tmp_path, ".coverage-aaaa") == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == [".coverage-bbbb"]


def test_a_posix_run_starts_its_own_session_sweeps_caps_and_reaps(
    world: World, tmp_path: Path
) -> None:
    (tmp_path / "src").mkdir()
    world.inheriting_code = 3
    world.tables = [[row(1, 0)], [row(4242, 1), row(50, 4242, "poetry", "poetry run pytest")]]
    code = run_tests(tmp_path, ["-k", "fast"], sweep=True)
    assert code == 3
    call = world.inheriting_calls[0]
    assert call["argv"] == (
        *PYTEST_ARGV,
        "-n",
        "auto",
        NO_WORKER_RESTART,
        "-v",
        "--cov-branch",
        "--cov-report=term-missing",
        "--cov=src",
        "-k",
        "fast",
    )
    assert call["cwd"] == tmp_path
    assert call["new_session"] is True
    assert call["env"]["COVERAGE_FILE"] == str((tmp_path / "runs").resolve() / ".coverage-deadbeef")
    assert call["env"]["OMP_NUM_THREADS"] == "1"
    assert call["env"]["MKL_NUM_THREADS"] == "1"
    assert call["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert call["env"]["PATH"] == "/bin"
    assert (tmp_path / "runs").is_dir()
    assert world.lines[0] == "run-tests: the suite runs in its own session and is reaped on exit"
    assert world.lines[1].startswith("reap: sweep stale (project '")
    assert world.killed == [50]
    assert world.job.calls == []


def test_a_windows_run_joins_the_job_and_does_not_ask_for_a_session(
    world: World, tmp_path: Path
) -> None:
    world.platform_name = "win32"
    code = run_tests(tmp_path, [], sweep=False)
    assert code == 0
    assert world.job.calls == ["create", "set", "assign"]
    assert world.inheriting_calls[0]["new_session"] is False
    assert (
        world.lines[0] == "run-tests: joined kill-on-close job object (tree dies with this process)"
    )
    assert not any(line.startswith("reap: sweep") for line in world.lines)


def test_a_windows_run_refuses_when_the_job_cannot_be_joined(world: World, tmp_path: Path) -> None:
    world.platform_name = "win32"
    world.job = FakeJobApi(handle=0)
    assert run_tests(tmp_path, [], sweep=False) == 1
    assert world.inheriting_calls == []
    assert world.errors == [
        "run-tests: could not join a kill-on-close job object: CreateJobObject returned NULL"
    ]


def test_this_runs_coverage_files_are_removed_after_the_suite(world: World, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / ".coverage-deadbeef").write_bytes(b"")
    (runs / ".coverage-deadbeef.h.1.z").write_bytes(b"")
    (runs / ".coverage-other").write_bytes(b"")
    run_tests(tmp_path, [], sweep=False)
    assert sorted(p.name for p in runs.iterdir()) == [".coverage-other"]
    assert len(world.removed_files) == 2
