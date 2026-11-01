"""The Linux launcher: new session, sweep, token, caps and teardown."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError

from maketools.commands import CommandResult
from maketools.test_run import (
    HUB_OFFLINE,
    NO_WORKER_RESTART,
    POETRY_EXECUTABLE_ARGV,
    PYTEST_ARGV,
    Runner,
    coverage_arguments,
    remove_run_files,
    run_tests,
)
from tests.conftest import World, failed, ok, row


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
    caches = (tmp_path / "runs").resolve() / ".caches-deadbeef"
    assert call["env"]["HF_HOME"] == str(caches / "huggingface")
    assert call["env"]["TRITON_HOME"] == str(caches / "triton")
    assert caches.is_dir()
    assert call["env"]["PATH"] == "/bin"
    assert (tmp_path / "runs").is_dir()
    assert world.lines[0] == "run-tests: the suite runs in its own session and is reaped on exit"
    assert world.lines[1].startswith("reap: sweep stale (project '")
    assert world.killed == [50]


def test_this_runs_coverage_files_are_removed_after_the_suite(world: World, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / ".coverage-deadbeef").write_bytes(b"")
    (runs / ".coverage-deadbeef.h.1.z").write_bytes(b"")
    (runs / ".coverage-other").write_bytes(b"")
    run_tests(tmp_path, [], sweep=False)
    assert sorted(p.name for p in runs.iterdir()) == [".caches-deadbeef", ".coverage-other"]
    assert len(world.removed_files) == 2
    assert world.removed_trees == [runs.resolve() / ".caches-deadbeef"]
    assert "HOME" not in world.inheriting_calls[0]["env"]
    assert HUB_OFFLINE[0] not in world.inheriting_calls[0]["env"]


def test_a_hub_offline_run_sets_the_variable_the_hub_clients_read(
    world: World, tmp_path: Path
) -> None:
    assert run_tests(tmp_path, [], sweep=False, hub_offline=True) == 0
    assert HUB_OFFLINE == ("HF_HUB_OFFLINE", "1")
    assert world.inheriting_calls[0]["env"]["HF_HUB_OFFLINE"] == "1"
    assert "HOME" not in world.inheriting_calls[0]["env"]


def _own_home(tmp_path: Path) -> Path:
    """The ``HOME`` a run under the World's token is given.

    Args:
        tmp_path: The project.

    Returns:
        Its path, which a case may fill before the run to stand for writes.
    """
    return (tmp_path / "runs").resolve() / ".home-deadbeef"


def test_an_own_home_run_reaches_pytest_through_poetrys_interpreter(
    world: World, tmp_path: Path
) -> None:
    world.capturing_answers[tuple(POETRY_EXECUTABLE_ARGV)] = ok("/v/bin/python\n")
    assert run_tests(tmp_path, ["-k", "x"], sweep=False, serial=True, own_home=True) == 0
    call = world.inheriting_calls[0]
    assert call["argv"][:4] == ("/v/bin/python", "-m", "pytest", "-v")
    assert call["env"]["HOME"] == str(_own_home(tmp_path))
    assert _own_home(tmp_path).is_dir()
    assert world.captured == [tuple(POETRY_EXECUTABLE_ARGV)]
    assert world.removed_trees == [
        (tmp_path / "runs").resolve() / ".caches-deadbeef",
        _own_home(tmp_path),
    ]
    assert world.errors == []


def test_a_file_left_under_the_runs_home_fails_a_passing_suite(
    world: World, tmp_path: Path
) -> None:
    world.capturing_answers[tuple(POETRY_EXECUTABLE_ARGV)] = ok("/v/bin/python\n")
    home = _own_home(tmp_path)
    (home / ".cache" / "torch").mkdir(parents=True)
    (home / ".cache" / "torch" / "kernels.bin").write_bytes(b"")
    (home / ".config").mkdir()
    assert run_tests(tmp_path, [], sweep=False, own_home=True) == 1
    assert world.errors == [
        f"run-tests: the suite wrote .cache/torch/kernels.bin under its own HOME, {home}"
    ]


def test_a_failing_suite_keeps_its_own_status_beside_a_home_write(
    world: World, tmp_path: Path
) -> None:
    world.capturing_answers[tuple(POETRY_EXECUTABLE_ARGV)] = ok("/v/bin/python\n")
    world.inheriting_code = 2
    home = _own_home(tmp_path)
    home.mkdir(parents=True)
    (home / ".netrc").write_bytes(b"")
    assert run_tests(tmp_path, [], sweep=False, own_home=True) == 2
    assert world.errors == [f"run-tests: the suite wrote .netrc under its own HOME, {home}"]


@pytest.mark.parametrize(
    ("answer", "named"),
    [(failed(1, "no venv\n"), "no venv"), (ok("\n"), "")],
)
def test_poetry_naming_no_interpreter_refuses_before_the_run(
    world: World, tmp_path: Path, answer: CommandResult, named: str
) -> None:
    world.capturing_answers[tuple(POETRY_EXECUTABLE_ARGV)] = answer
    with pytest.raises(AppError, match=f"poetry names no venv interpreter for .*: {named}$"):
        run_tests(tmp_path, [], sweep=False, own_home=True)
    assert world.inheriting_calls == []
    assert not (tmp_path / "runs").exists()


def test_an_own_home_venv_run_asks_poetry_nothing(world: World, tmp_path: Path) -> None:
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_bytes(b"")
    run_tests(tmp_path, [], sweep=False, runner=Runner.VENV, own_home=True)
    assert world.inheriting_calls[0]["argv"][:3] == (
        str(tmp_path / ".venv" / "bin" / "python"),
        "-m",
        "pytest",
    )
    assert world.captured == []
    assert world.inheriting_calls[0]["env"]["HOME"] == str(_own_home(tmp_path))
