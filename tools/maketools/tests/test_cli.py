"""The command registry, its argument rules and the one error boundary."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.errors import AppError

from maketools.cli import COMMANDS, dispatch, main, repository_root, require_integer
from maketools.guard_run import GUARD_ARGV
from maketools.makefile_grammar import SHELL_INCLUDE
from maketools.venv_check import PROBE_ARGV
from tests.conftest import World, ok, row
from tests.test_makefile_grammar import PORTABLE


@pytest.fixture()
def cwd(tmp_path: Path) -> Generator[Path, None, None]:
    before = Path.cwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(before)


def test_every_command_is_registered() -> None:
    assert sorted(COMMANDS) == [
        "compose-down",
        "compose-up",
        "env",
        "fan-out",
        "guard",
        "hooks",
        "lint-makefiles",
        "native-wheel",
        "poetry-build",
        "reap-stale",
        "require-tool",
        "test",
        "uv-venv-check",
        "venv-check",
        "venv-exec",
    ]


def test_no_command_and_an_unknown_command_are_usage_errors(world: World) -> None:
    assert main([]) == 1
    assert world.errors[0].startswith("MAKETOOLS_USAGE: a command is required")
    assert main(["frobnicate"]) == 1
    assert world.errors[1].startswith("MAKETOOLS_USAGE: unknown command 'frobnicate'")


def test_venv_check_runs_in_the_cwd_and_refuses_arguments(world: World, cwd: Path) -> None:
    (cwd / ".venv").mkdir()
    world.capturing_answers[PROBE_ARGV] = ok("mypy 2.3.1")
    assert dispatch(["venv-check"]) == 0
    assert world.captured == [PROBE_ARGV]
    with pytest.raises(AppError, match=r"venv-check takes no arguments, got \['x'\]"):
        dispatch(["venv-check", "x"])


def test_guard_runs_in_the_cwd(world: World, cwd: Path) -> None:
    (cwd / "scripts").mkdir()
    (cwd / "scripts" / "guard.py").write_text("", encoding="utf-8")
    world.inheriting_code = 2
    assert dispatch(["guard"]) == 2
    assert world.inheriting_calls[0]["argv"] == GUARD_ARGV


def test_test_forwards_pytest_arguments_and_honours_no_sweep(world: World, cwd: Path) -> None:
    assert dispatch(["test", "--no-sweep", "-k", "x"]) == 0
    assert world.inheriting_calls[0]["argv"][-2:] == ("-k", "x")
    assert "-n" in world.inheriting_calls[0]["argv"]
    assert not any(line.startswith("reap: sweep") for line in world.lines)
    assert dispatch(["test"]) == 0
    assert any(line.startswith("reap: sweep") for line in world.lines)


def test_test_serial_and_venv_runner_flags(world: World, cwd: Path) -> None:
    (cwd / ".venv" / "bin").mkdir(parents=True)
    (cwd / ".venv" / "bin" / "python").write_bytes(b"")
    assert dispatch(["test", "--serial", "--runner", "venv", "--no-sweep"]) == 0
    argv = world.inheriting_calls[0]["argv"]
    assert argv[:3] == (str(cwd / ".venv" / "bin" / "python"), "-m", "pytest")
    assert "-n" not in argv
    assert "--max-worker-restart=0" not in argv
    assert dispatch(["test", "--runner", "poetry", "--no-sweep"]) == 0
    assert world.inheriting_calls[1]["argv"][:3] == ("poetry", "run", "pytest")
    with pytest.raises(AppError, match=r"--runner must be poetry or venv, got 'conda'"):
        dispatch(["test", "--runner", "conda"])
    with pytest.raises(AppError, match=r"--runner must be poetry or venv, got ''"):
        dispatch(["test", "--runner"])


def test_env_fan_out_compose_hooks_and_tools_reach_their_modules(world: World, cwd: Path) -> None:
    assert dispatch(["env", "A=1", "--", "poetry", "run", "x"]) == 0
    assert world.inheriting_calls[-1]["env"]["A"] == "1"
    (cwd / "libs" / "p").mkdir(parents=True)
    (cwd / "libs" / "p" / "Makefile").write_text("", encoding="utf-8")
    assert dispatch(["fan-out", "lint", "libs"]) == 0
    assert world.inheriting_calls[-1]["argv"] == ("make", "lint")
    with pytest.raises(AppError, match=r"fan-out needs a target and at least one parent"):
        dispatch(["fan-out", "lint"])
    assert dispatch(["compose-up", "libs/p"]) == 0
    assert world.inheriting_calls[-1]["cwd"] == cwd / "libs" / "p"
    world.capturing_answers[("git", "rev-parse", "HEAD")] = ok("sha\n")
    assert dispatch(["compose-up", "libs/p", "--git-commit", "--build-progress", "plain"]) == 0
    assert world.inheriting_calls[-1]["env"]["GIT_COMMIT"] == "sha"
    with pytest.raises(AppError, match=r"compose-up needs a service directory"):
        dispatch(["compose-up"])
    with pytest.raises(AppError, match=r"compose-up does not take '--bogus'"):
        dispatch(["compose-up", "libs/p", "--bogus"])
    (cwd / "libs" / "p" / "docker-compose.yml").write_text("", encoding="utf-8")
    assert dispatch(["compose-down", "libs/p"]) == 0
    assert list(world.inheriting_calls[-1]["argv"]) == ["docker", "compose", "down"]
    with pytest.raises(AppError, match=r"compose-down needs at least one directory"):
        dispatch(["compose-down"])
    assert dispatch(["hooks", "install"]) == 0
    world.capturing_answers[("git", "config", "--get", "core.hooksPath")] = ok(".githooks")
    assert dispatch(["hooks", "check"]) == 0
    with pytest.raises(AppError, match=r"hooks takes install or check"):
        dispatch(["hooks", "remove"])
    assert dispatch(["require-tool", "python", "install it"]) == 0
    with pytest.raises(AppError, match=r"require-tool needs NAME and HINT"):
        dispatch(["require-tool", "python"])


def test_uv_venv_exec_native_wheel_and_poetry_build_reach_their_modules(
    world: World, cwd: Path
) -> None:
    assert dispatch(["uv-venv-check"]) == 0
    assert world.inheriting_calls[0]["argv"] == ("uv", "venv")
    with pytest.raises(AppError, match=r"uv-venv-check takes no arguments"):
        dispatch(["uv-venv-check", "x"])
    (cwd / ".venv" / "bin").mkdir(parents=True)
    (cwd / ".venv" / "bin" / "ruff").write_bytes(b"")
    assert dispatch(["venv-exec", "ruff", "format", "."]) == 0
    assert world.inheriting_calls[-1]["argv"][1:] == ("format", ".")
    assert dispatch(["native-wheel", "--crate", "../crate", "--package", "cleargbm_rs"]) == 0
    assert world.lines[-1].startswith("native-wheel: no cleargbm_rs wheel built")
    with pytest.raises(AppError, match=r"native-wheel takes --crate DIR --package NAME"):
        dispatch(["native-wheel", "--crate", "x"])
    assert dispatch(["poetry-build", "libs/a"]) == 0
    assert world.inheriting_calls[-1]["argv"] == ("poetry", "build", "--quiet")


def test_reap_stale_takes_an_age_and_reports_a_failed_kill(world: World, cwd: Path) -> None:
    old = world.now_value - 11 * 60
    stale = row(5, 0, executable=f"/x/{cwd.name}/.venv/bin/python", created_unix=old)
    world.tables = [[stale], [stale]]
    world.refuse_kill = {5}
    world.alive = {5}
    assert dispatch(["reap-stale", "--older-than-minutes", "10"]) == 1
    assert world.killed == [5]
    assert dispatch(["reap-stale"]) == 0
    with pytest.raises(AppError, match=r"needs an integer, got 'ten'"):
        dispatch(["reap-stale", "--older-than-minutes", "ten"])
    with pytest.raises(AppError, match=r"reap-stale takes only --older-than-minutes N"):
        dispatch(["reap-stale", "--bogus"])


def test_require_integer_parses_digits_only() -> None:
    assert require_integer("--f", "12") == 12
    with pytest.raises(AppError, match=r"--f needs an integer, got '-1'"):
        require_integer("--f", "-1")


def test_the_repository_root_is_four_levels_above_the_cli_module() -> None:
    root = repository_root()
    assert (root / "tools" / "maketools" / "pyproject.toml").is_file()
    assert (root / SHELL_INCLUDE).is_file()


def test_lint_makefiles_reports_the_count_and_the_violations(world: World) -> None:
    root = repository_root()
    world.tracked = [Path("tools/maketools/Makefile")]
    assert dispatch(["lint-makefiles"]) == 0
    assert world.lines == [
        "lint-makefiles: 1 tracked Makefile(s) in the portable grammar, "
        "every one beginning with the shell prologue and every check printing the pass banner"
    ]
    bad = root / "tools" / "maketools" / "runs" / "bad" / "Makefile"
    bad.parent.mkdir(parents=True, exist_ok=True)
    # The appended line lands in check's recipe after the banner, so it
    # breaks the grammar (a cmdlet) AND the banner rule (a command after the
    # banner); the prologue at the wrong depth is the third.
    bad.write_text(PORTABLE + "\tWrite-Host x\n", encoding="utf-8")
    world.tracked = [bad.relative_to(root)]
    assert dispatch(["lint-makefiles"]) == 1
    assert world.errors[-1] == "lint-makefiles: 3 violation(s) in 1 tracked Makefile(s)"
    assert world.errors[0].endswith(
        "Makefile:1: first line must be the shell prologue: "
        "include ../../../../scripts/make/shell.mk"
    )
    assert world.errors[2].endswith(
        "Makefile:14: check: runs a command after the pass banner: Write-Host x"
    )
    bad.unlink()
    with pytest.raises(AppError, match=r"lint-makefiles takes no arguments"):
        dispatch(["lint-makefiles", "x"])
