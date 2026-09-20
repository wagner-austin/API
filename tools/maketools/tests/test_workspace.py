"""The root Makefile's commands: fan-out, compose, hooks, require-tool."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError

from maketools.workspace import (
    compose_down,
    compose_up,
    fan_out,
    hooks_check,
    hooks_install,
    makefile_directories,
    require_tool,
)
from tests.conftest import World, failed, ok


def package(root: Path, name: str, *, makefile: bool = True) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    if makefile:
        (directory / "Makefile").write_text("check:\n", encoding="utf-8")
    return directory


def test_makefile_directories_lists_only_packages_with_a_makefile(tmp_path: Path) -> None:
    package(tmp_path / "libs", "b")
    package(tmp_path / "libs", "a")
    package(tmp_path / "libs", "no-makefile", makefile=False)
    (tmp_path / "libs" / "file").write_text("", encoding="utf-8")
    package(tmp_path / "tools", "t")
    found = makefile_directories([tmp_path / "libs", tmp_path / "tools"])
    assert [p.relative_to(tmp_path).as_posix() for p in found] == ["libs/a", "libs/b", "tools/t"]


def test_fan_out_runs_every_package_and_names_the_failures(world: World, tmp_path: Path) -> None:
    package(tmp_path / "libs", "a")
    package(tmp_path / "libs", "b")
    package(tmp_path / "tools", "c")
    world.inheriting_codes = [0, 2, 0]
    assert fan_out("check", [Path("libs"), Path("tools")], cwd=tmp_path) == 1
    assert [c["cwd"].name for c in world.inheriting_calls] == ["a", "b", "c"]
    assert all(c["argv"] == ("make", "check") for c in world.inheriting_calls)
    assert world.errors == ["\nfan-out: make check failed in 1 of 3 package(s): libs/b"]


def test_fan_out_reports_a_clean_run_and_refuses_an_empty_tree(
    world: World, tmp_path: Path
) -> None:
    package(tmp_path / "libs", "a")
    assert fan_out("lint", [Path("libs")], cwd=tmp_path) == 0
    assert world.lines[-1] == "\nfan-out: make lint passed in all 1 package(s)"
    (tmp_path / "empty").mkdir()
    assert fan_out("lint", [Path("empty")], cwd=tmp_path) == 1
    assert world.errors[-1].startswith("fan-out: no directory under")


def test_compose_up_builds_inline_by_default(world: World, tmp_path: Path) -> None:
    assert compose_up(tmp_path, build_progress="", git_commit=False) == 0
    assert world.inheriting_calls[0]["argv"] == ("docker", "compose", "up", "-d", "--build")
    assert "GIT_COMMIT" not in world.inheriting_calls[0]["env"]


def test_compose_up_can_export_the_commit_and_build_first(world: World, tmp_path: Path) -> None:
    world.capturing_answers[("git", "rev-parse", "HEAD")] = ok("abc123\n")
    assert compose_up(tmp_path, build_progress="plain", git_commit=True) == 0
    argvs = [c["argv"] for c in world.inheriting_calls]
    assert argvs == [
        ("docker", "compose", "build", "--progress", "plain"),
        ("docker", "compose", "up", "-d"),
    ]
    assert world.inheriting_calls[0]["env"]["GIT_COMMIT"] == "abc123"


def test_compose_up_stops_at_a_failed_build_or_an_unreadable_commit(
    world: World, tmp_path: Path
) -> None:
    world.inheriting_code = 4
    assert compose_up(tmp_path, build_progress="plain", git_commit=False) == 4
    assert len(world.inheriting_calls) == 1
    world.capturing_answers[("git", "rev-parse", "HEAD")] = failed(128, "not a git repository")
    assert compose_up(tmp_path, build_progress="", git_commit=True) == 1
    assert world.errors == ["compose-up: git rev-parse HEAD failed: not a git repository"]


def test_compose_down_skips_directories_without_a_compose_file(
    world: World, tmp_path: Path
) -> None:
    with_file = package(tmp_path, "a", makefile=False)
    (with_file / "docker-compose.yml").write_text("", encoding="utf-8")
    without = package(tmp_path, "b", makefile=False)
    assert compose_down([with_file, without]) == 0
    assert [c["cwd"] for c in world.inheriting_calls] == [with_file]
    assert world.lines == [f"compose-down: {without.as_posix()} has no docker-compose.yml"]
    world.inheriting_code = 1
    assert compose_down([with_file]) == 1


def test_hooks_install_and_check(world: World, tmp_path: Path) -> None:
    assert hooks_install(tmp_path) == 0
    assert world.inheriting_calls[0]["argv"] == ("git", "config", "core.hooksPath", ".githooks")
    assert world.lines == ["core.hooksPath = .githooks"]
    world.inheriting_code = 1
    assert hooks_install(tmp_path) == 1
    world.capturing_answers[("git", "config", "--get", "core.hooksPath")] = ok(".githooks\n")
    assert hooks_check(tmp_path) == 0
    world.capturing_answers[("git", "config", "--get", "core.hooksPath")] = failed(1, "")
    assert hooks_check(tmp_path) == 1
    assert world.errors[-1].startswith("hooks NOT installed.")


def test_require_tool_finds_the_interpreter_and_refuses_a_missing_one(world: World) -> None:
    require_tool("python", "install python")
    assert world.lines == ["require-tool: python found"]
    with pytest.raises(AppError) as caught:
        require_tool("no-such-tool-xyz", "pipx install it")
    assert caught.value.code is MaketoolsErrorCode.TOOL_MISSING
    assert caught.value.message == "no-such-tool-xyz not found. pipx install it"
