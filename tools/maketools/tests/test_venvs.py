"""uv venvs, venv executables, native wheels and first-party builds."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError

from maketools.venvs import (
    UV_INSTALL_ARGV,
    installed_dist_info,
    native_wheel,
    newest_wheel,
    poetry_build,
    uv_venv_check,
    venv_exec,
    venv_executable,
)
from tests.conftest import World, failed, ok


def make_venv(project: Path, layout: str, *names: str) -> None:
    directory = project / ".venv" / layout
    directory.mkdir(parents=True)
    for name in names:
        (directory / name).write_bytes(b"")


def test_venv_executable_finds_either_layout_with_or_without_exe(tmp_path: Path) -> None:
    make_venv(tmp_path, "Scripts", "ruff.exe", "python.exe")
    assert venv_executable(tmp_path, "ruff") == tmp_path / ".venv" / "Scripts" / "ruff.exe"
    posix = tmp_path / "posix"
    make_venv(posix, "bin", "ruff")
    assert venv_executable(posix, "ruff") == posix / ".venv" / "bin" / "ruff"


def test_venv_executable_refuses_a_missing_one(tmp_path: Path) -> None:
    make_venv(tmp_path, "bin", "python")
    with pytest.raises(AppError) as caught:
        venv_executable(tmp_path, "mypy")
    assert caught.value.code is MaketoolsErrorCode.ARTIFACT_MISSING
    assert caught.value.message.endswith("has no mypy under Scripts or bin")


def test_venv_exec_runs_the_resolved_executable(world: World, tmp_path: Path) -> None:
    make_venv(tmp_path, "bin", "ruff")
    world.inheriting_code = 3
    assert venv_exec(tmp_path, ["ruff", "check", "."]) == 3
    call = world.inheriting_calls[0]
    assert call["argv"] == (str(tmp_path / ".venv" / "bin" / "ruff"), "check", ".")
    assert call["cwd"] == tmp_path
    with pytest.raises(AppError, match=r"venv-exec needs an executable name"):
        venv_exec(tmp_path, [])


def test_uv_venv_check_creates_a_missing_venv(world: World, tmp_path: Path) -> None:
    assert uv_venv_check(tmp_path) == 0
    assert [c["argv"] for c in world.inheriting_calls] == [("uv", "venv"), UV_INSTALL_ARGV]
    assert world.lines[0] == "uv-venv-check: no .venv yet; creating one"


def test_uv_venv_check_keeps_an_answering_venv(world: World, tmp_path: Path) -> None:
    make_venv(tmp_path, "bin", "python")
    python = str(tmp_path / ".venv" / "bin" / "python")
    world.capturing_answers[(python, "-m", "mypy", "--version")] = ok("mypy 2.3.1")
    assert uv_venv_check(tmp_path) == 0
    assert world.inheriting_calls == []
    assert world.lines == ["uv-venv-check: .venv answers (mypy 2.3.1)"]


def test_uv_venv_check_recreates_a_stale_venv_and_stops_at_a_failed_uv(
    world: World, tmp_path: Path
) -> None:
    make_venv(tmp_path, "bin", "python")
    python = str(tmp_path / ".venv" / "bin" / "python")
    world.capturing_answers[(python, "-m", "mypy", "--version")] = failed(1, "no mypy")
    world.inheriting_codes = [7]
    assert uv_venv_check(tmp_path) == 7
    assert world.removed_trees == [tmp_path / ".venv"]
    assert [c["argv"] for c in world.inheriting_calls] == [("uv", "venv")]


def touch(path: Path, when: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(b"")
    os.utime(path, (when, when))


def test_newest_wheel_and_installed_dist_info(tmp_path: Path) -> None:
    crate = tmp_path / "crate"
    assert newest_wheel(crate) is None
    now = time.time()
    touch(crate / "target" / "wheels" / "old.whl", now - 100)
    touch(crate / "target" / "wheels" / "new.whl", now)
    assert newest_wheel(crate) == crate / "target" / "wheels" / "new.whl"
    assert installed_dist_info(tmp_path, "cleargbm_rs") is None
    info = (
        tmp_path / ".venv" / "lib" / "python3.11" / "site-packages" / "cleargbm_rs-0.1.0.dist-info"
    )
    info.mkdir(parents=True)
    assert installed_dist_info(tmp_path, "cleargbm_rs") == info


def test_native_wheel_skips_reinstalls_and_refreshes(world: World, tmp_path: Path) -> None:
    crate = Path("../crate")
    assert native_wheel(tmp_path / "proj", crate=crate, package="cleargbm_rs") == 0
    assert world.lines[-1] == "native-wheel: no cleargbm_rs wheel built under ../crate; skipping"
    now = time.time()
    wheel = tmp_path / "crate" / "target" / "wheels" / "cleargbm_rs-0.1.0.whl"
    touch(wheel, now - 50)
    project = tmp_path / "proj"
    info = project / ".venv" / "Lib" / "site-packages" / "cleargbm_rs-0.1.0.dist-info"
    info.mkdir(parents=True)
    os.utime(info, (now, now))
    assert native_wheel(project, crate=crate, package="cleargbm_rs") == 0
    assert world.lines[-1] == "native-wheel: cleargbm_rs up to date"
    os.utime(info, (now - 100, now - 100))
    world.inheriting_code = 0
    assert native_wheel(project, crate=crate, package="cleargbm_rs") == 0
    call = world.inheriting_calls[0]
    assert call["argv"] == (
        "poetry",
        "run",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        str(wheel.resolve()),
    )
    assert world.lines[-1] == "native-wheel: reinstalling cleargbm_rs from cleargbm_rs-0.1.0.whl"


def test_poetry_build_runs_each_package_and_stops_at_the_first_failure(
    world: World, tmp_path: Path
) -> None:
    world.inheriting_codes = [0, 2, 0]
    assert poetry_build(tmp_path, [Path("a"), Path("b"), Path("c")]) == 2
    assert [c["cwd"] for c in world.inheriting_calls] == [tmp_path / "a", tmp_path / "b"]
    assert all(c["argv"] == ("poetry", "build", "--quiet") for c in world.inheriting_calls)
    assert poetry_build(tmp_path, [Path("a")]) == 0
    with pytest.raises(AppError, match=r"poetry-build needs at least one package"):
        poetry_build(tmp_path, [])
