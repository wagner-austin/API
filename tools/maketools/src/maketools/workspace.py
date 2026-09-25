"""The root and fan-out Makefiles' commands: fan-out, compose, hooks, tools.

Each was a one-line PowerShell program in a recipe: a ``foreach`` over
subdirectories running ``make``, a ``Set-Location`` into a service before
``docker compose``, a ``git config`` read with a coloured verdict, a
``Get-Command`` probe. None of them has a portable spelling, so they are
commands here and the recipes name them.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError

from maketools import _test_hooks

#: The compose file a service directory is expected to carry.
COMPOSE_FILE: Final[str] = "docker-compose.yml"

#: The one value ``core.hooksPath`` may hold here.
HOOKS_PATH: Final[str] = ".githooks"

#: Wall clock on one fanned-out make target. Four hours, because what this
#: runs IS a package's whole lint or test, so it inherits the suite bound
#: rather than inventing a smaller one; a fan-out over libs, services,
#: clients and tools is the longest thing this package ever starts.
FANOUT_WALL_SECONDS: Final[int] = 14400

#: Wall clock on a docker image build. An hour: a cold build of the ML
#: images pulls a CUDA base and compiles wheels, which is tens of minutes on
#: this uplink, and a build still going after an hour is waiting on a
#: registry that is not answering.
COMPOSE_BUILD_WALL_SECONDS: Final[int] = 3600

#: Wall clock on bringing containers up or down, and on a git config write.
#: Ten minutes: these are local control-plane calls that take seconds, and
#: the one that hangs is hanging on a wedged daemon rather than working.
CONTROL_WALL_SECONDS: Final[int] = 600


def makefile_directories(parents: Sequence[Path]) -> list[Path]:
    """Every immediate subdirectory of the parents that carries a Makefile.

    Args:
        parents: The directories to look under, in order.

    Returns:
        The package directories, sorted within each parent.
    """
    found: list[Path] = []
    for parent in parents:
        for child in sorted(parent.iterdir()):
            if child.is_dir() and (child / "Makefile").is_file():
                found.append(child)
    return found


def fan_out(target: str, parents: Sequence[Path], *, cwd: Path) -> int:
    """Run ``make <target>`` in every package under the parents.

    Every package runs even after one fails, and the failures are named at
    the end: a fan-out that stopped at the first red package would hide the
    others and send the operator around the loop once per package.

    Args:
        target: The make target.
        parents: The directories whose children are packages.
        cwd: The recipe's directory, which relative parents resolve against.

    Returns:
        1 when any package failed, else 0.
    """
    packages = makefile_directories([cwd / parent for parent in parents])
    if not packages:
        _test_hooks.write_error(
            f"fan-out: no directory under {[str(p) for p in parents]} carries a Makefile"
        )
        return 1
    failed: list[str] = []
    for package in packages:
        _test_hooks.write_line(f"\n=== make {target} in {package.relative_to(cwd).as_posix()} ===")
        code = _test_hooks.run_inheriting(
            ["make", target],
            cwd=package,
            env=_test_hooks.environ(),
            new_session=False,
            timeout_seconds=FANOUT_WALL_SECONDS,
        )
        if code != 0:
            failed.append(package.relative_to(cwd).as_posix())
    if failed:
        _test_hooks.write_error(
            f"\nfan-out: make {target} failed in {len(failed)} of {len(packages)} package(s): "
            + ", ".join(failed)
        )
        return 1
    _test_hooks.write_line(f"\nfan-out: make {target} passed in all {len(packages)} package(s)")
    return 0


def compose_up(directory: Path, *, build_progress: str, git_commit: bool) -> int:
    """``docker compose up -d --build`` for a service directory.

    Args:
        directory: The service, carrying its own compose file.
        build_progress: Compose's ``--progress`` for a separate build step,
            or empty to let ``up --build`` build inline.
        git_commit: Export ``GIT_COMMIT`` from ``git rev-parse HEAD`` so the
            Dockerfile bakes the commit every manifest names; a build that
            forgets it records null, which every manifest archived by the
            2026-08-18 audit did.

    Returns:
        The first non-zero status, or 0.
    """
    environment = _test_hooks.environ()
    if git_commit:
        result = _test_hooks.run_capturing(["git", "rev-parse", "HEAD"], cwd=directory)
        if result["returncode"] != 0:
            _test_hooks.write_error(f"compose-up: git rev-parse HEAD failed: {result['stderr']}")
            return 1
        environment["GIT_COMMIT"] = result["stdout"].strip()
    if build_progress != "":
        code = _test_hooks.run_inheriting(
            ["docker", "compose", "build", "--progress", build_progress],
            cwd=directory,
            env=environment,
            new_session=False,
            timeout_seconds=COMPOSE_BUILD_WALL_SECONDS,
        )
        if code != 0:
            return code
        return _test_hooks.run_inheriting(
            ["docker", "compose", "up", "-d"],
            cwd=directory,
            env=environment,
            new_session=False,
            timeout_seconds=CONTROL_WALL_SECONDS,
        )
    return _test_hooks.run_inheriting(
        ["docker", "compose", "up", "-d", "--build"],
        cwd=directory,
        env=environment,
        new_session=False,
        timeout_seconds=COMPOSE_BUILD_WALL_SECONDS,
    )


def compose_down(directories: Sequence[Path]) -> int:
    """``docker compose down`` in every directory that carries a compose file.

    A directory without one is reported and skipped rather than failed: the
    root's ``down`` names every service, and a service that has not been
    dockerised yet is not a reason to leave the others running.

    Args:
        directories: The service directories.

    Returns:
        The first non-zero status, or 0.
    """
    for directory in directories:
        if not (directory / COMPOSE_FILE).is_file():
            _test_hooks.write_line(f"compose-down: {directory.as_posix()} has no {COMPOSE_FILE}")
            continue
        code = _test_hooks.run_inheriting(
            ["docker", "compose", "down"],
            cwd=directory,
            env=_test_hooks.environ(),
            new_session=False,
            timeout_seconds=CONTROL_WALL_SECONDS,
        )
        if code != 0:
            return code
    return 0


def hooks_install(cwd: Path) -> int:
    """Point this clone at the versioned hooks directory.

    Args:
        cwd: The repository root.

    Returns:
        git's status.
    """
    code = _test_hooks.run_inheriting(
        ["git", "config", "core.hooksPath", HOOKS_PATH],
        cwd=cwd,
        env=_test_hooks.environ(),
        new_session=False,
        timeout_seconds=CONTROL_WALL_SECONDS,
    )
    if code == 0:
        _test_hooks.write_line(f"core.hooksPath = {HOOKS_PATH}")
    return code


def hooks_check(cwd: Path) -> int:
    """Report whether this clone runs the versioned hooks.

    Args:
        cwd: The repository root.

    Returns:
        0 when ``core.hooksPath`` is :data:`HOOKS_PATH`, else 1.
    """
    result = _test_hooks.run_capturing(["git", "config", "--get", "core.hooksPath"], cwd=cwd)
    configured = result["stdout"].strip()
    if result["returncode"] == 0 and configured == HOOKS_PATH:
        _test_hooks.write_line(f"hooks installed: core.hooksPath = {configured}")
        return 0
    _test_hooks.write_error(
        "hooks NOT installed. This clone runs no pre-commit checks; the shared-index sweep "
        "is unguarded here. Run: make install-hooks"
    )
    return 1


def require_tool(name: str, hint: str) -> None:
    """Refuse when a tool is not on the PATH.

    Args:
        name: The executable.
        hint: How to install it.

    Raises:
        AppError: ``MAKETOOLS_TOOL_MISSING`` naming the tool and the hint.
    """
    if shutil.which(name) is None:
        raise AppError(MaketoolsErrorCode.TOOL_MISSING, f"{name} not found. {hint}")
    _test_hooks.write_line(f"require-tool: {name} found")


__all__ = [
    "COMPOSE_FILE",
    "HOOKS_PATH",
    "compose_down",
    "compose_up",
    "fan_out",
    "hooks_check",
    "hooks_install",
    "makefile_directories",
    "require_tool",
]
