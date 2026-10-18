"""Tests for freezing HEAD into a release on the game host's ladder.

Every command goes through ``scripts._test_hooks.run_command``, here a
recorder that also writes the archive ``git archive -o`` names, so the copy
the release sends can be read back. ``pyproject.toml`` is a real file under
``tmp_path``.
"""

from __future__ import annotations

import runpy
import shlex
from collections.abc import Generator
from pathlib import Path

import pytest
from scripts.fleet_host import RELEASES_DIR
from scripts.fleet_release import ARCHIVED_PATHS, release, release_name
from scripts.fleet_remote import GAME_HOST_SSH, FleetHostError, on_host

from scripts import _test_hooks as script_hooks
from tests.scripts._recorder import Recorder, fail, ok

NAME = "v0.1.0-88d4a0556"
FINAL = f"{RELEASES_DIR}/{NAME}"


class Archiving(Recorder):
    """A recorder whose ``git archive`` writes the archive it was asked for."""

    def __call__(self, argv: list[str], cwd: Path) -> script_hooks.CommandResult:
        """Write the archive a ``git archive -o`` names, then answer as the recorder does.

        Args:
            argv: The command.
            cwd: Ignored.

        Returns:
            The scripted answer.
        """
        if argv[:4] == ["git", "-C", "../..", "archive"]:
            Path(argv[argv.index("-o") + 1]).write_text("tar bytes", encoding="utf-8")
        return super().__call__(argv, cwd)


def _project(tmp_path: Path, version_line: str) -> Path:
    project = tmp_path / "TankpitBot"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        f'[tool.poetry]\nname = "tankpit-bot"\n{version_line}\n', encoding="utf-8"
    )
    return project


@pytest.fixture(autouse=True)
def _restore_run_command() -> Generator[None, None, None]:
    """Put the command runner back after each test.

    Yields:
        None, with the original runner restored after.
    """
    run_command = script_hooks.run_command
    yield
    script_hooks.run_command = run_command


def test_release_name_is_the_version_and_heads_short_sha(tmp_path: Path) -> None:
    """The name is read from pyproject.toml's version line and git's short sha."""
    project = _project(tmp_path, 'version = "0.1.0"')
    script_hooks.run_command = Recorder({"rev-parse": ok("88d4a0556\n")})
    assert release_name(project) == NAME


def test_release_name_refuses_a_project_without_a_version(tmp_path: Path) -> None:
    """A pyproject.toml with no version line is refused by name, before git is asked."""
    project = _project(tmp_path, 'description = "no version here"')
    recorder = Recorder({})
    script_hooks.run_command = recorder
    with pytest.raises(FleetHostError) as raised:
        release_name(project)
    assert str(raised.value) == f"FLEET_NO_VERSION: {project / 'pyproject.toml'} states none"
    assert recorder.calls == []


def test_a_release_the_host_holds_is_not_written_again(tmp_path: Path) -> None:
    """A release already on the ladder is reported and nothing is archived or sent."""
    project = _project(tmp_path, 'version = "0.1.0"')
    recorder = Recorder({"rev-parse": ok("88d4a0556\n")})
    script_hooks.run_command = recorder
    assert release(project, tmp_path) == [
        f"release {NAME} is already on {GAME_HOST_SSH}:{RELEASES_DIR}; nothing to do"
    ]
    assert recorder.calls[1:] == [on_host(["test", "-d", FINAL])]


def test_release_archives_head_and_unpacks_it_whole_on_the_host(tmp_path: Path) -> None:
    """HEAD's three trees are archived, sent, unpacked beside the ladder and renamed in."""
    project = _project(tmp_path, 'version = "0.1.0"')
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    recorder = Archiving({"rev-parse": ok("88d4a0556\n"), "test -d": fail("")})
    script_hooks.run_command = recorder
    lines = release(project, scratch)
    archive = scratch / f"{NAME}.tar"
    assert recorder.calls[2:] == [
        ["git", "-C", "../..", "archive", "--format=tar", "-o", str(archive), "HEAD", "--",
         *ARCHIVED_PATHS],
        on_host(["mkdir", "-p", RELEASES_DIR]),
        ["scp", "-q", str(archive), f"{GAME_HOST_SSH}:{RELEASES_DIR}/"],
        on_host(
            ["sh", "-c",
             'set -e; rm -rf "$2"; mkdir "$2"; tar -xf "$1" -C "$2"; rm "$1"; mv "$2" "$3"',
             "sh", f"{FINAL}.tar", f"{FINAL}.partial", FINAL]
        ),
    ]  # fmt: skip
    assert recorder.staged == {f"{NAME}.tar": "tar bytes"}
    assert ARCHIVED_PATHS == ("clients/TankpitBot", "libs/platform_core", "libs/monorepo_guards")
    assert lines == [
        f"release {NAME} written to {GAME_HOST_SSH}:{FINAL}",
        "activate: make down, then make up, which builds its image on the host",
    ]


def test_release_refuses_an_archive_git_cannot_make(tmp_path: Path) -> None:
    """A failed archive stops the release before anything reaches the host."""
    project = _project(tmp_path, 'version = "0.1.0"')
    recorder = Recorder(
        {"rev-parse": ok("88d4a0556\n"), "test -d": fail(""), " archive ": fail("bad object")}
    )
    script_hooks.run_command = recorder
    with pytest.raises(FleetHostError) as raised:
        release(project, tmp_path)
    assert str(raised.value).startswith("FLEET_ARCHIVE_FAILED: git -C ../.. archive ")
    assert str(raised.value).endswith("exited 1: bad object")
    assert len(recorder.calls) == 3


def test_release_refuses_an_unpack_the_host_refuses(tmp_path: Path) -> None:
    """An unpack that fails on the host is refused by name, and nothing is renamed in."""
    project = _project(tmp_path, 'version = "0.1.0"')
    recorder = Archiving(
        {"rev-parse": ok("88d4a0556\n"), "test -d": fail(""), "tar -xf": fail("No space left")}
    )
    script_hooks.run_command = recorder
    with pytest.raises(FleetHostError) as raised:
        release(project, tmp_path)
    assert str(raised.value).startswith(f"FLEET_RELEASE_FAILED: ssh {GAME_HOST_SSH} sh -c ")
    assert str(raised.value).endswith("exited 1: No space left")
    assert shlex.split(recorder.calls[-1][2])[-1] == FINAL


def test_module_runs_as_a_program(capsys: pytest.CaptureFixture[str]) -> None:
    """``python -m scripts.fleet_release`` releases this package's HEAD."""
    script_hooks.run_command = Recorder({"rev-parse": ok("88d4a0556\n")})
    runpy.run_module("scripts.fleet_release", run_name="__main__")
    assert capsys.readouterr().out == (
        f"release {NAME} is already on {GAME_HOST_SSH}:{RELEASES_DIR}; nothing to do\n"
    )
