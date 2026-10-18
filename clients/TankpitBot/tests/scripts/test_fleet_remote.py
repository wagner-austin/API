"""Tests for reaching the game host: the checked runner and the remote command line."""

from __future__ import annotations

import shlex
from collections.abc import Generator
from pathlib import Path

import pytest
from scripts.fleet_remote import GAME_HOST_SSH, FleetHostError, on_host, run_checked

from scripts import _test_hooks as script_hooks


@pytest.fixture(autouse=True)
def _restore_run_command() -> Generator[None, None, None]:
    """Put the command runner back after each test.

    Yields:
        None, with the original runner restored after.
    """
    run_command = script_hooks.run_command
    yield
    script_hooks.run_command = run_command


def test_run_checked_returns_the_output_of_a_command_that_succeeds(tmp_path: Path) -> None:
    """A zero exit hands back standard output, and the command ran where it was asked."""
    seen: list[tuple[list[str], Path]] = []

    def _answer(argv: list[str], cwd: Path) -> script_hooks.CommandResult:
        seen.append((argv, cwd))
        return script_hooks.CommandResult(returncode=0, stdout="colossus\n", stderr="ignored")

    script_hooks.run_command = _answer
    assert run_checked(["ssh", GAME_HOST_SSH, "hostname"], tmp_path, "UNUSED") == "colossus\n"
    assert seen == [(["ssh", GAME_HOST_SSH, "hostname"], tmp_path)]


def test_run_checked_names_the_code_command_exit_and_stderr(tmp_path: Path) -> None:
    """A failed command is refused with its code, the command, its exit and its stderr."""

    def _answer(argv: list[str], cwd: Path) -> script_hooks.CommandResult:
        return script_hooks.CommandResult(returncode=255, stdout="", stderr=" host down \n")

    script_hooks.run_command = _answer
    with pytest.raises(FleetHostError) as raised:
        run_checked(["ssh", GAME_HOST_SSH, "hostname"], tmp_path, "FLEET_X_FAILED")
    assert str(raised.value) == (
        f"FLEET_X_FAILED: ssh {GAME_HOST_SSH} hostname exited 255: host down"
    )


def test_on_host_hands_the_remote_shell_one_quoted_line() -> None:
    """Every word reaches the host's shell as itself, spaces and quotes included."""
    argv = ["curl", "-s", "--data", '{"room": "Practice"}', "http://127.0.0.1:27300/bots"]

    line = on_host(argv)

    assert line[:2] == ["ssh", GAME_HOST_SSH]
    assert len(line) == 3
    assert shlex.split(line[2]) == argv
