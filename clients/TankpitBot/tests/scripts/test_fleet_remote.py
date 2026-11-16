"""Tests for reaching the game host: the checked runner and the remote command line."""

from __future__ import annotations

import shlex
from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict, require_str
from scripts.fleet_remote import (
    GAME_HOST,
    GAME_HOST_HOME,
    GAME_HOST_SSH,
    ROSTER_DIR,
    FleetHostError,
    on_host,
    run_checked,
)

from scripts import _test_hooks as script_hooks


def test_the_game_host_is_reached_at_its_roster_user_and_tailnet_address() -> None:
    """The ssh destination and home come from the host's own roster file.

    colossus's account was typed here as an address until colossus was
    reinstalled and the account was gone; the roster is where a machine's
    address is kept, so the destination is read from demure's file.
    """
    assert GAME_HOST == "demure"
    machine = narrow_json_to_dict(
        load_json_str((ROSTER_DIR / "demure.json").read_text(encoding="utf-8"))
    )
    user = require_str(machine, "user")
    assert GAME_HOST_SSH == f"{user}@{require_str(machine, 'tailnetIp')}"
    assert GAME_HOST_HOME == f"/home/{user}"
    assert ROSTER_DIR.parts[-4:] == ("tools", "fleet", "registry", "machines")


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
        return script_hooks.CommandResult(returncode=0, stdout="demure\n", stderr="ignored")

    script_hooks.run_command = _answer
    assert run_checked(["ssh", GAME_HOST_SSH, "hostname"], tmp_path, "UNUSED") == "demure\n"
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
