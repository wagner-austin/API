"""The toolchain probe's ``hooks`` line (MCPs board tasks ec895824 and 09a34e03).

The ``hooks`` tag routes MCPs ``packages/claude-hooks``'s check, which runs
on the system interpreter and whose live suites read the build account's
``~/.claude/corvis-hooks.json``. So the line answers the route file only when
the file exists AND one ``python3 -c`` of
:data:`fleet.contracts.tagged_tools.HOOKS_CHECK_PROGRAM` finds every pin of
``HOOKS_CHECK_PACKAGES`` installed at its version. The line is RUN under
``sh`` with a ``python3`` ahead on PATH that records its arguments and exits
as told, and a ``HOME`` the case lays out; the program itself is run on this
interpreter against the distributions it really has.
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import subprocess
import sys

import pytest

from fleet.contracts.tagged_tools import (
    HOOKS_CHECK_ARGUMENTS,
    HOOKS_CHECK_PACKAGES,
    HOOKS_CHECK_PROGRAM,
)
from fleet.core.dialect_linux import PROLOGUE, SH_INVOCATION, LinuxDialect
from tests.test_dialect_linux import fields_of

DIALECT = LinuxDialect()

#: The pins as the line passes them, each name then its version.
PINS = (
    "ruff 0.17.1 mypy 2.4.0 pytest 9.1.1 pytest-xdist 3.8.0 pytest-cov 7.1.0 "
    "coverage 7.16.2 pytest-timeout 2.4.0"
)

#: The arguments the line passes, as the fake's ``$*`` records them.
CHECK_ARGUMENTS = f"-c {HOOKS_CHECK_PROGRAM} {PINS}"


def _hooks_line(tmp_path: pathlib.Path, home: pathlib.Path, check_exit: int) -> str:
    """Run the toolchain probe with a fake ``python3`` first on PATH.

    Args:
        tmp_path: Where the fake, its call record and the script go.
        home: The ``HOME`` the probe reads.
        check_exit: The fake's exit for a ``-c`` call.

    Returns:
        The probe's ``hooks`` field.
    """
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    fake = tools / "python3"
    fake.write_bytes(
        f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{(tmp_path / 'calls').as_posix()}'\n"
        f'[ "$1" = "-c" ] && exit {check_exit}\necho Python 3.11.15\n'.encode()
    )
    fake.chmod(0o755)
    script = tmp_path / "probe.sh"
    script.write_text(
        DIALECT.toolchain_probe_script().replace(
            PROLOGUE,
            PROLOGUE + f"PATH='{tools.as_posix()}':$PATH\nHOME='{home.as_posix()}'\n",
            1,
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [*SH_INVOCATION, str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    return fields_of(completed.stdout)["hooks"]


def _check_calls(tmp_path: pathlib.Path) -> int:
    """Count the version asks the fake recorded.

    Args:
        tmp_path: Where the record is.

    Returns:
        How many of its lines are the hooks line's ask.
    """
    return (tmp_path / "calls").read_text(encoding="utf-8").splitlines().count(CHECK_ARGUMENTS)


def _program_exit(*arguments: str) -> int:
    """Run the hooks program on this interpreter.

    Args:
        arguments: Names and versions, alternately.

    Returns:
        Its exit code.
    """
    return subprocess.run(
        [sys.executable, "-c", HOOKS_CHECK_PROGRAM, *arguments],
        capture_output=True,
        check=False,
        timeout=60,
    ).returncode


def test_the_line_asks_exactly_the_contract_pins() -> None:
    """The arguments the host cases record are the contract's pins, name then version."""
    assert HOOKS_CHECK_ARGUMENTS == PINS
    assert PINS.split() == [part for pin in HOOKS_CHECK_PACKAGES for part in pin.split("==")]
    assert f'python3 -c "{HOOKS_CHECK_PROGRAM}" {PINS} > /dev/null 2>&1' in (
        DIALECT.toolchain_probe_script()
    )


def test_the_program_holds_no_quote_a_shell_would_read() -> None:
    """Both probes pass it inside double quotes, PowerShell's inside single ones."""
    assert ('"' in HOOKS_CHECK_PROGRAM, "'" in HOOKS_CHECK_PROGRAM) == (False, False)


def test_the_program_answers_0_only_when_every_distribution_is_at_its_version() -> None:
    """Run for real against the distribution this suite runs under."""
    pytest_version = importlib.metadata.version("pytest")

    assert _program_exit("pytest", pytest_version) == 0
    assert _program_exit("pytest", pytest_version, "pytest", f"{pytest_version}.0") == 1
    assert _program_exit("pytest", pytest_version, "no-such-distribution-09a34e03", "1") == 1


@pytest.mark.host_linux
class TestTheHooksLineUnderSh:
    def test_a_home_without_the_route_file_answers_no_and_never_asks(
        self, tmp_path: pathlib.Path
    ) -> None:
        """pendragon's and serendipity's state, 2026-10-02."""
        home = tmp_path / "home"
        home.mkdir()

        assert _hooks_line(tmp_path, home, 0) == "no="
        assert _check_calls(tmp_path) == 0

    def test_a_route_file_without_the_pinned_tools_answers_no(self, tmp_path: pathlib.Path) -> None:
        """sedona's state, 2026-10-02: the file, and no ruff or mypy."""
        home = tmp_path / "home"
        (home / ".claude").mkdir(parents=True)
        (home / ".claude" / "corvis-hooks.json").write_text("{}", encoding="utf-8")

        assert _hooks_line(tmp_path, home, 1) == "no="
        assert _check_calls(tmp_path) == 1

    def test_the_route_file_and_the_pinned_tools_answer_the_file(
        self, tmp_path: pathlib.Path
    ) -> None:
        home = tmp_path / "home"
        (home / ".claude").mkdir(parents=True)
        route = home / ".claude" / "corvis-hooks.json"
        route.write_text("{}", encoding="utf-8")

        assert _hooks_line(tmp_path, home, 0) == f"yes={route.as_posix()}"
        assert _check_calls(tmp_path) == 1
