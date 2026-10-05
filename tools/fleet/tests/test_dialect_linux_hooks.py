"""The toolchain probe's ``hooks`` line (MCPs board task ec895824).

The ``hooks`` tag routes MCPs ``packages/claude-hooks``'s check, which runs
on the system interpreter and whose live suites read the build account's
``~/.claude/corvis-hooks.json``. So the line answers the route file only when
the file exists AND one ``python3 -c`` imports every module of
:data:`fleet.contracts.tagged_tools.HOOKS_CHECK_MODULES`. The line is RUN under
``sh`` with a ``python3`` ahead on PATH that records its arguments and exits
as told, and a ``HOME`` the case lays out.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from fleet.contracts.tagged_tools import HOOKS_CHECK_MODULES
from fleet.core.dialect_linux import PROLOGUE, SH_INVOCATION, LinuxDialect
from tests.test_dialect_linux import fields_of

DIALECT = LinuxDialect()

#: The arguments the line passes, as the fake's ``$*`` records them.
IMPORT_ARGUMENTS = "-c import ruff, mypy, pytest, xdist, pytest_cov"


def _hooks_line(tmp_path: pathlib.Path, home: pathlib.Path, import_exit: int) -> str:
    """Run the toolchain probe with a fake ``python3`` first on PATH.

    Args:
        tmp_path: Where the fake, its call record and the script go.
        home: The ``HOME`` the probe reads.
        import_exit: The fake's exit for a ``-c`` call.

    Returns:
        The probe's ``hooks`` field.
    """
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    fake = tools / "python3"
    fake.write_bytes(
        f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{(tmp_path / 'calls').as_posix()}'\n"
        f'[ "$1" = "-c" ] && exit {import_exit}\necho Python 3.11.15\n'.encode()
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


def _import_calls(tmp_path: pathlib.Path) -> int:
    """Count the import asks the fake recorded.

    Args:
        tmp_path: Where the record is.

    Returns:
        How many of its lines are the hooks line's import.
    """
    return (tmp_path / "calls").read_text(encoding="utf-8").splitlines().count(IMPORT_ARGUMENTS)


def test_the_line_imports_exactly_the_contract_modules() -> None:
    """The arguments the host cases record are the contract's modules."""
    assert ", ".join(HOOKS_CHECK_MODULES) == IMPORT_ARGUMENTS.removeprefix("-c import ")
    assert f'python3 -c "import {", ".join(HOOKS_CHECK_MODULES)}"' in (
        DIALECT.toolchain_probe_script()
    )


@pytest.mark.host_linux
class TestTheHooksLineUnderSh:
    def test_a_home_without_the_route_file_answers_no_and_never_imports(
        self, tmp_path: pathlib.Path
    ) -> None:
        """pendragon's and serendipity's state, 2026-10-02."""
        home = tmp_path / "home"
        home.mkdir()

        assert _hooks_line(tmp_path, home, 0) == "no="
        assert _import_calls(tmp_path) == 0

    def test_a_route_file_without_the_tools_answers_no(self, tmp_path: pathlib.Path) -> None:
        """sedona's state, 2026-10-02: the file, and no ruff or mypy."""
        home = tmp_path / "home"
        (home / ".claude").mkdir(parents=True)
        (home / ".claude" / "corvis-hooks.json").write_text("{}", encoding="utf-8")

        assert _hooks_line(tmp_path, home, 1) == "no="
        assert _import_calls(tmp_path) == 1

    def test_the_route_file_and_the_tools_answer_the_file(self, tmp_path: pathlib.Path) -> None:
        home = tmp_path / "home"
        (home / ".claude").mkdir(parents=True)
        route = home / ".claude" / "corvis-hooks.json"
        route.write_text("{}", encoding="utf-8")

        assert _hooks_line(tmp_path, home, 0) == f"yes={route.as_posix()}"
        assert _import_calls(tmp_path) == 1
