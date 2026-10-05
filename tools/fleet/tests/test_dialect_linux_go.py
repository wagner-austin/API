"""The toolchain probe's ``go`` line (MCPs board task 1da15750).

The ``go`` tag routes MCPs ``rcs-bridge``'s check. Go has no ``--version``
flag: asked it, go prints ``flag provided but not defined: -version`` and its
usage, so the probe asks
:data:`fleet.contracts.tagged_tools.GO_VERSION_ARGUMENT` instead, through the
third argument ``report`` takes. The line is RUN under ``sh`` with a ``go``
ahead on PATH that answers as go does: its version for ``version`` and the
refusal for anything else.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from fleet.contracts.tagged_tools import GO_VERSION_ARGUMENT
from fleet.core.dialect_linux import PROLOGUE, SH_INVOCATION, LinuxDialect
from tests.test_dialect_linux import fields_of

DIALECT = LinuxDialect()

#: What the fake answers ``go version`` with, as go1.27.1 does on Linux.
GO_ANSWER = "go version go1.27.1 linux/amd64"


def _probe_fields(tmp_path: pathlib.Path, with_go: bool) -> dict[str, str]:
    """Run the toolchain probe with ``PATH`` holding only the case's tools.

    Args:
        tmp_path: Where the fake and the script go.
        with_go: Whether the tools directory holds a ``go``.

    Returns:
        The probe's fields.
    """
    tools = tmp_path / "tools"
    tools.mkdir()
    if with_go:
        fake = tools / "go"
        fake.write_bytes(
            (
                "#!/bin/sh\n"
                f'if [ "$1" = "{GO_VERSION_ARGUMENT}" ]; then echo "{GO_ANSWER}"; exit 0; fi\n'
                "echo 'flag provided but not defined: -version'\nexit 2\n"
            ).encode()
        )
        fake.chmod(0o755)
    script = tmp_path / "probe.sh"
    script.write_text(
        DIALECT.toolchain_probe_script().replace(
            PROLOGUE, PROLOGUE + f"PATH='{tools.as_posix()}':/usr/bin:/bin\n", 1
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
    return fields_of(completed.stdout)


def test_go_is_asked_its_version_argument_and_every_other_tool_its_flag() -> None:
    body = DIALECT.toolchain_probe_script()

    assert GO_VERSION_ARGUMENT == "version"
    assert f"report go go {GO_VERSION_ARGUMENT}\n" in body
    assert '"$("$2" "${3:---version}" 2>&1' in body


@pytest.mark.host_linux
class TestTheGoLineUnderSh:
    def test_a_go_on_the_path_answers_its_version(self, tmp_path: pathlib.Path) -> None:
        """What a node answers once go is installed; asked ``--version`` it
        would have recorded go's refusal as the version."""
        fields = _probe_fields(tmp_path, with_go=True)

        assert fields["go"] == f"yes={GO_ANSWER}"
        # Every other tool is still asked --version: tar's answer is a version.
        assert fields["tar"].startswith("yes=tar (GNU tar)")

    def test_no_go_on_the_path_answers_no(self, tmp_path: pathlib.Path) -> None:
        """diphtheria and lavender-wsl, 2026-10-05."""
        assert _probe_fields(tmp_path, with_go=False)["go"] == "no="
