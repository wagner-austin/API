"""The toolchain probe's ``docker`` line (MCPs board task 6c4516af).

The ``docker`` tag routes the isolated lane: a build run as the node's
execdocker user against that user's rootless daemon, which runs ``docker
compose`` and builds images. So the line answers the daemon's version only
when the daemon says it is rootless AND that user has the compose and buildx
CLI plugins. The line is RUN under ``sh`` with an ``id`` that knows
execdocker and a ``sudo`` ahead on PATH answering as diphtheria's daemon did,
as a rootful one, as lavender-wsl's did before it had either plugin, or as a
``sudo`` that wants a password.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from fleet.core.dialect_linux import PROLOGUE, SH_INVOCATION, LinuxDialect
from tests.test_dialect_linux import fields_of

DIALECT = LinuxDialect()

#: What execdocker's daemon answered on diphtheria, 2026-09-29.
ROOTLESS_ANSWER = '29.8.1 ["name=seccomp,profile=builtin","name=rootless","name=cgroupns"]'


def _run_with_sudo(tmp_path: pathlib.Path, sudo_body: str) -> dict[str, str]:
    """Run the toolchain probe with a fake ``id`` and ``sudo`` first on PATH.

    Args:
        tmp_path: Where the fakes and the script go.
        sudo_body: The fake ``sudo``'s script, after its shebang line.

    Returns:
        The probe's fields.
    """
    tools = tmp_path / "tools"
    tools.mkdir()
    fake_id = tools / "id"
    fake_id.write_bytes(b'#!/bin/sh\n[ "$1" = "-u" ] && echo 1001\nexit 0\n')
    fake_id.chmod(0o755)
    fake_sudo = tools / "sudo"
    fake_sudo.write_bytes(f"#!/bin/sh\n{sudo_body}".encode())
    fake_sudo.chmod(0o755)
    script = tmp_path / "probe.sh"
    script.write_text(
        DIALECT.toolchain_probe_script().replace(
            PROLOGUE, PROLOGUE + f"PATH='{tools.as_posix()}':$PATH\n", 1
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


@pytest.mark.host_linux
class TestTheDockerLineUnderSh:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            (ROOTLESS_ANSWER, "yes=29.8.1"),
            ('29.8.1 ["name=apparmor","name=seccomp,profile=builtin"]', "no="),
            ("", "no="),
        ],
    )
    def test_the_docker_line_reports_only_a_rootless_daemon(
        self, tmp_path: pathlib.Path, answer: str, expected: str
    ) -> None:
        """A ``sudo`` answering every call as the daemon would, plugins
        included: the version comes through only when the security options
        name rootless, so the stack's rootful daemon, or no daemon, reads as
        absent."""
        fields = _run_with_sudo(tmp_path, f"printf '%s' '{answer}'\n")

        assert fields["docker"] == expected

    @pytest.mark.parametrize("missing", ["compose", "buildx"])
    def test_a_rootless_daemon_without_a_cli_plugin_reads_as_no_docker(
        self, tmp_path: pathlib.Path, missing: str
    ) -> None:
        """lavender-wsl's execdocker had a rootless daemon and neither plugin,
        and job 7ad01613 died there at ``docker: unknown command: docker
        compose``: a node lacking either reads as having no docker at all."""
        fields = _run_with_sudo(
            tmp_path,
            f'case "$*" in *" {missing} version"*) '
            f"echo 'docker: unknown command: docker {missing}' >&2; exit 1 ;; esac\n"
            f"printf '%s' '{ROOTLESS_ANSWER}'\n",
        )

        assert fields["docker"] == "no="
        assert list(fields)[-2:] == ["apt-get", "pipx"]

    def test_a_sudo_that_refuses_reads_as_no_daemon_and_the_probe_goes_on(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A ``sudo`` that wants a password exits 1, as execdocker's own did
        on diphtheria (MCPs board task a8ee9b21): the probe reports
        ``docker=no=`` and still reaches its last line, where ``set -e``
        once ended it at the assignment."""
        fields = _run_with_sudo(tmp_path, "echo 'sudo: a password is required' >&2\nexit 1\n")

        assert fields["docker"] == "no="
        assert list(fields)[-2:] == ["apt-get", "pipx"]
