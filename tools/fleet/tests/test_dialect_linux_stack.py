"""The toolchain probe's ``stack`` line (MCPs board task 554bffc1).

doc-extract-api, transcriber-api and pg-backup-sidecar start the corvis
compose stack's images on its network inside their own ``make check``, so a
node carries the ``stack`` tag only while the daemon its account reaches
holds both. The text is pinned everywhere; the line is RUN under ``sh`` with
a ``docker`` ahead on PATH that answers as diphtheria's did on 2026-09-29, or
as lavender-wsl's did, which has no ``mcp-network``. The same PATH daemon
then answers the ``testdb`` line's container ask after the stack line,
whatever the stack line found (MCPs board task 939ec5c7), so each case's
call log ends with that ask and the fake, holding no such container,
answers ``testdb=no=``.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from fleet.contracts.capability import STACK_IMAGES, STACK_NETWORK
from fleet.contracts.detection import TESTDB_CONTAINER
from fleet.core.dialect_linux import PROLOGUE, SH_INVOCATION, LinuxDialect
from tests.test_dialect_linux import fields_of

DIALECT = LinuxDialect()

#: The ``testdb`` line's ask of the PATH daemon, made on every run.
TESTDB_ASK = f"container inspect --format {{{{.Config.Image}}}} {TESTDB_CONTAINER}"


def test_the_stack_line_asks_for_the_network_every_image_and_the_version() -> None:
    body = DIALECT.toolchain_probe_script()
    condition = (
        f"if docker network inspect {STACK_NETWORK} > /dev/null 2>&1 &&"
        f" docker image inspect {' '.join(STACK_IMAGES)} > /dev/null 2>&1 &&"
        " v=\"$(docker version --format '{{.Server.Version}}' 2>/dev/null)\"; then\n"
        "  printf 'stack=yes=%s\\n' \"$v\"\n"
        "else\n"
        "  printf 'stack=no=\\n'\n"
        "fi\n"
    )
    assert condition in body
    assert body.index("printf 'docker=no=\\n'") < body.index(condition)
    assert body.index(condition) < body.index("report apt-get apt-get")


def _run_with_docker(tmp_path: pathlib.Path, network_status: int, image_status: int) -> str:
    """Run the toolchain probe with a fake ``docker`` first on PATH.

    Args:
        tmp_path: Where the fake, its call log and the script go.
        network_status: What ``docker network inspect`` exits with.
        image_status: What ``docker image inspect`` exits with.

    Returns:
        The probe's standard output.
    """
    tools = tmp_path / "tools"
    tools.mkdir()
    fake = tools / "docker"
    fake.write_bytes(
        (
            "#!/bin/sh\n"
            f"echo \"$*\" >> '{(tmp_path / 'calls').as_posix()}'\n"
            'case "$1" in\n'
            f"  network) exit {network_status} ;;\n"
            f"  image) exit {image_status} ;;\n"
            "  version) echo 29.8.1 ;;\n"
            "  container) exit 1 ;;\n"
            "esac\n"
        ).encode()
    )
    fake.chmod(0o755)
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
    return completed.stdout


def _calls(tmp_path: pathlib.Path) -> list[str]:
    """The fake ``docker``'s argument lines, in order.

    Args:
        tmp_path: Where the fake wrote them.

    Returns:
        One line per call.
    """
    return (tmp_path / "calls").read_text(encoding="utf-8").splitlines()


@pytest.mark.host_linux
class TestTheStackLineUnderSh:
    def test_a_daemon_holding_the_network_and_every_image_answers_its_version(
        self, tmp_path: pathlib.Path
    ) -> None:
        """diphtheria, 2026-09-29: mcp-network and all three images on 29.8.1."""
        fields = fields_of(_run_with_docker(tmp_path, 0, 0))

        assert fields["stack"] == "yes=29.8.1"
        assert fields["testdb"] == "no="
        assert _calls(tmp_path) == [
            f"network inspect {STACK_NETWORK}",
            f"image inspect {' '.join(STACK_IMAGES)}",
            "version --format {{.Server.Version}}",
            TESTDB_ASK,
        ]

    def test_a_daemon_without_the_network_answers_no_and_asks_for_no_image(
        self, tmp_path: pathlib.Path
    ) -> None:
        """lavender-wsl, 2026-09-29: 'network mcp-network not found'."""
        fields = fields_of(_run_with_docker(tmp_path, 1, 0))

        assert fields["stack"] == "no="
        assert _calls(tmp_path) == [f"network inspect {STACK_NETWORK}", TESTDB_ASK]
        assert list(fields)[-2:] == ["apt-get", "pipx"]

    def test_a_missing_image_answers_no(self, tmp_path: pathlib.Path) -> None:
        fields = fields_of(_run_with_docker(tmp_path, 0, 1))

        assert fields["stack"] == "no="
        assert _calls(tmp_path) == [
            f"network inspect {STACK_NETWORK}",
            f"image inspect {' '.join(STACK_IMAGES)}",
            TESTDB_ASK,
        ]
