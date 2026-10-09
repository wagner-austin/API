"""The toolchain probe's ``stack`` and ``testdb`` lines (MCPs board tasks 554bffc1, 28c43011).

doc-extract-api, transcriber-api and pg-backup-sidecar start the corvis
compose stack's images on its network inside their own ``make check``, so a
node carries the ``stack`` tag only while the daemon its builds reach holds
both. The text is pinned everywhere; the line is RUN under ``sh`` with a
``systemd-run`` and a ``docker`` ahead on PATH, the docker answering as
diphtheria's did on 2026-09-29, or as lavender-wsl's did, which has no
``mcp-network``. The same daemon then answers the ``testdb`` line's container
ask after the stack line, whatever the stack line found (MCPs board task
939ec5c7), so each case's call log ends with that ask and the fake, holding
no such container, answers ``testdb=no=``.

Every docker ask goes through ``systemd-run --user``, where builds run (MCPs
board task 28c43011): the fake ``systemd-run`` logs its own argument line
and runs what follows its options, or, standing for colossus's user manager
on 2026-10-09, refuses, and then both lines answer ``no`` without the probe's
own process ever asking docker.
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

#: The options every docker ask is started inside a user unit with.
UNIT = "systemd-run --user --wait --pipe --quiet --collect"

#: The ``testdb`` line's ask of the daemon, made on every run.
TESTDB_ASK = f"container inspect --format {{{{.Config.Image}}}} {TESTDB_CONTAINER}"


def test_the_stack_line_asks_for_the_network_every_image_and_the_version_inside_a_unit() -> None:
    body = DIALECT.toolchain_probe_script()
    unit = 'unit() {\n  systemd-run --user --wait --pipe --quiet --collect "$@"\n}\n'
    condition = (
        f"if unit docker network inspect {STACK_NETWORK} > /dev/null 2>&1 &&"
        f" unit docker image inspect {' '.join(STACK_IMAGES)} > /dev/null 2>&1 &&"
        " v=\"$(unit docker version --format '{{.Server.Version}}' 2>/dev/null)\"; then\n"
        "  printf 'stack=yes=%s\\n' \"$v\"\n"
        "else\n"
        "  printf 'stack=no=\\n'\n"
        "fi\n"
    )
    assert unit + condition in body
    assert body.index("printf 'docker=no=\\n'") < body.index(unit)
    assert body.index(condition) < body.index("report apt-get apt-get")


def test_the_testdb_line_asks_inside_a_unit() -> None:
    assert (
        'if t="$(unit docker container inspect'
        f" --format '{{{{.Config.Image}}}}' {TESTDB_CONTAINER}" in DIALECT.toolchain_probe_script()
    )


def _run_with_docker(
    tmp_path: pathlib.Path, network_status: int, image_status: int, *, unit_starts: bool = True
) -> str:
    """Run the toolchain probe with a fake ``systemd-run`` and ``docker`` first on PATH.

    Args:
        tmp_path: Where the fakes, their call log and the script go.
        network_status: What ``docker network inspect`` exits with.
        image_status: What ``docker image inspect`` exits with.
        unit_starts: False for a ``systemd-run`` that refuses every unit.

    Returns:
        The probe's standard output.
    """
    tools = tmp_path / "tools"
    tools.mkdir()
    calls = (tmp_path / "calls").as_posix()
    runner = tools / "systemd-run"
    runner.write_bytes(
        (
            "#!/bin/sh\n"
            f"echo \"systemd-run $*\" >> '{calls}'\n"
            + ("" if unit_starts else "exit 1\n")
            + 'while [ "${1#--}" != "$1" ]; do shift; done\n'
            'exec "$@"\n'
        ).encode()
    )
    runner.chmod(0o755)
    fake = tools / "docker"
    fake.write_bytes(
        (
            "#!/bin/sh\n"
            f"echo \"$*\" >> '{calls}'\n"
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
    """The fakes' argument lines, in order.

    Args:
        tmp_path: Where the fakes wrote them.

    Returns:
        One line per call.
    """
    return (tmp_path / "calls").read_text(encoding="utf-8").splitlines()


def _in_unit(ask: str) -> list[str]:
    """The two lines one docker ask through a unit logs.

    Args:
        ask: The docker arguments.

    Returns:
        ``systemd-run``'s line, then docker's.
    """
    return [f"{UNIT} docker {ask}", ask]


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
            *_in_unit(f"network inspect {STACK_NETWORK}"),
            *_in_unit(f"image inspect {' '.join(STACK_IMAGES)}"),
            *_in_unit("version --format {{.Server.Version}}"),
            *_in_unit(TESTDB_ASK),
        ]

    def test_a_daemon_without_the_network_answers_no_and_asks_for_no_image(
        self, tmp_path: pathlib.Path
    ) -> None:
        """lavender-wsl, 2026-09-29: 'network mcp-network not found'."""
        fields = fields_of(_run_with_docker(tmp_path, 1, 0))

        assert fields["stack"] == "no="
        assert _calls(tmp_path) == [
            *_in_unit(f"network inspect {STACK_NETWORK}"),
            *_in_unit(TESTDB_ASK),
        ]
        assert list(fields)[-2:] == ["apt-get", "pipx"]

    def test_a_missing_image_answers_no(self, tmp_path: pathlib.Path) -> None:
        fields = fields_of(_run_with_docker(tmp_path, 0, 1))

        assert fields["stack"] == "no="
        assert _calls(tmp_path) == [
            *_in_unit(f"network inspect {STACK_NETWORK}"),
            *_in_unit(f"image inspect {' '.join(STACK_IMAGES)}"),
            *_in_unit(TESTDB_ASK),
        ]

    def test_a_unit_that_cannot_reach_docker_answers_no_on_both_lines(
        self, tmp_path: pathlib.Path
    ) -> None:
        """colossus, 2026-10-09: a login reached the daemon, the user manager's units did not."""
        fields = fields_of(_run_with_docker(tmp_path, 0, 0, unit_starts=False))

        assert (fields["stack"], fields["testdb"]) == ("no=", "no=")
        assert _calls(tmp_path) == [
            f"{UNIT} docker network inspect {STACK_NETWORK}",
            f"{UNIT} docker {TESTDB_ASK}",
        ]
