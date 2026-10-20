"""The one writer of a runner host's unit files, executed under a real bash.

A ``systemctl`` stand-in on ``PATH`` records every call, so what is asserted
is what the lines do to the file and to systemd, not their text.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from fleet.core.runner_unit_files import (
    UNIT_FILE_END,
    drop_in_path,
    render_drop_in_lines,
    render_unit_file_lines,
)
from tests._host_bash import bash_path, host_bash
from tests._runner_fixtures import an_install

#: A unit body with the shapes a real one carries: a section, directives and
#: a value holding quotes and a dollar sign the heredoc must keep verbatim.
BODY = "[Service]\nEnvironment=\"A=$HOME\" 'B=c'\nKillMode=process\n"


def _run(tmp_path: pathlib.Path, lines: list[str]) -> tuple[str, list[str]]:
    """Run rendered lines with a recording ``systemctl`` first on ``PATH``.

    Args:
        tmp_path: The case's directory.
        lines: The lines.

    Returns:
        What they printed, and each systemctl call.
    """
    stand_ins = tmp_path / "bin"
    stand_ins.mkdir(exist_ok=True)
    (stand_ins / "systemctl").write_bytes(
        b'#!/usr/bin/env bash\necho "$*" >> "$(dirname "$0")/../systemctl.log"\n'
    )
    log = tmp_path / "systemctl.log"
    log.write_bytes(b"")
    (tmp_path / "lines.sh").write_bytes(("\n".join(lines) + "\n").encode())
    ran = subprocess.run(
        [
            host_bash(),
            "-c",
            'chmod +x bin/systemctl && PATH="$PWD/bin:$PATH" bash -eu lines.sh',
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    assert ran.returncode == 0, ran.stderr
    return ran.stdout, log.read_text(encoding="utf-8").splitlines()


def test_an_absent_unit_is_written_byte_for_byte_and_systemd_reloaded(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "etc" / "unit.d" / "fleet.conf"
    spelled = f"{bash_path(tmp_path)}/etc/unit.d/fleet.conf"
    out, calls = _run(tmp_path, render_unit_file_lines(spelled, BODY, announcement="set"))
    assert path.read_bytes() == BODY.encode()
    assert (out, calls) == ("set\n", ["daemon-reload"])


def test_a_unit_holding_the_body_already_is_left_alone(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "unit.conf"
    path.write_bytes(BODY.encode())
    spelled = f"{bash_path(tmp_path)}/unit.conf"
    out, calls = _run(tmp_path, render_unit_file_lines(spelled, BODY, announcement="set"))
    assert (out, calls) == ("", [])


def test_a_unit_holding_other_text_is_rewritten(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "unit.conf"
    path.write_bytes(b"[Service]\nKillMode=control-group\n")
    spelled = f"{bash_path(tmp_path)}/unit.conf"
    out, calls = _run(tmp_path, render_unit_file_lines(spelled, BODY, announcement="set"))
    assert path.read_bytes() == BODY.encode()
    assert (out, calls) == ("set\n", ["daemon-reload"])


def test_a_body_with_a_line_that_would_end_the_heredoc_is_refused() -> None:
    with pytest.raises(ValueError) as refused:
        render_unit_file_lines("/etc/x.conf", f"[Service]\n{UNIT_FILE_END}\n", announcement="set")
    assert str(refused.value) == f"the body for /etc/x.conf holds a line {UNIT_FILE_END!r}"


@pytest.mark.parametrize(
    ("path", "announcement", "label"),
    [("/etc/it's.conf", "set", "unit file path"), ("/etc/x.conf", "it's set", "announcement")],
)
def test_a_path_or_announcement_that_cannot_be_embedded_is_refused(
    path: str, announcement: str, label: str
) -> None:
    with pytest.raises(ValueError, match=label):
        render_unit_file_lines(path, BODY, announcement=announcement)


def test_a_drop_in_lives_in_its_unit_s_directory_and_is_written_by_the_same_lines() -> None:
    install = an_install("wagner-austin/MCPs")
    path = drop_in_path(install, "fleet-kill.conf")
    assert path == (
        "/etc/systemd/system/actions.runner.wagner-austin-MCPs.lavender.service.d/fleet-kill.conf"
    )
    assert render_drop_in_lines(
        install, name="fleet-kill.conf", body=BODY, announcement="set"
    ) == render_unit_file_lines(path, BODY, announcement="set")


def test_a_unit_name_that_cannot_be_embedded_is_refused() -> None:
    install = an_install()
    install["service"] = "bad'unit.service"
    with pytest.raises(ValueError, match="service"):
        drop_in_path(install, "fleet-kill.conf")
