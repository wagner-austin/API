"""The ``sh`` phase lines, executed (MCPs board task 74b13c20).

The Linux builds are asserted line by line in their own suites; this runs
the lines :func:`fleet.core.phase_markers.sh_phase_lines` renders under a
real bash, so the stamps, the duration arithmetic and the status a failing
command leaves are what ``sh`` actually produces rather than what the text
looks like.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
from datetime import UTC, datetime

from fleet.core.phase_markers import PHASE_MARKER, sh_phase_lines
from tests._host_bash import host_bash

STAMP = r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)"


def _run(log: pathlib.Path, command: str) -> str:
    """Run one phase under ``set -eu`` and print the status it left.

    Args:
        log: The transcript.
        command: The phase's command.

    Returns:
        What the script printed: the status.
    """
    lines = ["set -eu", *sh_phase_lines(phase="install", command=command, log=log.as_posix())]
    script = "\n".join([*lines, 'printf "%s" "$status"']) + "\n"
    completed = subprocess.run(
        [host_bash(), "-c", script], capture_output=True, text=True, check=False, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def _stamp(text: str) -> datetime:
    """Read one phase line's stamp."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def test_a_phase_brackets_its_command_with_utc_stamps_and_whole_seconds(
    tmp_path: pathlib.Path,
) -> None:
    log = tmp_path / "result.txt.log"
    before = datetime.now(UTC).replace(microsecond=0)

    assert _run(log, "echo working") == "0"

    after = datetime.now(UTC)
    started, said, ended = log.read_text(encoding="utf-8").splitlines()
    opening = re.fullmatch(rf"{PHASE_MARKER} install started {STAMP}", started)
    closing = re.fullmatch(rf"{PHASE_MARKER} install ended {STAMP} after (\d+) s, exit 0", ended)
    assert opening is not None and closing is not None
    assert said == "working"
    assert before <= _stamp(opening.group(1)) <= _stamp(closing.group(1)) <= after
    assert int(closing.group(2)) <= (after - before).total_seconds() + 1


def test_a_failing_command_is_recorded_and_its_status_left_for_the_caller(
    tmp_path: pathlib.Path,
) -> None:
    log = tmp_path / "result.txt.log"

    assert _run(log, "( echo broken; exit 3 )") == "3"

    assert log.read_text(encoding="utf-8").splitlines()[-1].endswith(", exit 3")
