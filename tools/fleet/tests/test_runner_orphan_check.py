"""The audit row for what a finished job left behind (MCPs board task 53528106, A2 and A4).

The row asks the reaper's own ``--audit`` mode, so it runs here under the
audit's functions with the reaper a stand-in that answers as each case says
(tests/_audit_rows.py); the reaper's counting itself runs in
test_runner_reaper.
"""

from __future__ import annotations

import pathlib

import pytest

from fleet.contracts.runners import HostRunnerSpec
from fleet.core import runner_orphan_check
from fleet.core.runner_reaper_render import REAPER_PATH
from tests._audit_rows import run_rows
from tests._host_bash import bash_path
from tests._runner_fixtures import a_host, an_install

#: The install every case audits.
INSTALL = an_install("wagner-austin/MCPs")

#: Its unit.
UNIT = INSTALL["service"]


def _host(minutes: int) -> HostRunnerSpec:
    """A host bounded at ``minutes``.

    Args:
        minutes: Its job timeout.

    Returns:
        The host.
    """
    host = a_host([INSTALL])
    host["job_timeout_minutes"] = minutes
    return host


def _row(tmp_path: pathlib.Path, minutes: int, reaper: str) -> str:
    """The row's ``CHECK`` line, the reaper a stand-in.

    Args:
        tmp_path: The case's directory.
        minutes: The host's job timeout.
        reaper: The stand-in reaper's body; it records its arguments.

    Returns:
        The line.
    """
    [row] = run_rows(
        tmp_path,
        runner_orphan_check.render_orphan_check_lines(_host(minutes), INSTALL),
        {REAPER_PATH: f'echo "$*" > "{bash_path(tmp_path)}/asked"\n{reaper}'},
    )
    return row


def test_the_row_is_named_for_the_repository_and_runner() -> None:
    assert runner_orphan_check.orphan_check_id(INSTALL) == "orphans:wagner-austin/MCPs:lavender"


def test_a_reaper_counting_nothing_passes_and_was_asked_at_the_host_s_bound(
    tmp_path: pathlib.Path,
) -> None:
    assert _row(tmp_path, 60, "echo 0") == "CHECK orphans:wagner-austin/MCPs:lavender OK"
    assert (tmp_path / "asked").read_text(encoding="utf-8") == f"--audit 3600 {UNIT}\n"


@pytest.mark.parametrize(
    ("reaper", "counted"), [("echo 3", "3 (exit 0)"), ("echo 0; exit 1", "0 (exit 1)")]
)
def test_a_count_or_a_failed_reaper_drifts_with_what_it_said(
    tmp_path: pathlib.Path, reaper: str, counted: str
) -> None:
    assert _row(tmp_path, 360, reaper) == (
        "CHECK orphans:wagner-austin/MCPs:lavender DRIFT processes older than 360 minutes that "
        f"a finished job left in {UNIT}, or under a Worker past that job timeout; the reaper "
        f"counted: {counted}"
    )


def test_an_unscriptable_unit_is_refused() -> None:
    with pytest.raises(ValueError, match="service"):
        runner_orphan_check.render_orphan_check_lines(
            _host(60), {**INSTALL, "service": "bad'unit.service"}
        )
