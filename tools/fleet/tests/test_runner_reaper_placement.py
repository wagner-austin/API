"""The reaper's pass finishing a listener move the provision left pending.

MCPs board task 715b9ec5. The provision writes each runner unit's ``Slice=``
drop-in for the listener slice and restarts only a unit no job runs in. A
unit it found busy reads ``Slice=`` the listener slice while its cgroup stays
in the CI slice, and the reaper's 30 s pass restarts it once its Listener
runs and it holds no Worker. These run the rendered pass over the laid-out
table of :mod:`tests.test_runner_reaper`, whose ``systemctl`` stand-in
records each restart.
"""

from __future__ import annotations

import pathlib

from fleet.contracts.runner_slice import LISTENER_SLICE_NAME
from tests.test_runner_reaper import (
    BOUND,
    OWN,
    SCOPE,
    STEP,
    UNIT,
    WORKER,
    _killed,
    _lay_out,
    _run,
    _scope,
)


def _restarted(table: pathlib.Path) -> list[str]:
    """The units the pass restarted, in order.

    Args:
        table: The laid-out table.

    Returns:
        The units, empty when none was.
    """
    record = table / "restarted"
    return record.read_text(encoding="utf-8").split() if record.exists() else []


#: The line the pass writes for a unit it moved.
MOVED = (
    f"fleet-runner-reaper: {UNIT}: restarted between jobs into {LISTENER_SLICE_NAME}, as its "
    "drop-in says\n"
)


class TestAPendingMove:
    def test_an_idle_unit_the_drop_in_names_for_the_listener_slice_is_restarted(
        self, tmp_path: pathlib.Path
    ) -> None:
        _lay_out(tmp_path, OWN, configured=LISTENER_SLICE_NAME)
        ran = _run(tmp_path, BOUND)
        assert (ran.returncode, ran.stdout, ran.stderr) == (0, MOVED, "")
        assert _restarted(tmp_path) == [UNIT]
        assert _killed(tmp_path) == []

    def test_a_unit_running_a_job_waits_for_a_pass_between_jobs(
        self, tmp_path: pathlib.Path
    ) -> None:
        _lay_out(tmp_path, (*OWN, WORKER, STEP), configured=LISTENER_SLICE_NAME)
        ran = _run(tmp_path, BOUND)
        assert (ran.returncode, ran.stdout, _restarted(tmp_path)) == (0, "", [])

    def test_a_job_in_its_scope_counts_as_running(self, tmp_path: pathlib.Path) -> None:
        _lay_out(tmp_path, OWN, configured=LISTENER_SLICE_NAME)
        _scope(tmp_path, SCOPE, (WORKER, STEP))
        ran = _run(tmp_path, BOUND)
        assert (ran.returncode, ran.stdout, _restarted(tmp_path)) == (0, "", [])

    def test_a_unit_mid_self_update_is_left_for_a_later_pass(self, tmp_path: pathlib.Path) -> None:
        """runsvc.sh and the node service, the Listener gone: an update."""
        _lay_out(tmp_path, OWN[:2], configured=LISTENER_SLICE_NAME)
        ran = _run(tmp_path, BOUND)
        assert (ran.returncode, ran.stdout, _restarted(tmp_path)) == (0, "", [])


class TestNoMoveOwed:
    def test_a_unit_whose_drop_in_names_the_ci_slice_is_not_moved(
        self, tmp_path: pathlib.Path
    ) -> None:
        _lay_out(tmp_path, OWN)
        ran = _run(tmp_path, BOUND)
        assert (ran.returncode, ran.stdout, _restarted(tmp_path)) == (0, "", [])

    def test_a_unit_already_in_the_listener_slice_is_not_moved(
        self, tmp_path: pathlib.Path
    ) -> None:
        _lay_out(tmp_path, OWN, configured=LISTENER_SLICE_NAME)
        placed = tmp_path / "cgroup" / LISTENER_SLICE_NAME / UNIT
        placed.mkdir(parents=True)
        (placed / "cgroup.procs").write_bytes(b"10\n11\n12\n")
        (tmp_path / "show" / f"{UNIT}.ControlGroup").write_bytes(
            f"/{LISTENER_SLICE_NAME}/{UNIT}\n".encode()
        )
        ran = _run(tmp_path, BOUND)
        assert (ran.returncode, ran.stdout, _restarted(tmp_path)) == (0, "", [])

    def test_a_stopped_unit_is_placed_by_its_next_start(self, tmp_path: pathlib.Path) -> None:
        _lay_out(tmp_path, OWN, configured=LISTENER_SLICE_NAME)
        (tmp_path / "show" / f"{UNIT}.ControlGroup").write_bytes(b"\n")
        ran = _run(tmp_path, BOUND)
        assert (ran.returncode, ran.stdout, _restarted(tmp_path)) == (0, "", [])

    def test_a_unit_whose_cgroup_cannot_be_read_is_not_moved(self, tmp_path: pathlib.Path) -> None:
        _lay_out(tmp_path, OWN, configured=LISTENER_SLICE_NAME)
        (tmp_path / "cgroup" / "runners.slice" / UNIT / "cgroup.procs").unlink()
        ran = _run(tmp_path, BOUND)
        assert (ran.returncode, ran.stdout, _restarted(tmp_path)) == (0, "", [])
