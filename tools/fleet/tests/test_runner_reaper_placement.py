"""The reaper's pass finishing a listener move the provision left pending.

MCPs board task 715b9ec5. The provision writes each runner unit's ``Slice=``
drop-in for the listener slice and restarts only a unit no job runs in. A
unit it found busy reads ``Slice=`` the listener slice while its cgroup stays
in the CI slice, and the reaper's 30 s pass restarts it once its Listener
runs and it holds no Worker, watching a busy one for its job's end. These run
the rendered pass over the laid-out table of :mod:`tests.test_runner_reaper`,
whose ``systemctl`` stand-in records each restart and whose ``sleep`` stand-in
records each wait and can end the job at one.
"""

from __future__ import annotations

import pathlib

from fleet.contracts.runner_slice import LISTENER_SLICE_NAME
from fleet.core.runner_job_scope_render import HOOK_ENVIRONMENT
from fleet.core.runner_reaper_render import HOOK_CASES, PLACE_WAIT_SECONDS, REAPER_SCRIPT
from tests.test_runner_reaper import (
    BOUND,
    OWN,
    SCOPE,
    STEP,
    UNIT,
    WORKER,
    Process,
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


def _slept(table: pathlib.Path) -> list[str]:
    """Each wait the pass took, in order.

    Args:
        table: The laid-out table.

    Returns:
        The seconds of each, empty when it took none.
    """
    record = table / "slept"
    return record.read_text(encoding="utf-8").split() if record.exists() else []


def _job_ends_after(table: pathlib.Path, waits: int, processes: tuple[Process, ...]) -> None:
    """Have the ``sleep`` stand-in end the job's processes at its ``waits``-th call.

    Args:
        table: The laid-out table.
        waits: The call that ends them.
        processes: The job's processes.
    """
    (table / "job-ends-after").write_bytes(f"{waits}\n".encode())
    (table / "job-pids").write_bytes(" ".join(str(each["pid"]) for each in processes).encode())


#: The rendered bound of the watch: five reads a second for PLACE_WAIT_SECONDS.
WATCH_BOUND = f'[ "$tries" -lt {PLACE_WAIT_SECONDS * 5} ]'

#: The reads a watch the job outlasts takes here. Each read runs the stand-ins
#: through host bash, so the rendered hundred made each such case 23 to 26 s,
#: the two slowest of the suite, and pushed tools/fleet's check past its
#: 300 s budget on pendragon (fleet job a3af5548, 309 s, MCPs board task
#: 3d9eec0d). The bound itself is pinned on the rendered text instead.
SHORT_TRIES = 3

#: The rendered reaper with its watch bounded at SHORT_TRIES reads.
SHORT_WATCH = REAPER_SCRIPT.replace(WATCH_BOUND, f'[ "$tries" -lt {SHORT_TRIES} ]')

#: Every wait of a shortened watch that never sees the job end.
WHOLE_WATCH = ["0.2"] * SHORT_TRIES


#: The line the pass writes for a unit it moved.
MOVED = (
    f"fleet-runner-reaper: {UNIT}: restarted between jobs into {LISTENER_SLICE_NAME}, as its "
    "drop-in says\n"
)


#: The line the pass writes for a unit it restarted to read its hooks.
HOOKED = (
    f"fleet-runner-reaper: {UNIT}: restarted between jobs to read the job hooks its drop-in names\n"
)


def _placed(table: pathlib.Path) -> None:
    """Put the laid-out unit's own processes in the listener slice, as a placed unit's are.

    Args:
        table: The table :func:`tests.test_runner_reaper._lay_out` wrote.
    """
    placed = table / "cgroup" / LISTENER_SLICE_NAME / UNIT
    placed.mkdir(parents=True)
    (placed / "cgroup.procs").write_bytes(b"10\n11\n12\n")
    (table / "show" / f"{UNIT}.ControlGroup").write_bytes(
        f"/{LISTENER_SLICE_NAME}/{UNIT}\n".encode()
    )


def _environ(table: pathlib.Path, *assignments: str) -> None:
    """Give the unit's main process, pid 10, its environment.

    Args:
        table: The laid-out table.
        *assignments: Its ``NAME=value`` entries.
    """
    (table / "proc" / "10" / "environ").write_bytes(
        "".join(f"{assignment}\0" for assignment in assignments).encode()
    )


class TestStaleHooks:
    """MCPs board task 621902c3: the provision found both MCPs units on lavender
    running jobs and left them without the job-completed hook."""

    def test_an_idle_placed_unit_whose_listener_lacks_a_hook_is_restarted(
        self, tmp_path: pathlib.Path
    ) -> None:
        _lay_out(tmp_path, OWN, configured=LISTENER_SLICE_NAME)
        _placed(tmp_path)
        _environ(tmp_path, "HOME=/home/gharunner", HOOK_ENVIRONMENT[0])
        ran = _run(tmp_path, BOUND)
        assert (ran.returncode, ran.stdout, ran.stderr) == (0, HOOKED, "")
        assert (_restarted(tmp_path), _slept(tmp_path)) == ([UNIT], [])

    def test_a_listener_holding_every_hook_is_left_running(self, tmp_path: pathlib.Path) -> None:
        _lay_out(tmp_path, OWN, configured=LISTENER_SLICE_NAME)
        _placed(tmp_path)
        _environ(tmp_path, *HOOK_ENVIRONMENT)
        ran = _run(tmp_path, BOUND)
        assert (ran.returncode, ran.stdout, _restarted(tmp_path)) == (0, "", [])

    def test_a_unit_whose_job_runs_in_its_scope_is_restarted_when_the_job_ends(
        self, tmp_path: pathlib.Path
    ) -> None:
        _lay_out(tmp_path, OWN, configured=LISTENER_SLICE_NAME)
        _placed(tmp_path)
        _environ(tmp_path, HOOK_ENVIRONMENT[0])
        _scope(tmp_path, SCOPE, (WORKER, STEP))
        _job_ends_after(tmp_path, 2, (WORKER, STEP))
        ran = _run(tmp_path, BOUND)
        assert (ran.returncode, ran.stdout, ran.stderr) == (0, HOOKED, "")
        assert (_restarted(tmp_path), _slept(tmp_path)) == ([UNIT], ["0.2", "0.2"])

    def test_the_cases_name_every_hook_the_drop_in_sets(self) -> None:
        assert HOOK_ENVIRONMENT == (
            "ACTIONS_RUNNER_HOOK_JOB_STARTED=/usr/local/bin/fleet-runner-job-started.sh",
            "ACTIONS_RUNNER_HOOK_JOB_COMPLETED=/usr/local/bin/fleet-runner-job-completed.sh",
        )
        assert HOOK_CASES in REAPER_SCRIPT
        assert HOOK_CASES.count("*) return 0 ;; esac") == 2


class TestAPendingMove:
    def test_an_idle_unit_the_drop_in_names_for_the_listener_slice_is_restarted(
        self, tmp_path: pathlib.Path
    ) -> None:
        _lay_out(tmp_path, OWN, configured=LISTENER_SLICE_NAME)
        ran = _run(tmp_path, BOUND)
        assert (ran.returncode, ran.stdout, ran.stderr) == (0, MOVED, "")
        assert (_restarted(tmp_path), _slept(tmp_path)) == ([UNIT], [])
        assert _killed(tmp_path) == []

    def test_a_unit_running_a_job_is_restarted_the_moment_the_job_ends(
        self, tmp_path: pathlib.Path
    ) -> None:
        """MCPs lavender-wsl-3 went job to job past every 30 s pass for 82 min."""
        _lay_out(tmp_path, (*OWN, WORKER, STEP), configured=LISTENER_SLICE_NAME)
        _job_ends_after(tmp_path, 3, (WORKER, STEP))
        ran = _run(tmp_path, BOUND)
        assert (ran.returncode, ran.stdout, ran.stderr) == (0, MOVED, "")
        assert (_restarted(tmp_path), _slept(tmp_path)) == ([UNIT], ["0.2", "0.2", "0.2"])

    def test_a_job_that_outlasts_the_watch_is_left_for_the_next_pass(
        self, tmp_path: pathlib.Path
    ) -> None:
        _lay_out(tmp_path, (*OWN, WORKER, STEP), configured=LISTENER_SLICE_NAME)
        ran = _run(tmp_path, BOUND, script=SHORT_WATCH)
        assert (ran.returncode, ran.stdout, _restarted(tmp_path)) == (0, "", [])
        assert _slept(tmp_path) == WHOLE_WATCH

    def test_the_rendered_watch_reads_five_times_a_second_for_its_bound(self) -> None:
        assert REAPER_SCRIPT.count(WATCH_BOUND) == 1
        assert SHORT_WATCH.count(WATCH_BOUND) == 0

    def test_a_job_in_its_scope_counts_as_running(self, tmp_path: pathlib.Path) -> None:
        _lay_out(tmp_path, OWN, configured=LISTENER_SLICE_NAME)
        _scope(tmp_path, SCOPE, (WORKER, STEP))
        ran = _run(tmp_path, BOUND, script=SHORT_WATCH)
        assert (ran.returncode, ran.stdout, _restarted(tmp_path)) == (0, "", [])
        assert _slept(tmp_path) == WHOLE_WATCH

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
        _placed(tmp_path)
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
