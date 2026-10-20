"""The dropped-job reading and its audit row, against fakes at the command seam.

MCPs board task 715b9ec5, A1. Each fake answers through ``_test_hooks.run``
exactly as :mod:`tests.test_core_io` describes: first one ``gh`` runner list
per repository of the host's installs, in name order, then the job scope
helper's ``--workers`` lines over ssh.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import _test_hooks, runner_dropped_check
from fleet.core.runner_audit_types import AuditFinding, AuditOutcome
from fleet.core.runner_dropped_check import DroppedReading, DroppedWorker, HostWorker
from fleet.core.runner_job_scope_render import JOB_SCOPE_PATH
from tests._runner_fixtures import a_host, an_install
from tests.conftest import FakeRun, failed, ok

#: The installs: three runners of API and one of x, x sorting after API.
INSTALLS: tuple[RunnerInstall, ...] = (
    an_install("wagner-austin/API"),
    an_install("wagner-austin/API", "lavender-2", slot=2),
    an_install("wagner-austin/API", "lavender-3", slot=3),
    an_install("wagner-austin/x"),
)


def _host() -> HostRunnerSpec:
    """A lavender-shaped host holding :data:`INSTALLS`.

    Returns:
        The spec.
    """
    return a_host(INSTALLS)


#: The command the helper's read runs on the host.
WORKERS_READ = (
    "sudo",
    "-n",
    JOB_SCOPE_PATH,
    "--workers",
    *(install["service"] for install in INSTALLS),
)


def _worker(pid: int, age: int) -> HostWorker:
    """A Worker, its start in ticks a fixed function of its pid.

    Args:
        pid: Its pid.
        age: Its age in seconds.

    Returns:
        The Worker.
    """
    return HostWorker(pid=pid, start=pid * 100, age=age)


def _answer(*units: tuple[HostWorker, ...]) -> str:
    """The helper's ``--workers`` answer, one tuple of Workers per install in order.

    Args:
        *units: Each unit's Workers, none for ``<unit> -``.

    Returns:
        The lines.
    """
    lines: list[str] = []
    for install, workers in zip(INSTALLS, units, strict=True):
        if not workers:
            lines.append(f"{install['service']} -\n")
        for worker in workers:
            lines.append(
                f"{install['service']} {worker['pid']} {worker['start']} {worker['age']}\n"
            )
    return "".join(lines)


#: GitHub reading every API runner online and busy.
API_BUSY = "1 lavender online true\n2 lavender-2 online true\n3 lavender-3 online true\n"

#: GitHub reading x's runner online and busy.
X_BUSY = "4 lavender online true\n"


def _one(*, passed: bool, detail: str) -> AuditOutcome:
    """The outcome holding the row's one finding.

    Args:
        passed: Its verdict.
        detail: Its detail.

    Returns:
        The outcome, compared whole.
    """
    finding = AuditFinding(
        check_id="dropped-jobs:lavender",
        ok=passed,
        detail=detail,
        reason=runner_dropped_check.DROPPED_REASON,
    )
    return AuditOutcome(findings=[finding], reason="")


#: The clean detail over the four installs.
CLEAN = (
    "examined 4 runner(s) against GitHub; none holds a Runner.Worker over 300 s that "
    "GitHub reads not busy or unlisted, or that a newer Worker superseded"
)


class TestTheReading:
    def test_each_dropped_worker_is_named_by_its_runner_pid_and_start(self) -> None:
        """The 2026-10-16 shape: GitHub reads the runners offline or idle, or
        lists them no more, while their Workers run on."""
        github = "1 lavender offline false\n2 lavender-2 online false\n3 lavender-3 online true\n"
        runner = FakeRun(
            [
                ok(github),
                ok(""),
                ok(
                    _answer(
                        (_worker(41, 6300),),
                        (_worker(42, 400),),
                        (_worker(43, 9000),),
                        (_worker(44, 301),),
                    )
                ),
            ]
        )
        _test_hooks.run = runner
        reading = runner_dropped_check.read_dropped(_host())
        assert reading == DroppedReading(
            examined=4,
            dropped=[
                DroppedWorker(
                    install=INSTALLS[0], worker=_worker(41, 6300), said="offline busy=false"
                ),
                DroppedWorker(
                    install=INSTALLS[1], worker=_worker(42, 400), said="online busy=false"
                ),
                DroppedWorker(install=INSTALLS[3], worker=_worker(44, 301), said="unlisted"),
            ],
            failure=None,
        )
        assert runner.calls[0][3] == "repos/wagner-austin/API/actions/runners"
        assert runner.calls[1][3] == "repos/wagner-austin/x/actions/runners"
        assert tuple(runner.calls[2][-(len(WORKERS_READ) + 1) :]) == ("lavender", *WORKERS_READ)

    def test_an_older_worker_a_newer_one_superseded_is_dropped_whatever_github_says(
        self,
    ) -> None:
        """A runner runs one job at a time, so a unit holding two Workers lost
        the older one's job and took the next: GitHub reads it online and busy
        with the newer job."""
        older, newer = _worker(51, 900), _worker(52, 30)
        # The second unit's older Worker is inside the grace: a job's handoff.
        handing, taking = _worker(53, 200), _worker(54, 30)
        _test_hooks.run = FakeRun(
            [ok(API_BUSY), ok(X_BUSY), ok(_answer((newer, older), (handing, taking), (), ()))]
        )
        reading = runner_dropped_check.read_dropped(_host())
        assert reading["dropped"] == [
            DroppedWorker(install=INSTALLS[0], worker=older, said="superseded by Runner.Worker 52")
        ]

    def test_a_worker_inside_the_grace_or_none_is_not_named(self) -> None:
        """GitHub's list lags a job's start, so a Worker of exactly the grace
        under a runner GitHub reads idle is a job starting, not a dropped one."""
        idle = "1 lavender offline false\n2 lavender-2 online false\n"
        _test_hooks.run = FakeRun(
            [ok(idle), ok(""), ok(_answer((_worker(61, 300),), (), (_worker(62, 12),), ()))]
        )
        assert runner_dropped_check.read_dropped(_host())["dropped"] == []

    def test_a_host_that_did_not_answer_is_its_failure(self) -> None:
        _test_hooks.run = FakeRun([ok(API_BUSY), ok(X_BUSY), failed(255, "No route to host")])
        reading = runner_dropped_check.read_dropped(_host())
        failure = reading["failure"]
        assert reading["examined"] == 4
        assert reading["dropped"] == []
        assert failure == {
            "code": FleetErrorCode.NODE_UNREACHABLE,
            "message": "ssh to lavender failed while running "
            f"`{' '.join(WORKERS_READ)}`: No route to host",
        }

    @pytest.mark.parametrize(
        "line",
        ["actions.runner.x", "actions.runner.x 12 13", "actions.runner.x a b c", "u 1 2 3 4"],
    )
    def test_a_line_of_any_other_shape_is_unparsable(self, line: str) -> None:
        _test_hooks.run = FakeRun([ok(API_BUSY), ok(X_BUSY), ok(_answer((), (), (), ()) + line)])
        with pytest.raises(AppError) as fault:
            runner_dropped_check.read_dropped(_host())
        assert fault.value.code is FleetErrorCode.RUNNER_AUDIT_UNPARSABLE
        assert repr(line) in fault.value.message

    def test_an_answer_that_stops_early_is_unparsable_not_workerless(self) -> None:
        stopped = "".join(f"{install['service']} -\n" for install in INSTALLS[:2])
        _test_hooks.run = FakeRun([ok(API_BUSY), ok(X_BUSY), ok(stopped)])
        with pytest.raises(AppError) as fault:
            runner_dropped_check.read_dropped(_host())
        assert fault.value.code is FleetErrorCode.RUNNER_AUDIT_UNPARSABLE
        assert fault.value.message == (
            "lavender's job scope helper named no Worker for "
            f"{INSTALLS[2]['service']}, {INSTALLS[3]['service']}"
        )


class TestTheAuditRow:
    def test_busy_runners_with_old_workers_are_ok_and_the_count_is_named(self) -> None:
        _test_hooks.run = FakeRun(
            [
                ok(API_BUSY),
                ok(X_BUSY),
                ok(_answer((_worker(71, 9000),), (_worker(72, 9000),), (), (_worker(74, 9000),))),
            ]
        )
        outcome = runner_dropped_check.attempt_dropped_check(_host())
        assert outcome == _one(passed=True, detail=CLEAN)

    def test_each_dropped_worker_drifts_the_row_by_name(self) -> None:
        _test_hooks.run = FakeRun(
            [
                ok("1 lavender online false\n"),
                ok(X_BUSY),
                ok(_answer((_worker(81, 700),), (), (), ())),
            ]
        )
        outcome = runner_dropped_check.attempt_dropped_check(_host())
        assert outcome == _one(
            passed=False,
            detail="examined 4 runner(s) against GitHub; 1 Runner.Worker(s) hold a job "
            "GitHub dropped: wagner-austin/API lavender Runner.Worker 81 "
            "(online busy=false, 700 s)",
        )

    def test_a_runner_github_reads_offline_and_busy_still_holds_its_job(self) -> None:
        """GitHub keeps the job of a runner whose listener is out of touch:
        the first rolled reap pass, which ended such Workers, failed two jobs
        GitHub still held on 2026-10-17."""
        _test_hooks.run = FakeRun(
            [
                ok("1 lavender offline true\n2 lavender-2 offline true\n"),
                ok(X_BUSY),
                ok(_answer((_worker(91, 9000),), (_worker(92, 9000),), (), ())),
            ]
        )
        outcome = runner_dropped_check.attempt_dropped_check(_host())
        assert outcome == _one(passed=True, detail=CLEAN)

    def test_a_host_ssh_cannot_reach_is_a_value_with_its_reason(self) -> None:
        _test_hooks.run = FakeRun([ok(API_BUSY), ok(X_BUSY), failed(255, "No route to host")])
        outcome = runner_dropped_check.attempt_dropped_check(_host())
        assert outcome["findings"] is None
        assert "No route to host" in outcome["reason"]

    def test_a_helper_that_ran_and_failed_drifts_the_row_with_its_words(self) -> None:
        """A host the helper has not reached yet answers, and says so."""
        _test_hooks.run = FakeRun(
            [ok(API_BUSY), ok(X_BUSY), failed(127, "fleet-runner-job-scope: not found")]
        )
        outcome = runner_dropped_check.attempt_dropped_check(_host())
        assert outcome == _one(
            passed=False,
            detail="the job scope helper did not read the Workers: running "
            f"`{' '.join(WORKERS_READ)}` on lavender exited 127: "
            "fleet-runner-job-scope: not found",
        )
