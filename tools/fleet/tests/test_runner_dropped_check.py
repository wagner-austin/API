"""The dropped-job row, against fakes at the command seam (MCPs board task 715b9ec5, A1).

Each fake answers through ``_test_hooks.run`` exactly as
:mod:`tests.test_core_io` describes: first one ``gh`` runner list per
repository of the host's WSL installs, in name order, then the job scope
helper's ``--workers`` lines over ssh.
"""

from __future__ import annotations

from typing import Literal

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import _test_hooks, runner_dropped_check
from fleet.core.runner_audit_types import AuditFinding, AuditOutcome
from fleet.core.runner_job_scope_render import JOB_SCOPE_PATH
from tests._runner_fixtures import a_base, a_ci_slice
from tests.conftest import FakeRun, failed, ok


def _install(repo: str, runner_name: str, side: Literal["wsl", "windows"]) -> RunnerInstall:
    """One install of the test host.

    Args:
        repo: Its repository.
        runner_name: Its runner's name.
        side: ``wsl`` or ``windows``.

    Returns:
        The install, its unit named after both.
    """
    stem = repo.replace("/", "-")
    return RunnerInstall(
        repo=repo,
        runner_name=runner_name,
        side=side,
        service=f"actions.runner.{stem}.{runner_name}" + (".service" if side == "wsl" else ""),
        workdir=f"/home/gharunner/{runner_name}/_work",
        labels=[runner_name],
        python_toolcache=[],
    )


#: The WSL installs: three runners of API and one of x, x sorting after API.
WSL: tuple[RunnerInstall, ...] = (
    _install("wagner-austin/API", "lavender-wsl", "wsl"),
    _install("wagner-austin/API", "lavender-wsl-2", "wsl"),
    _install("wagner-austin/API", "lavender-wsl-3", "wsl"),
    _install("wagner-austin/x", "lavender-wsl", "wsl"),
)

#: A Windows install, which the row does not read.
WINDOWS = _install("wagner-austin/chat", "lavender", "windows")


def _host(*installs: RunnerInstall) -> HostRunnerSpec:
    """A lavender-shaped host with the given installs.

    Args:
        *installs: Its installs.

    Returns:
        The spec.
    """
    return HostRunnerSpec(
        name="lavender",
        host="lavender",
        wsl_distro="Ubuntu",
        keepalive_task=None,
        wslconfig_min_memory_gb=None,
        scratch_dir="C:/fleet/stage",
        gpu_required=False,
        systemd_timers=[],
        job_timeout_minutes=360,
        installs=list(installs),
        assets=[],
        base=a_base(),
        ci_slice=a_ci_slice(),
    )


def _ages(*ages: str) -> str:
    """The helper's answer, one age per WSL install in order.

    Args:
        *ages: Each unit's oldest Worker age, ``-`` for none.

    Returns:
        The lines.
    """
    return "".join(f"{install['service']} {age}\n" for install, age in zip(WSL, ages, strict=True))


#: GitHub reading every API runner online and busy.
API_BUSY = "lavender-wsl online true\nlavender-wsl-2 online true\nlavender-wsl-3 online true\n"

#: GitHub reading x's runner online and busy.
X_BUSY = "lavender-wsl online true\n"


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


#: The clean detail over the four WSL installs.
CLEAN = (
    "examined 4 WSL runner(s) against GitHub; none holds a Runner.Worker over 300 s that "
    "GitHub reads idle, offline or unlisted"
)


class TestAHostThatDropsNothing:
    def test_a_host_with_no_wsl_install_is_not_read(self) -> None:
        runner = FakeRun([])
        _test_hooks.run = runner
        outcome = runner_dropped_check.attempt_dropped_check(_host(WINDOWS))
        assert outcome == AuditOutcome(findings=[], reason="")
        assert runner.calls == []

    def test_busy_runners_with_old_workers_are_ok_and_the_count_is_named(self) -> None:
        runner = FakeRun([ok(API_BUSY), ok(X_BUSY), ok(_ages("9000", "9000", "-", "9000"))])
        _test_hooks.run = runner
        outcome = runner_dropped_check.attempt_dropped_check(_host(*WSL, WINDOWS))
        assert outcome == _one(passed=True, detail=CLEAN)
        assert runner.calls[0][3] == "repos/wagner-austin/API/actions/runners"
        assert runner.calls[1][3] == "repos/wagner-austin/x/actions/runners"
        assert tuple(runner.calls[2][-13:]) == (
            "lavender",
            "wsl",
            "-d",
            "Ubuntu",
            "-u",
            "root",
            "--",
            JOB_SCOPE_PATH,
            "--workers",
            *(install["service"] for install in WSL),
        )

    def test_an_idle_runner_s_worker_inside_the_grace_or_absent_is_not_named(self) -> None:
        """GitHub's list lags a job's start, so a Worker of exactly the grace
        under a runner GitHub reads idle is a job starting, not a dropped one."""
        idle = "lavender-wsl offline false\nlavender-wsl-2 online false\n"
        _test_hooks.run = FakeRun([ok(idle), ok(""), ok(_ages("300", "-", "12", "-"))])
        outcome = runner_dropped_check.attempt_dropped_check(_host(*WSL))
        assert outcome == _one(passed=True, detail=CLEAN)


class TestAHostHoldingDroppedJobs:
    def test_each_worker_past_the_grace_that_github_reads_jobless_is_named(self) -> None:
        """The 2026-10-16 shape: GitHub reads the runners offline or idle, or
        lists them no more, while their Workers run on."""
        github = (
            "lavender-wsl offline false\nlavender-wsl-2 online false\nlavender-wsl-3 online true\n"
        )
        _test_hooks.run = FakeRun([ok(github), ok(""), ok(_ages("6300", "400", "9000", "301"))])
        outcome = runner_dropped_check.attempt_dropped_check(_host(*WSL))
        assert outcome == _one(
            passed=False,
            detail="examined 4 WSL runner(s) against GitHub; 3 hold a job GitHub dropped: "
            "wagner-austin/API lavender-wsl (offline busy=false, Worker 6300 s); "
            "wagner-austin/API lavender-wsl-2 (online busy=false, Worker 400 s); "
            "wagner-austin/x lavender-wsl (unlisted, Worker 301 s)",
        )

    def test_an_offline_runner_github_still_reads_busy_is_named(self) -> None:
        _test_hooks.run = FakeRun(
            [ok("lavender-wsl offline true\n"), ok(X_BUSY), ok(_ages("700", "-", "-", "-"))]
        )
        outcome = runner_dropped_check.attempt_dropped_check(_host(*WSL))
        assert outcome == _one(
            passed=False,
            detail="examined 4 WSL runner(s) against GitHub; 1 hold a job GitHub dropped: "
            "wagner-austin/API lavender-wsl (offline busy=true, Worker 700 s)",
        )


class TestAReadThatFails:
    def test_a_host_ssh_cannot_reach_is_a_value_with_its_reason(self) -> None:
        _test_hooks.run = FakeRun([ok(API_BUSY), ok(X_BUSY), failed(255, "No route to host")])
        outcome = runner_dropped_check.attempt_dropped_check(_host(*WSL))
        assert outcome["findings"] is None
        assert "No route to host" in outcome["reason"]

    def test_a_helper_that_ran_and_failed_drifts_the_row_with_its_words(self) -> None:
        """A host the helper has not reached yet answers, and says so."""
        _test_hooks.run = FakeRun(
            [ok(API_BUSY), ok(X_BUSY), failed(127, "fleet-runner-job-scope: not found")]
        )
        outcome = runner_dropped_check.attempt_dropped_check(_host(*WSL))
        command = " ".join(
            (
                "wsl -d Ubuntu -u root --",
                JOB_SCOPE_PATH,
                "--workers",
                *(install["service"] for install in WSL),
            )
        )
        assert outcome == _one(
            passed=False,
            detail="the job scope helper did not read the Worker ages: running "
            f"`{command}` on lavender exited 127: fleet-runner-job-scope: not found",
        )

    @pytest.mark.parametrize(
        "line", ["actions.runner.x", "actions.runner.x 12 13", "actions.runner.x soon"]
    )
    def test_a_line_of_any_other_shape_is_unparsable(self, line: str) -> None:
        _test_hooks.run = FakeRun([ok(API_BUSY), ok(X_BUSY), ok(_ages("-", "-", "-", "-") + line)])
        with pytest.raises(AppError) as fault:
            runner_dropped_check.attempt_dropped_check(_host(*WSL))
        assert fault.value.code is FleetErrorCode.RUNNER_AUDIT_UNPARSABLE
        assert repr(line) in fault.value.message

    def test_an_answer_that_stops_early_is_unparsable_not_workerless(self) -> None:
        stopped = "".join(f"{install['service']} -\n" for install in WSL[:2])
        _test_hooks.run = FakeRun([ok(API_BUSY), ok(X_BUSY), ok(stopped)])
        with pytest.raises(AppError) as fault:
            runner_dropped_check.attempt_dropped_check(_host(*WSL))
        assert fault.value.code is FleetErrorCode.RUNNER_AUDIT_UNPARSABLE
        assert fault.value.message == (
            "lavender's job scope helper named no Worker age for "
            f"{WSL[2]['service']}, {WSL[3]['service']}"
        )
