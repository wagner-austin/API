"""An idle runner unit whose listener predates its hooks is restarted (MCPs board task 621902c3).

The provision's lines run under bash against a real process standing as the
unit's main process, started with the environment the case gives it, so its
``/proc/<pid>/environ`` is the kernel's own. ``systemctl`` and the job-scope
helper are stand-ins: ``systemctl show`` names that process and any other
call is logged, and the helper's ``--workers`` answer is the case's.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from fleet.core import runner_job_scope_render, runner_render
from fleet.core.runner_work_clear import COMPLETED_HOOK_VARIABLE, JOB_COMPLETED_HOOK_PATH
from tests._runner_fixtures import RESERVED_CORES, a_host, an_install
from tests._stand_ins import run_with_stand_ins

#: The unit every case runs.
UNIT = "actions.runner.wagner-austin-API.lavender.service"

#: The job-started hook's line in a listener's environment.
STARTED = "ACTIONS_RUNNER_HOOK_JOB_STARTED=/usr/local/bin/fleet-runner-job-started.sh"

#: The job-completed hook's line in a listener's environment.
COMPLETED = f"{COMPLETED_HOOK_VARIABLE}={JOB_COMPLETED_HOOK_PATH}"


def _run(tmp_path: pathlib.Path, environment: tuple[str, ...], workers: str) -> tuple[str, str]:
    """Run one unit's lines with the stand-ins, its main process a real one.

    Args:
        tmp_path: The case's directory.
        environment: The main process's environment, one ``NAME=value`` each.
        workers: The helper's ``--workers`` answer.

    Returns:
        What the lines printed, and every ``systemctl`` call but ``show``.
    """
    install = an_install("wagner-austin/API")
    assert install["service"] == UNIT
    log = tmp_path / "systemctl.log"
    listener = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)"],
        env={name: value for name, _, value in (line.partition("=") for line in environment)},
    )
    try:
        ran = run_with_stand_ins(
            tmp_path,
            runner_job_scope_render.render_hooks_live_lines(install),
            {
                "systemctl": (
                    f'if [ "$1" = show ]; then echo {listener.pid}; else echo "$*" >> {log}; fi'
                ),
                runner_job_scope_render.JOB_SCOPE_PATH: f"echo '{workers}'",
            },
            options=("-eu",),
        )
    finally:
        listener.kill()
        listener.wait(timeout=10)
    assert ran.returncode == 0, ran.stderr
    calls = log.read_text(encoding="utf-8") if log.exists() else ""
    return ran.stdout, calls


@pytest.mark.host_linux
def test_a_listener_that_reads_both_hooks_is_left_running(tmp_path: pathlib.Path) -> None:
    assert _run(tmp_path, (STARTED, COMPLETED), f"{UNIT} -") == ("", "")


@pytest.mark.host_linux
@pytest.mark.parametrize("environment", [(STARTED,), ()])
def test_an_idle_listener_missing_a_hook_is_restarted(
    tmp_path: pathlib.Path, environment: tuple[str, ...]
) -> None:
    assert _run(tmp_path, environment, f"{UNIT} -") == (
        f"job hooks live: {UNIT} restarted to read fleet-job-scope.conf\n",
        f"restart {UNIT}\n",
    )


@pytest.mark.host_linux
def test_a_listener_running_a_job_is_named_and_left_running(tmp_path: pathlib.Path) -> None:
    assert _run(tmp_path, (STARTED,), f"{UNIT} 13 99000 10") == (
        f"job hooks pending: {UNIT} is running a job; a later provision restarts it once it "
        "is idle\n",
        "",
    )


def test_the_provision_checks_each_unit_s_hooks_after_placing_it() -> None:
    install = an_install("wagner-austin/API")
    lines = runner_job_scope_render.render_hooks_live_lines(install)
    assert lines[2:5] == [
        "hooks=live",
        f'case "$environ" in *"{STARTED}"*) ;; *) hooks=stale ;; esac',
        f'case "$environ" in *"{COMPLETED}"*) ;; *) hooks=stale ;; esac',
    ]
    script = runner_render.render_provision(a_host([install]), RESERVED_CORES)["provision_script"]
    assert script.index("listener placed: ") < script.index("\n".join(lines))
