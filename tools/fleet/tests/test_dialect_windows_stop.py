"""How a Windows node stops a dispatch: the build's whole tree, by verified id.

MEASURED BEFORE IT WAS WRITTEN (MCPs board task fd5cabfa). On sedona,
2026-09-23, a probe task whose ``build.ps1`` ran a native child read parent
alive=False, child alive=True after ``Stop-ScheduledTask``: stopping a task
ends the process it started and nothing below it, which is how a cancelled
slime run's vitest held sedona for four hours. The same probe with the build
recording ``$PID`` and the stop running ``taskkill /PID <pid> /T /F`` ended the
parent, its child and its grandchild.

The text tests pin the order and the guard. The stop script is RUN, against a
real process tree started the way a node starts one and a real disposable
scheduled task, by tests/pester/rendered-dialect-task.Tests.ps1 over its
committed render (MCPs board task d69786fa), so the claim that the tree dies
is measured on every harness run rather than remembered from sedona.
"""

from __future__ import annotations

from fleet.contracts.source import InstallStep
from fleet.core import names
from fleet.core.dialect_windows import WindowsDialect
from tests.conftest import DEMO_RUN_ID

DIALECT = WindowsDialect()


def _stop(target: str) -> str:
    """Render the stop script for the fixture's run under ``target``.

    Args:
        target: The dispatch's directory.

    Returns:
        The script's text.
    """
    return DIALECT.stop_script(target=target, run_id=DEMO_RUN_ID)


class TestTheBuildRecordsItself:
    def test_the_build_writes_its_process_id_before_anything_else(self) -> None:
        """First, so a stop that arrives during a slow install still finds
        the id; a build that recorded it later would be unstoppable for as
        long as ``npm ci`` ran."""
        body = DIALECT.build_script(
            target="C:/s/run-1",
            path="",
            workers=2,
            install=(InstallStep(phase="install", argv=("npm", "ci")),),
            cache_root="C:/c",
            isolated_docker=False,
            elevated=False,
            agent="opus-demo-0929",
        )

        lines = body.splitlines()
        header = lines.index("$ErrorActionPreference = 'Stop'")

        assert lines[header + 1] == f'$PID | Set-Content -LiteralPath "$Target/{names.PID_NAME}"'


class TestTheStopScriptText:
    def test_the_tree_is_ended_before_the_task_is_stopped(self) -> None:
        """Stopping the task first would end the build process and orphan
        its children, leaving nothing whose tree ``taskkill /T`` could walk."""
        body = _stop("C:/s/run-1")

        assert body.index("& $Taskkill /PID $buildPid /T /F") < body.index("Stop-ScheduledTask")
        assert body.index("Stop-ScheduledTask") < body.index("$root.DeleteTask($TaskName, 0)")

    def test_only_a_process_running_this_dispatchs_build_is_killed(self) -> None:
        body = _stop("C:/s/run-1")

        assert "[string]$Target = 'C:/s/run-1'" in body
        assert f'$recorded = "$Target/{names.PID_NAME}"' in body
        assert '($process.CommandLine -like "*$build*")' in body

    def test_a_kill_that_fails_fails_the_script(self) -> None:
        assert "FLEET_STOP_KILL_FAILED: taskkill of $buildPid exited $LASTEXITCODE" in _stop(
            "C:/s/run-1"
        )

    def test_a_task_already_gone_is_found_absent_rather_than_ignored(self) -> None:
        """``-ErrorAction SilentlyContinue`` on the stop read every scheduler
        failure as a task already gone; the task is looked up by name."""
        body = _stop("C:/s/run-1")

        assert "SilentlyContinue" not in body
        assert "$task = @($root.GetTasks(1) | Where-Object { $_.Name -eq $TaskName })" in body
        assert "Get-ScheduledTask " not in body
        assert "Unregister-ScheduledTask" not in body

    def test_the_task_name_is_the_one_the_launch_registers(self) -> None:
        """Both come from names.task_name, so a rename cannot make a stop
        report success having stopped nothing."""
        launched = DIALECT.launch_script(target="C:/s/run-1", run_id=DEMO_RUN_ID, elevated=False)

        assert names.task_name(DEMO_RUN_ID) in launched
        assert names.task_name(DEMO_RUN_ID) in _stop("C:/s/run-1")

    def test_the_stop_script_never_prompts(self) -> None:
        """There is nobody at the node to answer, and a prompt would hang.
        ``Unregister-ScheduledTask`` asks for confirmation unless told not
        to; Task Scheduler's own ``DeleteTask`` has no prompt to answer, and
        the stop script calls nothing that does."""
        body = _stop("C:/s/run-1")

        assert "$root.DeleteTask($TaskName, 0)" in body
        assert "-Confirm" not in body
