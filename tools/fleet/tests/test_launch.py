"""The scripts a node is handed, and the quoting that broke them.

EVERY TEST IN THE FIRST TWO CLASSES IS A REGRESSION FROM ONE DISPATCH. On
2026-09-04 the first ``fleet-run`` to reach a node registered a scheduled task
whose ``-Argument`` was the eleven characters ``-Command "cd`` and whose
WORKING DIRECTORY was the remaining two hundred, because the build was
interpolated into a single-quoted PowerShell string that contained single
quotes. The task could not be started. Nothing failed: PowerShell exited 0 and
the ledger recorded a run that did not exist.

These assertions are about the TEXT of the scripts rather than about running
them, and that is the honest limit of what a test on this machine can say. The
things they pin -- no inner quote in the argument, a wait for the task to
actually start, the battery settings -- are each the difference between a
dispatch that runs and one that silently does not, and each was found by
reading a task's XML off a node rather than by reasoning about the string.
"""

from __future__ import annotations

from fleet.cli import cancel
from fleet.core import launch
from tests.conftest import DEMO_PROJECT, DEMO_RUN_ID


class TestBuildScript:
    def test_it_runs_the_recipe_in_the_project(self) -> None:
        body = launch.build_script(target="C:/s/run-1", project=DEMO_PROJECT, workers=6)

        assert f"Set-Location -LiteralPath 'C:/s/run-1/{DEMO_PROJECT}'" in body
        assert "make check" in body

    def test_it_pins_the_worker_count(self) -> None:
        body = launch.build_script(target="C:/s/run-1", project=DEMO_PROJECT, workers=6)

        assert "PYTEST_XDIST_AUTO_NUM_WORKERS = '6'" in body

    def test_it_records_the_status_last(self) -> None:
        """The result file's absence is how a run is known to be unfinished.

        Written after the recipe, so it can never exist while make is still
        going -- which is what lets `fleet-collect` treat absence as running.
        """
        body = launch.build_script(target="C:/s/run-1", project=DEMO_PROJECT, workers=6)
        lines = [line for line in body.splitlines() if line.strip()]

        assert lines[-1].startswith("$LASTEXITCODE")
        assert launch.RESULT_NAME in lines[-1]

    def test_it_reads_the_exit_code_and_not_the_success_flag(self) -> None:
        """`make` writes to stderr on a passing run; under redirection that
        sets $? false in PS 5.1 while $LASTEXITCODE stays correct."""
        body = launch.build_script(target="C:/s/run-1", project=DEMO_PROJECT, workers=6)

        assert "$LASTEXITCODE" in body
        assert "$?" not in body


class TestRegisterScript:
    def test_it_registers_and_starts_a_scheduled_task(self) -> None:
        """Not an ssh child. Windows OpenSSH puts that in a job object that
        dies with the connection, and this command returns immediately."""
        body = launch.register_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)

        assert "Register-ScheduledTask" in body
        assert "Start-ScheduledTask" in body

    def test_it_sets_priority_four(self) -> None:
        """Priority 7 is the Register-ScheduledTask default and sets LOW I/O.

        A run that inherits it crawls, and the symptom reads as a slow node
        rather than a misconfigured launch.
        """
        body = launch.register_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)

        assert "-Priority 4" in body
        assert "[TimeSpan]::Zero" in body
        assert "-LogonType S4U" in body

    def test_it_runs_the_build_by_path_and_never_inlines_it(self) -> None:
        """THE REGRESSION. Interpolating the build into -Argument split the
        task in two: PowerShell ended the single-quoted string at the first
        inner quote and bound the rest to -WorkingDirectory. Measured on
        sedona 2026-09-04; the task could not be started at all.
        """
        body = launch.register_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)

        assert f'-File "C:/s/run-1/{launch.BUILD_SCRIPT_NAME}"' in body
        assert "-Command" not in body
        assert "make check" not in body

    def test_the_argument_string_contains_no_single_quotes(self) -> None:
        """The mechanical form of the same defect, asserted directly.

        -Argument is passed as a single-quoted PowerShell string, so ANY
        single quote inside it terminates the argument early.
        """
        body = launch.register_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)
        argument = body.split("-Argument '", 1)[1].split("'\n", 1)[0]

        assert "'" not in argument

    def test_it_survives_the_lid_being_shut(self) -> None:
        """Two of the three nodes are laptops and both battery settings
        default to refusing: without these a dispatch to an unplugged sedona
        registers a task that never runs, and reports nothing.
        """
        body = launch.register_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)

        assert "-AllowStartIfOnBatteries" in body
        assert "-DontStopIfGoingOnBatteries" in body

    def test_it_waits_for_the_task_to_actually_start(self) -> None:
        """Start-ScheduledTask reports a refusal as a NON-terminating error.

        On 2026-09-04 it failed with 'Element not found', PowerShell exited 0,
        and the dispatch was recorded as running. The script now watches for
        the task to leave SCHED_S_TASK_HAS_NOT_RUN and throws if it does not.
        """
        body = launch.register_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)

        assert "$ErrorActionPreference = 'Stop'" in body
        assert str(launch.TASK_HAS_NOT_RUN) in body
        assert "throw" in body

    def test_the_task_name_is_the_one_cancel_stops(self) -> None:
        """Both come from launch.task_name, so a rename cannot make
        fleet-cancel report success having stopped nothing."""
        registered = launch.register_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)

        assert launch.task_name(DEMO_RUN_ID) in registered
        assert launch.task_name(DEMO_RUN_ID) in cancel.stop_script(DEMO_RUN_ID)


class TestResultScript:
    def test_it_reports_when_as_well_as_what(self) -> None:
        """Whether a run was PROTECTED is a question about whether its lease
        covered it, and only the node knows when the build ended. Asking for
        the status alone forced the reader to substitute "is a lease held now"
        -- a question about how promptly somebody collected -- which refused a
        run that finished three minutes inside its window."""
        body = launch.result_script("C:/s/run-1")

        assert "LastWriteTimeUtc" in body
        assert "1970-01-01" in body

    def test_it_does_not_use_uformat_for_the_epoch(self) -> None:
        """PowerShell 5.1's -UFormat %s converts from LOCAL time, which would
        put every node's answer out by its own offset."""
        assert "-UFormat" not in launch.result_script("C:/s/run-1")

    def test_the_result_script_prints_nothing_while_running(self) -> None:
        """Absence is the signal, so an unfinished run is not read as exit 0."""
        body = launch.result_script("C:/s/run-1")

        assert "Test-Path" in body

    def test_the_stop_script_never_prompts(self) -> None:
        """There is nobody at the node to answer, and a prompt would hang."""
        assert "-Confirm:$false" in cancel.stop_script(DEMO_RUN_ID)
