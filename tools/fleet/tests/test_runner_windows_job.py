"""The job-started hook a Windows runner host lays, as rendered.

What the hook and the lines that lay it DO is executed by
tests/pester/rendered-runner-windows-job.Tests.ps1 over the committed hook,
and by tests/pester/rendered-runner-windows.Tests.ps1 over the committed
install and audit.
"""

from __future__ import annotations

import base64

from fleet.core import runner_windows_job_render
from fleet.core.powershell_text import STRICT_HEADER
from fleet.core.runner_audit_types import ExpectedCheck


class TestTheHook:
    def test_it_takes_the_runner_s_worker_and_its_own_process_by_default(self) -> None:
        lines = runner_windows_job_render.hook_script().splitlines()
        assert lines[:6] == [
            "param(",
            "    [int]$Start = $PID,",
            "    [string]$WorkerName = 'Runner.Worker.exe'",
            ")",
            *STRICT_HEADER,
        ]

    def test_it_is_ascii_lines_ending_in_a_newline_after_the_report(self) -> None:
        """ASCII, because Windows PowerShell reads a script with no byte order
        mark in the machine's ANSI code page."""
        script = runner_windows_job_render.hook_script()
        assert script.isascii()
        assert script.splitlines()[-1] == (
            '"runner job: $WorkerName $($process.ProcessId) and every process it starts run at '
            "BelowNormal, under the fleet's checks\""
        )
        assert script.endswith('checks"\n')


class TestTheLines:
    def test_the_parameters_carry_the_path_the_variable_and_the_hook_as_base64(self) -> None:
        path, variable, hook = runner_windows_job_render.parameter_lines()
        assert path == "    [string]$JobHookPath = 'C:\\fleet\\runner-hooks\\job-started.ps1',"
        assert variable == "    [string]$JobHookVariable = 'ACTIONS_RUNNER_HOOK_JOB_STARTED',"
        prefix, suffix = "    [string]$JobHook = '", "',"
        assert hook.startswith(prefix)
        assert hook.endswith(suffix)
        encoded = hook[len(prefix) : -len(suffix)]
        assert base64.b64decode(encoded).decode("utf-8") == runner_windows_job_render.hook_script()

    def test_a_written_variable_restarts_the_runners_and_a_written_hook_does_not(self) -> None:
        lines = runner_windows_job_render.INSTALL_LINES
        assert lines.count("    $restart = $true") == 1
        assert lines.index("    $restart = $true") > lines.index(
            '    "base: set the machine variable $JobHookVariable"'
        )

    def test_the_rows_are_the_hook_then_its_variable(self) -> None:
        assert runner_windows_job_render.expected_checks() == [
            ExpectedCheck(check_id="job-hook", reason=runner_windows_job_render.JOB_HOOK_REASON),
            ExpectedCheck(
                check_id="job-hook-variable", reason=runner_windows_job_render.JOB_HOOK_REASON
            ),
        ]
