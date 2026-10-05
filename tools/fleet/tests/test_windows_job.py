"""The build's kill-on-close job (fleet.core.windows_job, MCPs board task e40bca34).

``tests/pester/rendered-dialect-build.Tests.ps1`` executes the committed
build render: every in-process case enters the job, the first compiling its
type and the rest finding it, and one case runs the build as a node does and
shows the stop ending an orphan the stop's ``taskkill /T`` cannot reach.
This pins what that suite cannot see from outside: the limit is
kill-on-close and nothing lets a process break away, and the build records
itself before the compile and enters the job before it runs anything.
"""

from __future__ import annotations

from fleet.core import names
from fleet.core.powershell_text import add_type_lines
from fleet.core.windows_build import build_script
from fleet.core.windows_job import JOB_SOURCE, JOB_TYPE, enter_job_lines


def test_the_job_ends_its_processes_on_close_and_allows_no_breakaway() -> None:
    assert "const uint KillOnJobClose = 0x2000;" in JOB_SOURCE
    assert "limits.Basic.LimitFlags = KillOnJobClose;" in JOB_SOURCE
    assert "Breakaway" not in JOB_SOURCE
    assert "0x800" not in JOB_SOURCE
    assert "0x1000" not in JOB_SOURCE


def test_a_second_entry_answers_the_job_already_held() -> None:
    held = JOB_SOURCE.index("if (held != IntPtr.Zero) {")

    assert JOB_SOURCE[held:].index("return held;") < JOB_SOURCE[held:].index("CreateJobObjectW(")


def test_the_lines_compile_the_type_then_enter() -> None:
    assert enter_job_lines() == (
        *add_type_lines(JOB_TYPE, JOB_SOURCE),
        "[void][FleetNode.KillOnCloseJob]::Enter()",
    )


def test_the_build_records_itself_then_enters_the_job_before_it_runs_anything() -> None:
    lines = build_script(
        target="C:/fleet/stage/run-1",
        path="packages/x",
        workers=2,
        install=(),
        cache_root="C:/fleet/stage/cache",
        elevated=False,
        agent="opus-example-0929",
    ).splitlines()

    recorded = lines.index(f'$PID | Set-Content -LiteralPath "$Target/{names.PID_NAME}"')
    entered = lines.index("[void][FleetNode.KillOnCloseJob]::Enter()")
    assert lines[recorded - 1] == "$ErrorActionPreference = 'Stop'"
    assert lines[recorded + 1] == "if ($null -eq ('FleetNode.KillOnCloseJob' -as [type])) {"
    assert entered < lines.index("Set-Location -LiteralPath $Target")
    assert entered < next(index for index, line in enumerate(lines) if "& $Shell" in line)
