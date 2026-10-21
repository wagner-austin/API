"""The job-started hook a Windows runner host gives every job, and the base lines that lay it.

MCPs board task db32d739. On Linux every runner job runs in
:data:`~fleet.contracts.runner_slice.CI_SLICE_NAME` under a CPU weight below
every other unit's, so a CI job yields the processor to the fleet's own
checks and takes all of it when nothing else wants it. loki's runners had
nothing of the kind: on 2026-10-20 the corvis-stick fleet check took 308 s
and 325 s of its 300 s budget on the two runs MCPs' powershell CI job ran
beside it on loki, and 183 to 259 s on the four runs it ran alone.

WINDOWS HAS THE SAME LEVER IN THE PRIORITY CLASS. A process created without
a priority flag by a BelowNormal parent is BelowNormal itself, so setting a
job's ``Runner.Worker`` to BelowNormal before its first step sets every step
it starts. The runner runs the script ``ACTIONS_RUNNER_HOOK_JOB_STARTED``
names before the job's steps, as a descendant of that worker, as the Linux
drop-in uses it (:mod:`fleet.core.runner_job_scope_render`). The hook walks
from its own process up its ancestors to the first ``Runner.Worker.exe``,
accepting a parent only when it was created before its child, so a process
id Windows reused after the real parent ended is never taken for it, and
refuses a walk that finds none rather than letting the job run at the
fleet's priority.

THE BASE LAYS IT ON EVERY HOST, as it holds long paths on every host: the
install writes the hook where :data:`JOB_HOOK_PATH` says and the variable
naming it, and a written variable restarts the running services as every
machine variable does (:mod:`fleet.core.runner_windows_base_render`); the
audit reads both back. The hook travels inside the install as base64, so
no quoting of its text is needed, and is committed under ``rendered/`` as
:data:`JOB_HOOK_STEM`, where the Pester suite executes it.
"""

from __future__ import annotations

import base64

from fleet.core.powershell_text import STRICT_HEADER
from fleet.core.runner_audit_types import ExpectedCheck

#: The committed hook's file stem under ``rendered/``.
JOB_HOOK_STEM = "runner-windows-job-started"

#: Where every Windows runner host keeps the hook.
JOB_HOOK_PATH = "C:\\fleet\\runner-hooks\\job-started.ps1"

#: The runner's job-started hook variable.
JOB_HOOK_VARIABLE = "ACTIONS_RUNNER_HOOK_JOB_STARTED"

#: The process a job runs in, whose priority class its steps inherit.
WORKER_NAME = "Runner.Worker.exe"

#: The hook file's row id.
JOB_HOOK_CHECK_ID = "job-hook"

#: The hook variable's row id.
JOB_HOOK_VARIABLE_CHECK_ID = "job-hook-variable"

#: Why both rows matter.
JOB_HOOK_REASON = (
    "a CI job at the fleet's own priority takes the processor from the fleet's checks: "
    "corvis-stick's check overran its 300 s budget on loki beside MCPs' powershell job"
)

#: The hook, one line per element. ``$Start`` and ``$WorkerName`` are what
#: the Pester suite moves; the runner passes neither.
HOOK_LINES: tuple[str, ...] = (
    "param(",
    "    [int]$Start = $PID,",
    f"    [string]$WorkerName = '{WORKER_NAME}'",
    ")",
    *STRICT_HEADER,
    '$process = Get-CimInstance Win32_Process -Filter "ProcessId = $Start"',
    "do {",
    "    $child = $process",
    "    $process = Get-CimInstance Win32_Process -Filter "
    '"ProcessId = $($child.ParentProcessId) AND ProcessId <> 0" |',
    "        Where-Object { $_.CreationDate -lt $child.CreationDate }",
    "} while ($null -ne $process -and $process.Name -ne $WorkerName)",
    "if ($null -eq $process) {",
    '    throw "RUNNER_JOB_PRIORITY_FAILED: no $WorkerName is among the ancestors of process '
    "$Start, so the job would run at the fleet's own priority\"",
    "}",
    "(Get-Process -Id $process.ProcessId).PriorityClass = "
    "[System.Diagnostics.ProcessPriorityClass]::BelowNormal",
    '"runner job: $WorkerName $($process.ProcessId) and every process it starts run at '
    "BelowNormal, under the fleet's checks\"",
)


def hook_script() -> str:
    """The hook as a host receives it and ``rendered/`` commits it.

    Returns:
        The script.
    """
    return "\n".join(HOOK_LINES) + "\n"


def parameter_lines() -> list[str]:
    """The install's and the audit's hook parameters, each ending in a comma.

    Returns:
        The hook's path, its variable's name and the hook itself as base64.
    """
    encoded = base64.b64encode(hook_script().encode("utf-8")).decode("ascii")
    return [
        f"    [string]$JobHookPath = '{JOB_HOOK_PATH}',",
        f"    [string]$JobHookVariable = '{JOB_HOOK_VARIABLE}',",
        f"    [string]$JobHook = '{encoded}',",
    ]


#: The install's hook lines, after the base's, where ``$restart`` is set.
INSTALL_LINES: tuple[str, ...] = (
    "$jobHookText = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($JobHook))",
    "if (-not (Test-Path -LiteralPath $JobHookPath) -or "
    "[IO.File]::ReadAllText($JobHookPath) -cne $jobHookText) {",
    "    [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($JobHookPath))",
    "    [IO.File]::WriteAllText($JobHookPath, $jobHookText)",
    '    "base: wrote the job-started hook $JobHookPath"',
    "}",
    "if ([Environment]::GetEnvironmentVariable($JobHookVariable, $VariableScope) "
    "-cne $JobHookPath) {",
    "    [Environment]::SetEnvironmentVariable($JobHookVariable, $JobHookPath, $VariableScope)",
    '    "base: set the machine variable $JobHookVariable"',
    "    $restart = $true",
    "}",
)

#: The audit's hook rows, after the base's, in :func:`expected_checks` order.
AUDIT_LINES: tuple[str, ...] = (
    "$heldHook = ''",
    "if (Test-Path -LiteralPath $JobHookPath) {",
    "    $heldHook = [IO.File]::ReadAllText($JobHookPath)",
    "}",
    "if ($heldHook -ceq [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($JobHook))) {",
    f"    'CHECK {JOB_HOOK_CHECK_ID} OK'",
    "} else {",
    f'    "CHECK {JOB_HOOK_CHECK_ID} DRIFT $JobHookPath does not hold the job-started hook"',
    "}",
    "$heldHookVariable = [string][Environment]::GetEnvironmentVariable($JobHookVariable, "
    "$VariableScope)",
    "if ($heldHookVariable -ceq $JobHookPath) {",
    f"    'CHECK {JOB_HOOK_VARIABLE_CHECK_ID} OK'",
    "} else {",
    f'    "CHECK {JOB_HOOK_VARIABLE_CHECK_ID} DRIFT $JobHookVariable holds: $heldHookVariable"',
    "}",
)


def expected_checks() -> list[ExpectedCheck]:
    """The hook's rows the audit prints, in order.

    Returns:
        The hook file's row, then its variable's.
    """
    return [
        ExpectedCheck(check_id=JOB_HOOK_CHECK_ID, reason=JOB_HOOK_REASON),
        ExpectedCheck(check_id=JOB_HOOK_VARIABLE_CHECK_ID, reason=JOB_HOOK_REASON),
    ]


__all__ = [
    "AUDIT_LINES",
    "HOOK_LINES",
    "INSTALL_LINES",
    "JOB_HOOK_CHECK_ID",
    "JOB_HOOK_PATH",
    "JOB_HOOK_REASON",
    "JOB_HOOK_STEM",
    "JOB_HOOK_VARIABLE",
    "JOB_HOOK_VARIABLE_CHECK_ID",
    "WORKER_NAME",
    "expected_checks",
    "hook_script",
    "parameter_lines",
]
