"""The scheduled task that runs a Windows build: registered, started, stopped.

Split out of :mod:`fleet.core.dialect_windows` (MCPs board task d69786fa)
when its scripts took parameters and strict headers and the file neared the
600-line ceiling; :class:`~fleet.core.dialect_windows.WindowsDialect` hands
each of its acts here. Both scripts are committed as renders under
``rendered/`` and executed by ``tests/pester/rendered-dialect-task.Tests.ps1``
against real, disposable scheduled tasks and real process trees, because a
task that registers but never starts, or a stop that leaves a child running,
is exactly what reading the text never caught.

WHY TASK SCHEDULER AND NOT AN SSH CHILD. Windows OpenSSH assigns the
session's process tree to a job object precisely so the tree dies when the
connection ends, and a process cannot be moved out of a job object once it
is in one (``memory/reference_long_runs_need_task_scheduler.md``).
"""

from __future__ import annotations

from typing import Final

from fleet.core import names
from fleet.core.powershell_text import STRICT_HEADER, system32_parameter
from fleet.core.script_values import scriptable

#: How long the node waits for a started build to record itself, in seconds.
#:
#: Generous because it bounds a Task Scheduler round trip and a PowerShell
#: start, not any work: the build's first act is writing its process id, so
#: a build that starts and fails in its first second has still recorded it.
LAUNCH_TIMEOUT_SECONDS: Final = 30


def launch_script(*, target: str, run_id: str) -> str:
    """Register a scheduled task for the build, start it, and prove it began.

    ``-AllowStartIfOnBatteries`` and ``-DontStopIfGoingOnBatteries`` are not
    optional and their defaults are the wrong way round for this fleet: two
    of the three Windows nodes are laptops, so a dispatch to an unplugged
    sedona would register a task that never runs, or would have a running
    suite killed the moment somebody unplugged it. ``-Priority 4`` because 7,
    the default, sets LOW I/O and a run that inherits it crawls in a way that
    reads as a slow node.

    IT WAITS FOR THE BUILD'S OWN PROCESS ID, not for the task's status.
    ``Start-ScheduledTask`` reports a refusal as a NON-TERMINATING error:
    PowerShell prints it, exits 0, and the dispatch records a run that does
    not exist (measured 2026-09-04, a ``running`` ledger row for a task whose
    ``LastRunTime`` was still the 1999 sentinel). Until 2026-09-27 the script
    waited for ``LastTaskResult`` to leave ``SCHED_S_TASK_HAS_NOT_RUN``, which
    a task that fails to START also leaves, with its launch error, and so read
    as launched. ``build.ps1`` writes :data:`~fleet.core.names.PID_NAME` as
    its first act (:meth:`~fleet.core.dialect_windows.WindowsDialect.build_script`),
    so the file existing is the build running, and nothing less is.

    Args:
        target: Absolute remote directory holding the staged tree; a fresh
            directory per dispatch, so no earlier run's id is in it.
        run_id: The dispatch, which names its own task.

    Returns:
        The script's text. It prints ``launched`` once the build has
        recorded itself, and throws naming the task when it has not within
        ``$LaunchSeconds``.

    Raises:
        ValueError: When the target cannot be embedded verbatim.
    """
    lines = [
        "param(",
        f"    [string]$Target = '{scriptable(target, label='target')}',",
        f"    [string]$TaskName = '{names.task_name(run_id)}',",
        f"    [int]$LaunchSeconds = {LAUNCH_TIMEOUT_SECONDS}",
        ")",
        *STRICT_HEADER,
        f'$build = "$Target/{names.BUILD_STEM}.ps1"',
        f'$recorded = "$Target/{names.PID_NAME}"',
        "$action = New-ScheduledTaskAction -Execute 'powershell.exe' "
        '-Argument "-NoProfile -ExecutionPolicy Bypass -File `"$build`""',
        "$settings = New-ScheduledTaskSettingsSet -Priority 4 "
        "-ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew "
        "-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries",
        "$principal = New-ScheduledTaskPrincipal "
        "-UserId ([Security.Principal.WindowsIdentity]::GetCurrent().User.Value) -LogonType S4U",
        "Register-ScheduledTask -TaskName $TaskName -Action $action -Settings $settings "
        "-Principal $principal -Force | Out-Null",
        "Start-ScheduledTask -TaskName $TaskName",
        "$deadline = (Get-Date).AddSeconds($LaunchSeconds)",
        "while (-not (Test-Path -LiteralPath $recorded) -and ((Get-Date) -lt $deadline)) {",
        "    Start-Sleep -Milliseconds 500",
        "}",
        "if (-not (Test-Path -LiteralPath $recorded)) {",
        '    throw "FLEET_LAUNCH_NOT_STARTED: $TaskName registered, but its build had not '
        'recorded itself after $LaunchSeconds s"',
        "}",
        "Write-Output 'launched'",
    ]
    return "\n".join(lines) + "\n"


def stop_script(*, target: str, run_id: str) -> str:
    """End the build's process tree, then stop and unregister the task.

    STOPPING THE TASK IS NOT STOPPING THE BUILD. ``Stop-ScheduledTask`` ends
    the process the task started and leaves its children running: measured on
    sedona 2026-09-23, a probe task whose ``build.ps1`` ran a native child read
    parent alive=False, child alive=True afterwards. So the tree is ended
    first, by the process id the build recorded as its first act, with
    ``taskkill /T /F``, which on the same probe ended the parent, its child
    and its grandchild.

    THE ID IS CHECKED BEFORE ANYTHING IS KILLED. A recorded id outlives its
    process, and Windows reuses ids, so the kill happens only when the process
    holding that id right now is running this dispatch's own ``build.ps1``,
    read off its command line. It kills by id and never by name or pattern,
    which is the fleet's one rule about killing.

    THE TASK IS LOOKED UP BY ITS EXACT NAME before it is stopped, where the
    script once passed ``-ErrorAction SilentlyContinue`` to both cmdlets: that
    read every scheduler failure, not only a task already gone, as nothing to
    do. The lookup and the delete go through Task Scheduler's COM service,
    not ``Get-ScheduledTask`` and ``Unregister-ScheduledTask``, because those
    read every task's definition and fail with "The system cannot find the
    file specified" when another task is deleted while they run. Measured on
    the hub 2026-09-27 against a process registering and deleting tasks: 3 of
    150 ``Get-ScheduledTask`` listings, 2 of 150 name-filtered CIM queries
    and 172 of 178 ``Unregister-ScheduledTask -TaskName`` calls failed, while
    0 of 1,873 COM name listings did, and ``Register-``, ``Start-`` and
    ``Stop-ScheduledTask`` by name failed 0 of 178 times each.

    Args:
        target: Absolute remote directory holding the staged tree and the
            build's recorded process id.
        run_id: The dispatch.

    Returns:
        The script's text. ``taskkill`` exiting non-zero on a verified
        process throws ``FLEET_STOP_KILL_FAILED``, so a stop that ended
        nothing is never reported as one that did.

    Raises:
        ValueError: When the target cannot be embedded verbatim.
    """
    lines = [
        "param(",
        f"    [string]$Target = '{scriptable(target, label='target')}',",
        f"    [string]$TaskName = '{names.task_name(run_id)}',",
        "    " + system32_parameter("Taskkill", "taskkill.exe"),
        ")",
        *STRICT_HEADER,
        f'$recorded = "$Target/{names.PID_NAME}"',
        f'$build = "$Target/{names.BUILD_STEM}.ps1"',
        "if (Test-Path -LiteralPath $recorded) {",
        "    $buildPid = [int](Get-Content -Raw -LiteralPath $recorded).Trim()",
        '    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$buildPid"',
        '    if (($null -ne $process) -and ($process.CommandLine -like "*$build*")) {',
        "        & $Taskkill /PID $buildPid /T /F",
        "        if ($LASTEXITCODE -ne 0) {",
        '            throw "FLEET_STOP_KILL_FAILED: taskkill of $buildPid exited $LASTEXITCODE"',
        "        }",
        "    }",
        "}",
        "$scheduler = New-Object -ComObject Schedule.Service",
        "$scheduler.Connect()",
        "$root = $scheduler.GetFolder('\\')",
        "$task = @($root.GetTasks(1) | Where-Object { $_.Name -eq $TaskName })",
        "if ($task.Count -gt 0) {",
        "    Stop-ScheduledTask -TaskName $TaskName",
        "    $root.DeleteTask($TaskName, 0)",
        "}",
        'Write-Output "stopped $TaskName"',
    ]
    return "\n".join(lines) + "\n"


__all__ = ["LAUNCH_TIMEOUT_SECONDS", "launch_script", "stop_script"]
