"""What a Windows node reports about a dispatch, including one whose build was ended.

MCPs board task 4e3afe4f, the Windows half of board task c8585623. A build
writes its exit status to the result file as its LAST act
(:func:`fleet.core.windows_build.build_script`), and the collector reads an
absent result as a run still going (:mod:`fleet.core.collect`). A Linux
build's unit writes the result itself when the build could not
(:mod:`fleet.core.linux_unit_end`); a Windows build had no such writer. Its
``powershell.exe`` runs as a scheduled task inside a kill-on-close job object
(:mod:`fleet.core.windows_job`), so when anything other than the build's last
line ends it (an out-of-memory termination, a ``taskkill`` by hand, a crash)
the whole tree is gone, nothing writes the result, and the runner renewed the
claim until the lease ran out and closed it ``LEASE_NOT_HELD`` with exit 124,
which reads as a suite that hung.

THE SCHEDULED TASK KNOWS. Task Scheduler waits on the process it started, so
the build's task reads ``Running`` exactly as long as the build's
``powershell.exe`` lives, and records that process's exit status as the
task's ``LastTaskResult`` once it ends. Measured on the hub on 2026-10-07,
with a build registered as :func:`fleet.core.windows_task.launch_script`
registers it (S4U, priority 4, ``IgnoreNew``) and read through Task
Scheduler's COM service: before the kill the task read state 4 (Running)
with ``0x00041301``; after ``taskkill /PID <build> /F`` it read state 3
(Ready) with ``0x00000001``, three times out of three, and after
``Stop-Process -Force`` state 3 with ``0xFFFFFFFF`` (-1). Read in a tight
loop, about 1,600 reads over the four kills, the state and the result
changed together at the first read 8 to 15 ms after the kill, with no read
showing one without the other.

SO THE RESULT SCRIPT READS THE TASK FIRST, THEN THE RESULT FILE. A task no
longer Running or Queued means the build's process has ended, and the build
writes its status before it ends, so a result file absent AFTER that read is
one the build will never write. Reading the file first would race: a build
writing its status and exiting between the two reads would be recorded as
ended without one. When the file is absent the script appends one
:data:`~fleet.core.linux_unit_end.UNIT_ENDED_MARKER` line to the transcript,
naming the task, its state and its last result, and writes the status to the
result file, which the same pass then reports like any other. The marker is
the Linux unit's own, so :func:`fleet.core.verdict.read_unit_end` carries
either onto the verdict line. The status is the task's ``LastTaskResult``,
the exit status of the process Task Scheduler started; a zero there, or a
task no longer registered, records
:data:`~fleet.core.linux_unit_end.UNEXPLAINED_EXIT_CODE`, since a build that
ran its recipe always wrote its status and a zero would close a check that
never did as passed. The status carries the moment it was recorded, the
first collect pass after the ending, which on a runner watching its runs
every few seconds is within seconds of it; Task Scheduler keeps the run's
start (``LastRunTime``) and not its end.

THE TASK IS VISIBLE TO THE READER. The runner that launched a build is the
runner that collects it, over ssh as the same account, and an elevated
build's task is no exception: measured on the hub on 2026-10-07, a task
registered at ``-RunLevel Highest`` from an elevated session and one
registered without it carried the same security descriptor, owned by the
account with ``(A;;FR;;;<account>)`` among its entries, so the account's
filtered token reads either. It is looked up by its exact name through the
COM service, as the stop looks it up (:func:`fleet.core.windows_task.stop_script`
says why the scheduled-task cmdlets are not used). A task no longer
registered has been deleted by the stop or the retire, after which nothing
runs the build.

The script is committed as a render under ``rendered/`` and executed by
``tests/pester/rendered-dialect-result.Tests.ps1`` against real scheduled
tasks, and by a ``host_windows`` case in ``tests/test_windows_result.py``
that launches a real build, ends its process and reads what the script
reports.
"""

from __future__ import annotations

from typing import Final

from fleet.core import names
from fleet.core.linux_unit_end import UNEXPLAINED_EXIT_CODE, UNIT_ENDED_MARKER
from fleet.core.powershell_text import STRICT_HEADER
from fleet.core.script_values import scriptable

#: Task Scheduler's ``TASK_STATE`` values, by number, as the line names them.
TASK_STATES: Final[dict[int, str]] = {
    0: "Unknown",
    1: "Disabled",
    2: "Queued",
    3: "Ready",
    4: "Running",
}

#: The states in which the task's process has not yet ended.
GOING_STATES: Final[tuple[int, ...]] = (2, 4)


def _state_table() -> str:
    """The PowerShell hashtable literal of :data:`TASK_STATES`.

    Returns:
        ``@{ 0 = 'Unknown'; ... }``.
    """
    pairs = "; ".join(f"{number} = '{name}'" for number, name in TASK_STATES.items())
    return "@{ " + pairs + " }"


def result_script(*, target: str, run_id: str) -> str:
    """Print the status and the epoch second it was written, or nothing.

    IT REPORTS *WHEN* AS WELL AS *WHAT*. Whether a run was safe is a
    question about whether its lease covered the whole of it, answerable
    only against the moment the build ended, which the node knows and nobody
    else does. Measured 2026-09-04, a run that finished three minutes inside
    its window was refused twenty minutes later for having been collected
    late. The epoch is computed by subtracting the Unix epoch from a UTC
    timestamp rather than with ``-UFormat %s``, which in PowerShell 5.1
    converts from LOCAL time and would put every node's answer out by its
    own offset.

    Args:
        target: Absolute remote directory holding the staged tree.
        run_id: The dispatch, which names the scheduled task its build runs as.

    Returns:
        The script's text. It prints nothing while the task is still
        Running or Queued and no result exists, and ``<status> <epoch>``
        otherwise, having first written the status of a build that ended
        without one.

    Raises:
        ValueError: When the target cannot be embedded verbatim.
    """
    going = ", ".join(str(state) for state in GOING_STATES)
    lines = [
        "param(",
        f"    [string]$Target = '{scriptable(target, label='target')}',",
        f"    [string]$TaskName = '{names.task_name(run_id)}'",
        ")",
        *STRICT_HEADER,
        f'$result = "$Target/{names.RESULT_NAME}"',
        f'$log = "{names.log_path("$Target")}"',
        "$scheduler = New-Object -ComObject Schedule.Service",
        "$scheduler.Connect()",
        "$task = @($scheduler.GetFolder('\\').GetTasks(1) | "
        "Where-Object { $_.Name -eq $TaskName })",
        f"$going = ($task.Count -gt 0) -and (@({going}) -contains [int]$task[0].State)",
        "if ((-not $going) -and (-not (Test-Path -LiteralPath $result))) {",
        f"    $code = {UNEXPLAINED_EXIT_CODE}",
        '    $how = "task $TaskName is no longer registered"',
        "    if ($task.Count -gt 0) {",
        f"        $states = {_state_table()}",
        "        $state = [int]$task[0].State",
        "        $last = [int]$task[0].LastTaskResult",
        "        if ($last -ne 0) {",
        "            $code = $last",
        "        }",
        "        $how = 'task {0} ended with Task Scheduler state {1} ({2}) and last result "
        "0x{3:X8}' -f $TaskName, $state, $states[$state], $last",
        "    }",
        f'    $line = "{UNIT_ENDED_MARKER}: $how before the build wrote its status; '
        'recorded exit $code"',
        '    [System.IO.File]::AppendAllText($log, "$line`r`n")',
        "    $code | Set-Content -LiteralPath $result",
        "}",
        "if (Test-Path -LiteralPath $result) {",
        "    $file = Get-Item -LiteralPath $result",
        "    $code = (Get-Content -Raw -LiteralPath $result).Trim()",
        "    $epoch = [int]($file.LastWriteTimeUtc - [datetime]'1970-01-01').TotalSeconds",
        '    "$code $epoch"',
        "}",
    ]
    return "\n".join(lines) + "\n"


__all__ = ["GOING_STATES", "TASK_STATES", "result_script"]
