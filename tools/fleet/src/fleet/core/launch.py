"""What a node is actually told to do, and how it is told.

SPLIT OUT OF :mod:`fleet.core.dispatch` when that file crossed the 600-line
ceiling, and split along the seam the guard's message names: by ROLE. This
module renders the four scripts a dispatch sends to a node -- build, register,
result -- and names the scheduled task they share. ``dispatch`` keeps the
lifecycle: the lease, the ledger rows, the feed events, and the order the
three happen in.

THE ONE RULE, AND IT COST A DISPATCH TO RELEARN.
:mod:`fleet.core.remote` forbids interpolating a remote command, because
quotes do not survive the trip through the local shell, ssh and ``cmd`` into
``powershell``. That rule was applied to ssh and then broken one layer further
in: the first version of this code passed the whole build to
``New-ScheduledTaskAction -Argument '-Command "cd ''{path}''; ..."'``, and
PowerShell ended the single-quoted argument at the first inner quote. The
registered task carried ``-Command "cd`` as its arguments and the remaining
two hundred characters as its WORKING DIRECTORY -- a path that does not exist,
so the task could not be started at all.

So the build is its OWN FILE, sent and run by path, and the registration
interpolates one path and no code.
"""

from __future__ import annotations

from fleet.contracts.project import MAKE_TARGET

#: Where a dispatch's result is left on the node, under its own directory.
RESULT_NAME = "result.txt"

#: What the build's own script is called under a dispatch's directory.
#:
#: THE BUILD IS A SEPARATE FILE FROM THE THING THAT SCHEDULES IT, and that is
#: the fix for a real bug rather than a tidiness preference. The first version
#: passed the whole build as ``-Argument '-Command "cd ''{path}''; ..."'``, and
#: PowerShell ended the single-quoted argument at the first inner quote: the
#: registered task carried ``-Command "cd`` as its arguments and the remaining
#: two hundred characters as its WORKING DIRECTORY. Measured on sedona
#: 2026-09-04, the resulting task could not be started at all -- ``Element not
#: found``, because that working directory does not exist.
#:
#: Sending the script and naming it by path is the same rule
#: :mod:`fleet.core.remote` follows for ssh, applied one layer further in. The
#: registration then interpolates one path and no code.
BUILD_SCRIPT_NAME = "build.ps1"

#: What the script that registers and starts the task is called.
REGISTER_SCRIPT_NAME = "register.ps1"

#: Task Scheduler's ``SCHED_S_TASK_HAS_NOT_RUN``, 0x00041303.
#:
#: The status a registered task reports until it has run once. It is the
#: signal :func:`register_script` waits to stop seeing, because
#: ``Start-ScheduledTask`` reports a failure to start as a NON-TERMINATING
#: error: PowerShell prints it, exits 0, and the dispatch records a run that
#: does not exist. Measured 2026-09-04 -- the ledger said ``running`` for a
#: task whose ``LastRunTime`` was still the 1999 sentinel.
TASK_HAS_NOT_RUN = 267011

#: How long the node waits for a started task to leave that state.
#:
#: Generous because it is bounding a Task Scheduler round trip and not any
#: work: the task only has to BEGIN. A build that starts and fails in the
#: first second still leaves this state, so the wait ends on the first status
#: change rather than on success.
LAUNCH_TIMEOUT_SECONDS = 30


def build_script(*, target: str, project: str, workers: int) -> str:
    """Render the script that runs the suite, which is all it does.

    It knows nothing about scheduling. That separation is what keeps the
    registration free of nested quoting -- see :data:`BUILD_SCRIPT_NAME`.

    ``$LASTEXITCODE`` rather than ``$?`` because the recipe is a native
    program: PowerShell sets ``$?`` false whenever a native command writes to
    a redirected stderr, which ``make`` does routinely on a passing run.

    Args:
        target: Absolute remote directory holding the staged tree.
        project: Repo-relative project path.
        workers: Test workers the capacity check granted.

    Returns:
        The script's text. Its last act writes the recipe's exit status to
        :data:`RESULT_NAME`, which is what
        :func:`~fleet.core.collect.poll_result` later reads -- so the file's
        ABSENCE means the run is still going and its presence means it is
        over, with no third state to disambiguate.
    """
    return (
        f"$ErrorActionPreference = 'Continue'\n"
        f"Set-Location -LiteralPath '{target}/{project}'\n"
        f"$env:PYTEST_XDIST_AUTO_NUM_WORKERS = '{workers}'\n"
        f"make {MAKE_TARGET} *> '{target}/{RESULT_NAME}.log'\n"
        f"$LASTEXITCODE | Set-Content -LiteralPath '{target}/{RESULT_NAME}'\n"
    )


def register_script(*, target: str, run_id: str) -> str:
    """Render the script that schedules the build and proves it started.

    ``-AllowStartIfOnBatteries`` and ``-DontStopIfGoingOnBatteries`` are not
    optional here and their defaults are the wrong way round for this fleet.
    ``New-ScheduledTaskSettingsSet`` defaults both battery settings to
    refusing, and two of the three nodes are laptops -- so a dispatch to an
    unplugged sedona would register a task that never runs, or would have a
    running suite killed the moment somebody unplugged it, in both cases
    reporting nothing.

    Args:
        target: Absolute remote directory holding the staged tree.
        run_id: The dispatch, which names its own task.

    Returns:
        The script's text. It registers the task, starts it, and then WAITS
        for the task to leave :data:`TASK_HAS_NOT_RUN` before saying so --
        because ``Start-ScheduledTask`` reports a refusal as a non-terminating
        error that would otherwise exit 0 and be recorded as a launch.
    """
    task = task_name(run_id)
    return (
        f"$ErrorActionPreference = 'Stop'\n"
        f"$action = New-ScheduledTaskAction -Execute 'powershell.exe' "
        f"-Argument '-NoProfile -ExecutionPolicy Bypass -File \"{target}/{BUILD_SCRIPT_NAME}\"'\n"
        f"$settings = New-ScheduledTaskSettingsSet -Priority 4 "
        f"-ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew "
        f"-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries\n"
        f"$principal = New-ScheduledTaskPrincipal "
        f"-UserId ([Security.Principal.WindowsIdentity]::GetCurrent().User.Value) "
        f"-LogonType S4U\n"
        f"Register-ScheduledTask -TaskName '{task}' -Action $action "
        f"-Settings $settings -Principal $principal -Force | Out-Null\n"
        f"Start-ScheduledTask -TaskName '{task}'\n"
        f"$deadline = (Get-Date).AddSeconds({LAUNCH_TIMEOUT_SECONDS})\n"
        f"while ((Get-Date) -lt $deadline) {{\n"
        f"  if ((Get-ScheduledTaskInfo -TaskName '{task}').LastTaskResult "
        f"-ne {TASK_HAS_NOT_RUN}) {{ Write-Output 'launched'; exit 0 }}\n"
        f"  Start-Sleep -Milliseconds 500\n"
        f"}}\n"
        f'throw "{task} registered but has not run after {LAUNCH_TIMEOUT_SECONDS}s"\n'
    )


def task_name(run_id: str) -> str:
    """Name the scheduled task a dispatch owns.

    Derived from the run id rather than from the stage path, so the name a
    dispatch registers and the name :mod:`fleet.cli.cancel` stops are the same
    string produced by the same function. They were separately spelled before,
    which is a rename away from a cancel that silently stops nothing.

    Args:
        run_id: The dispatch.

    Returns:
        The task's name.
    """
    return f"fleet-{run_id}"


def result_script(target: str) -> str:
    """Render the script that reports whether a dispatch has finished.

    IT REPORTS *WHEN* AS WELL AS *WHAT*, and the timestamp is not decoration.
    Whether a run was safe is a question about whether its lease covered the
    whole of it, and that can only be answered against the moment the build
    ended -- which the node knows and nobody else does. Asking only for the
    status forces the reader to substitute "is a lease held right now", which
    is a question about how promptly somebody collected: measured 2026-09-04,
    a run that finished three minutes inside its window was refused twenty
    minutes later for having been collected late.

    Args:
        target: Absolute remote directory holding the staged tree.

    Returns:
        The script's text. It prints the exit status and the epoch second the
        result was written, space-separated -- or nothing at all while the run
        is still going. Absence is the signal, so a run that has not written
        its status cannot be mistaken for one that exited zero.

        The epoch is computed by subtracting the Unix epoch from a UTC
        timestamp rather than with ``-UFormat %s``, which in PowerShell 5.1
        converts from LOCAL time and would put every node's answer out by its
        own offset.
    """
    return (
        f"if (Test-Path -LiteralPath '{target}/{RESULT_NAME}') {{\n"
        f"  $file = Get-Item -LiteralPath '{target}/{RESULT_NAME}'\n"
        f"  $code = (Get-Content -Raw -LiteralPath '{target}/{RESULT_NAME}').Trim()\n"
        f"  $epoch = [int]($file.LastWriteTimeUtc - [datetime]'1970-01-01').TotalSeconds\n"
        f'  "$code $epoch"\n'
        f"}}\n"
    )


__all__ = [
    "BUILD_SCRIPT_NAME",
    "LAUNCH_TIMEOUT_SECONDS",
    "REGISTER_SCRIPT_NAME",
    "RESULT_NAME",
    "TASK_HAS_NOT_RUN",
    "build_script",
    "register_script",
    "result_script",
    "task_name",
]
