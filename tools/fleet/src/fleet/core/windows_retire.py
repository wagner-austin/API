"""The script a Windows node runs to retire one settled dispatch.

Split out of :mod:`fleet.core.dialect_windows` when the directories it
removes stopped going through ``Remove-Item`` (MCPs board task 74b13c20),
as :mod:`fleet.core.windows_build` did for the build. The script is
committed as a render under ``rendered/`` and executed by
``tests/pester/rendered-dialect-state.Tests.ps1``.

WHY ``rd /s /q`` ON THE VERBATIM PATH. Windows PowerShell 5.1 is not
long-path aware, so on a node that does not enable long paths its
``Remove-Item`` cannot reach a file past 260 characters. Measured
2026-10-04 on loki: ``MCPs/packages/maketools`` at d0118c27 passed at
14:17:19Z, its suite having left trees past that limit in pytest's
directory inside the export, and the retire stopped on ``Could not find a
part of the path`` on every tick after, so fleet job 5cb3dbe5 never
closed and every tick of loki's runner exited 1. ``cmd.exe``'s ``rd``
takes a ``\\\\?\\`` path, which lifts the limit; measured the same day it
deletes read-only files and a tree past 300 characters, and removes a
junction inside the tree without entering it, so a link pointing out of
the export never empties its target. ``rd`` can report success while
leaving something behind, so the directory is looked for again afterwards
and its survival is an error by name.

A HELD TRANSCRIPT IS RELEASED, NOT FAILED ON (MCPs board task e40bca34).
Before the move, every leftover process of the run still holding the
transcript is named and ended (:mod:`fleet.core.windows_holders`), so one
orphan can no longer stop every tick of its node's runner, which it did on
sedona for 26 hours.

A HELD RUN FILE IS WAITED FOR, AND ITS HOLDER NAMED (MCPs board task
a703ec42). ``rd`` leaves every file another process holds and still exits
0. On loki the retire of MCPs-packages-session-audit-loki-1792577608 at
2026-10-21T10:15Z stopped on the staging directory's ``tree.tgz``, held
about a minute after the archive was written by a process the failure did
not name, and since a retire that failed is never sent again
(:mod:`fleet.core.retire_owed`) the directory stayed 21 hours, until the
hub suite's stage-roots case found it (fleet job c612689e). The holder had
let go by the time anyone looked. So a directory ``rd`` leaves is removed
again every second for up to :data:`HELD_SECONDS`, the first refusal prints
one :data:`RETIRE_HELD` line naming who holds what is left, and a directory
that outlasts the bound fails naming them again. Nothing is ended: unlike
the transcript, a file in the directories is reachable by processes that
are not the run's.
"""

from __future__ import annotations

from typing import Final

from fleet.core import names
from fleet.core.powershell_text import STRICT_HEADER, indented, system32_parameter
from fleet.core.script_values import scriptable
from fleet.core.windows_holders import (
    ENDABLE_APP_TYPES,
    RESTART_MANAGER_PATH_LIMIT,
    compile_lines,
    end_holders_lines,
    named_holders_lines,
)

#: The prefix that exempts an absolute Windows path from MAX_PATH, as the
#: rendered PowerShell spells it (a single-quoted literal).
VERBATIM_PREFIX_LITERAL = "'\\\\?\\'"

#: The code the render throws when a directory survives its removal.
RETIRE_INCOMPLETE = "FLEET_RETIRE_INCOMPLETE"

#: The prefix of the line the render prints when ``rd`` first leaves a
#: directory, naming who holds what it left.
RETIRE_HELD: Final = "FLEET_RETIRE_HELD"

#: How long, in seconds from the first removal, the render keeps removing a
#: directory ``rd`` left. The whole retire is one ssh command under
#: :data:`fleet.core.remote.SSH_TIMEOUT_SECONDS` (120), past which the hub
#: reads the node as unreachable, so the wait keeps three quarters of that
#: for the compile, the transcript and the scripts.
HELD_SECONDS: Final = 30

#: The render's function describing what ``rd`` left in one directory.
DESCRIBE_FUNCTION: Final = "Get-HolderDescription"

#: The render's function running ``rd`` over one verbatim path and answering
#: its exit code, which each message reports: its one call site keeps the
#: exit read beside the native command, as MCPs' PowerShell harness requires
#: (``PS-NATIVE-EXIT-UNCHECKED``), wherever the retire calls it.
RD_FUNCTION: Final = "Invoke-VerbatimRd"


def _describe_function_lines() -> tuple[str, ...]:
    """The render's function saying what ``rd`` left in one directory and who holds it.

    The files left are listed through the verbatim path, which reaches past
    MAX_PATH, and Restart Manager is asked about those whose plain path it
    registers (:data:`fleet.core.windows_holders.RESTART_MANAGER_PATH_LIMIT`);
    the others are counted, so a holder it could not be asked about is
    never read as no holder.

    Returns:
        The function's lines. It takes the directory's verbatim path and
        answers one string: how many files are left, how many of them are
        too long to ask about, and the holders' names, or that Restart
        Manager names none.
    """
    return (
        f"function {DESCRIBE_FUNCTION} {{",
        "    param([string]$Verbatim)",
        "    $left = @([IO.Directory]::GetFiles($Verbatim, '*', 'AllDirectories') |",
        f"        ForEach-Object {{ $_.Substring({VERBATIM_PREFIX_LITERAL}.Length) }})",
        f"    $asked = @($left | Where-Object {{ $_.Length -le {RESTART_MANAGER_PATH_LIMIT} }})",
        *indented(named_holders_lines(paths_variable="$asked"), depth=1),
        "    $heldBy = 'no process Restart Manager names'",
        "    if ($holders.Count -gt 0) {",
        "        $heldBy = @($holders | ForEach-Object { $_.Named }) -join '; '",
        "    }",
        '    return "$($left.Count) files left, $($left.Count - $asked.Count) of them too '
        'long for Restart Manager to ask about, held by $heldBy"',
        "}",
    )


def retire_script(*, target: str, retained: str, scripts: tuple[str, ...], task: str) -> str:
    """Keep a settled run's transcript, then remove its directories, scripts and task.

    Every location is a parameter defaulting to the rendered path, so a node
    runs it with no arguments and the Pester suite over its committed render
    points it at a directory it laid out. Removing its own file is safe:
    ``powershell -File`` has read the whole script before the first
    statement runs.

    THE RUN'S TASK GOES TOO (MCPs board task a146760d). A build's task stays
    registered after the build exits, and only a stop deleted it, so every
    run that finished left one: 153 on sedona and 293 on serendipity on
    2026-09-30. It is looked up and deleted by its exact name through Task
    Scheduler's COM service, for the reason the stop script gives
    (:func:`fleet.core.windows_task.stop_script`): the cmdlets fail while
    another task is deleted.

    Args:
        target: The dispatch's absolute remote directory; its staging
            directory beside it (:func:`fleet.core.names.staging_directory`)
            goes with it.
        retained: Where its transcript is kept.
        scripts: The scripts it left under the stage root.
        task: The run's scheduled task, in the root folder.

    Returns:
        The script's text. Each removal is guarded by ``Test-Path`` or by the
        task being listed, because a retire that failed part way is run again
        by the next tick and meets exactly what is already gone. It prints
        one ``FLEET_RETIRE_HOLDER_ENDED`` line for each process it ended
        because it held the transcript, one :data:`RETIRE_HELD` line for
        each directory ``rd`` first left, and nothing else. ``$HeldSeconds``
        is a parameter defaulting to :data:`HELD_SECONDS`, so the Pester
        suite can fail a held directory without waiting it out.

    Raises:
        ValueError: When a path cannot be embedded verbatim.
    """
    # One plain string parameter per script, gathered in the body: a list as
    # a parameter's default is an expression the coverage harness counts as a
    # command, reached only by a run that takes the default.
    parameters = [f"$Script{index}" for index in range(len(scripts))]
    staging = names.staging_directory(target)
    declared = [
        f"    [string]$Target = '{scriptable(target, label='target')}'",
        f"    [string]$Staging = '{scriptable(staging, label='staging')}'",
        f"    [string]$Log = '{scriptable(names.log_path(target), label='log')}'",
        f"    [string]$Retained = '{scriptable(retained, label='retained')}'",
        f"    [string]$TaskName = '{scriptable(task, label='task')}'",
        *(
            f"    [string]{parameter} = '{scriptable(script, label='script')}'"
            for parameter, script in zip(parameters, scripts, strict=True)
        ),
        f"    {system32_parameter('Cmd', 'cmd.exe')}",
        f"    [int[]]$EndableTypes = @({', '.join(str(kind) for kind in ENDABLE_APP_TYPES)})",
        f"    [int]$HeldSeconds = {HELD_SECONDS}",
    ]
    body = [
        "param(",
        ",\n".join(declared),
        ")",
        *STRICT_HEADER,
        *compile_lines(),
        *_describe_function_lines(),
        "[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Retained)) | Out-Null",
        "if (Test-Path -LiteralPath $Log) {",
        *indented(end_holders_lines(path_variable="$Log"), depth=1),
        "    Move-Item -Force -LiteralPath $Log -Destination $Retained",
        "}",
        f"function {RD_FUNCTION} {{",
        "    param([string]$Verbatim)",
        "    & $Cmd /d /c rd /s /q $Verbatim",
        "    return $LASTEXITCODE",
        "}",
        "$removing = [Diagnostics.Stopwatch]::StartNew()",
        "foreach ($directory in @($Target, $Staging)) {",
        "    if (Test-Path -LiteralPath $directory) {",
        f"        $verbatim = {VERBATIM_PREFIX_LITERAL} + [IO.Path]::GetFullPath($directory)",
        f"        $exited = {RD_FUNCTION} $verbatim",
        "        if (Test-Path -LiteralPath $directory) {",
        f'            Write-Output "{RETIRE_HELD}: rd exited $exited and left $directory, '
        f'$({DESCRIBE_FUNCTION} $verbatim); removing it again for up to $HeldSeconds s"',
        "        }",
        "        while ((Test-Path -LiteralPath $directory) -and "
        "$removing.Elapsed.TotalSeconds -lt $HeldSeconds) {",
        "            Start-Sleep -Seconds 1",
        f"            $exited = {RD_FUNCTION} $verbatim",
        "        }",
        "        if (Test-Path -LiteralPath $directory) {",
        f'            throw "{RETIRE_INCOMPLETE}: rd exited $exited and left $directory '
        f'after $HeldSeconds s, $({DESCRIBE_FUNCTION} $verbatim)"',
        "        }",
        "    }",
        "}",
        f"foreach ($script in @({', '.join(parameters)})) {{",
        "    if (Test-Path -LiteralPath $script) {",
        "        Remove-Item -Force -LiteralPath $script",
        "    }",
        "}",
        "$scheduler = New-Object -ComObject Schedule.Service",
        "$scheduler.Connect()",
        "$root = $scheduler.GetFolder('\\')",
        "if (@($root.GetTasks(1) | Where-Object { $_.Name -eq $TaskName }).Count -gt 0) {",
        "    $root.DeleteTask($TaskName, 0)",
        "}",
    ]
    return "\n".join(body) + "\n"


__all__ = [
    "DESCRIBE_FUNCTION",
    "HELD_SECONDS",
    "RD_FUNCTION",
    "RETIRE_HELD",
    "RETIRE_INCOMPLETE",
    "VERBATIM_PREFIX_LITERAL",
    "retire_script",
]
