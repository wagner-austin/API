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
"""

from __future__ import annotations

from fleet.core import names
from fleet.core.powershell_text import STRICT_HEADER, indented, system32_parameter
from fleet.core.script_values import scriptable
from fleet.core.windows_holders import ENDABLE_APP_TYPES, end_holders_lines

#: The prefix that exempts an absolute Windows path from MAX_PATH, as the
#: rendered PowerShell spells it (a single-quoted literal).
VERBATIM_PREFIX_LITERAL = "'\\\\?\\'"

#: The code the render throws when a directory survives its removal.
RETIRE_INCOMPLETE = "FLEET_RETIRE_INCOMPLETE"


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
        because it held the transcript, and nothing else.

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
    ]
    body = [
        "param(",
        ",\n".join(declared),
        ")",
        *STRICT_HEADER,
        "[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Retained)) | Out-Null",
        "if (Test-Path -LiteralPath $Log) {",
        *indented(end_holders_lines(path_variable="$Log"), depth=1),
        "    Move-Item -Force -LiteralPath $Log -Destination $Retained",
        "}",
        "foreach ($directory in @($Target, $Staging)) {",
        "    if (Test-Path -LiteralPath $directory) {",
        f"        $verbatim = {VERBATIM_PREFIX_LITERAL} + [IO.Path]::GetFullPath($directory)",
        "        & $Cmd /d /c rd /s /q $verbatim",
        "        if (Test-Path -LiteralPath $directory) {",
        f'            throw "{RETIRE_INCOMPLETE}: rd exited $LASTEXITCODE and left $directory"',
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


__all__ = ["RETIRE_INCOMPLETE", "VERBATIM_PREFIX_LITERAL", "retire_script"]
