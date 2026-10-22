"""The Windows retire's text (fleet.core.windows_retire, MCPs board task 74b13c20).

``tests/pester/rendered-dialect-state.Tests.ps1`` executes the committed
render, a tree past MAX_PATH, an inert ``cmd`` and a held transcript
included; this pins what that suite cannot see from outside: the
directories go through ``rd`` on the verbatim path through the parameter it
can replace, nothing in the script calls ``Remove-Item -Recurse`` again, the
transcript's holders are ended before it is moved (MCPs board task
e40bca34), and a directory rd leaves is removed again for a bound that the
ssh deadline leaves room for, its holders named (MCPs board task a703ec42).
"""

from __future__ import annotations

import pytest

from fleet.core.powershell_text import indented
from fleet.core.remote import SSH_TIMEOUT_SECONDS
from fleet.core.windows_holders import (
    RESTART_MANAGER_PATH_LIMIT,
    compile_lines,
    end_holders_lines,
    named_holders_lines,
)
from fleet.core.windows_retire import (
    DESCRIBE_FUNCTION,
    HELD_SECONDS,
    RD_FUNCTION,
    RETIRE_HELD,
    RETIRE_INCOMPLETE,
    VERBATIM_PREFIX_LITERAL,
    retire_script,
)


def _script() -> str:
    """The retire for one example run."""
    return retire_script(
        target="C:/fleet/stage/run-1",
        retained="C:/fleet/stage/logs/run-1.log",
        scripts=("C:/fleet/stage/stop-run-1.ps1", "C:/fleet/stage/retire-run-1.ps1"),
        task="fleet-run-1",
    )


def _param_end() -> int:
    """The index of the line closing the render's param block."""
    return _script().splitlines().index(")")


def test_each_directory_goes_through_rd_verbatim_and_is_removed_again_while_held() -> None:
    """rd leaves a file another process holds and exits 0 (loki, 2026-10-21,
    MCPs board task a703ec42), so a directory it left is named once, removed
    again every second up to the bound, and named again in the failure."""
    lines = _script().splitlines()

    verbatim = lines.index(
        f"        $verbatim = {VERBATIM_PREFIX_LITERAL} + [IO.Path]::GetFullPath($directory)"
    )
    assert VERBATIM_PREFIX_LITERAL == "'\\\\?\\'"
    assert lines[lines.index("$removing = [Diagnostics.Stopwatch]::StartNew()") + 1] == (
        "foreach ($directory in @($Target, $Staging)) {"
    )
    assert lines[verbatim + 1 : verbatim + 14] == [
        f"        $exited = {RD_FUNCTION} $Cmd $verbatim",
        "        if (Test-Path -LiteralPath $directory) {",
        f'            Write-Output "{RETIRE_HELD}: rd exited $exited and left $directory, '
        f'$({DESCRIBE_FUNCTION} $verbatim); removing it again for up to $HeldSeconds s"',
        "        }",
        "        while ((Test-Path -LiteralPath $directory) -and "
        "$removing.Elapsed.TotalSeconds -lt $HeldSeconds) {",
        "            Start-Sleep -Seconds 1",
        f"            $exited = {RD_FUNCTION} $Cmd $verbatim",
        "        }",
        "        if (Test-Path -LiteralPath $directory) {",
        f'            throw "{RETIRE_INCOMPLETE}: rd exited $exited and left $directory '
        f'after $HeldSeconds s, $({DESCRIBE_FUNCTION} $verbatim)"',
        "        }",
        "    }",
        "}",
    ]
    assert "Remove-Item -Recurse" not in _script()


def test_rd_runs_in_one_function_that_reads_its_exit_beside_it() -> None:
    """MCPs' PowerShell harness refuses a native command whose exit is not read
    in its statement or the next (PS-NATIVE-EXIT-UNCHECKED), so the retire
    has one call site and every message reports the code it answered. $Cmd
    is passed in, since PSScriptAnalyzer reads a parameter used only inside a
    nested function as unused (fleet job f629ec98)."""
    lines = _script().splitlines()
    start = lines.index(f"function {RD_FUNCTION} {{")

    assert lines[start + 1 : start + 6] == [
        "    param([string]$Command, [string]$Verbatim)",
        "    & $Command /d /c rd /s /q $Verbatim",
        "    return $LASTEXITCODE",
        "}",
        "$removing = [Diagnostics.Stopwatch]::StartNew()",
    ]
    assert _script().count("& $Command") == 1
    assert "& $Cmd" not in _script()


def test_the_wait_leaves_most_of_the_ssh_deadline_to_the_rest_of_the_retire() -> None:
    assert HELD_SECONDS == 30
    assert HELD_SECONDS * 4 <= SSH_TIMEOUT_SECONDS
    assert _script().splitlines()[_param_end() - 1] == f"    [int]$HeldSeconds = {HELD_SECONDS}"


def test_the_description_lists_what_is_left_verbatim_and_asks_about_the_short() -> None:
    lines = _script().splitlines()
    start = lines.index(f"function {DESCRIBE_FUNCTION} {{")
    named = indented(named_holders_lines(paths_variable="$asked"), depth=1)

    assert lines[start + 1 : start + 5] == [
        "    param([string]$Verbatim)",
        "    $left = @([IO.Directory]::GetFiles($Verbatim, '*', 'AllDirectories') |",
        f"        ForEach-Object {{ $_.Substring({VERBATIM_PREFIX_LITERAL}.Length) }})",
        f"    $asked = @($left | Where-Object {{ $_.Length -le {RESTART_MANAGER_PATH_LIMIT} }})",
    ]
    assert lines[start + 5 : start + 5 + len(named)] == list(named)
    assert lines[start + 5 + len(named) :][:7] == [
        "    $heldBy = 'no process Restart Manager names'",
        "    if ($holders.Count -gt 0) {",
        "        $heldBy = @($holders | ForEach-Object { $_.Named }) -join '; '",
        "    }",
        '    return "$($left.Count) files left, $($left.Count - $asked.Count) of them too '
        'long for Restart Manager to ask about, held by $heldBy"',
        "}",
        "[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Retained)) | Out-Null",
    ]


def test_the_holders_type_is_compiled_once_after_the_strict_header() -> None:
    lines = _script().splitlines()
    compiled = compile_lines()
    start = lines.index("$ErrorActionPreference = 'Stop'") + 1

    assert lines[start : start + len(compiled)] == list(compiled)
    assert _script().count("Add-Type") == 1


def test_the_transcript_s_holders_are_ended_inside_its_guard_before_it_is_moved() -> None:
    lines = _script().splitlines()
    guard = lines.index("if (Test-Path -LiteralPath $Log) {")
    moved = lines.index("    Move-Item -Force -LiteralPath $Log -Destination $Retained")

    assert lines[guard + 1 : moved] == list(
        indented(end_holders_lines(path_variable="$Log"), depth=1)
    )
    assert lines[moved + 1] == "}"
    assert lines[_param_end() - 2] == "    [int[]]$EndableTypes = @(0, 1, 2, 5),"


def test_cmd_is_a_parameter_naming_system32_s_by_default() -> None:
    assert '    [string]$Cmd = "$env:SystemRoot\\System32\\cmd.exe",' in _script().splitlines()


def test_a_path_that_cannot_be_embedded_is_refused() -> None:
    with pytest.raises(ValueError, match="target"):
        retire_script(target="C:/it's", retained="C:/r.log", scripts=(), task="t")
