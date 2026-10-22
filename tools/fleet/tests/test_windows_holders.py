"""Ending a settled run's transcript holders (fleet.core.windows_holders, MCPs board task e40bca34).

``tests/pester/rendered-dialect-state.Tests.ps1`` executes the committed
retire render against a transcript a real process holds, ending it in one
case and refusing it by name in the other. This pins what that suite cannot
see from outside: which Restart Manager kinds may be ended, the 12-byte
layout of the structure that identifies a holder, that a holder is named
only after its start time is read again, the longest path Restart Manager
is asked about, and what cannot be embedded.
"""

from __future__ import annotations

import pytest

from fleet.core.powershell_text import add_type_lines
from fleet.core.windows_holders import (
    ENDABLE_APP_TYPES,
    HOLDER_ENDED,
    HOLDER_PROTECTED,
    HOLDERS_SOURCE,
    HOLDERS_TYPE,
    RESTART_MANAGER_PATH_LIMIT,
    compile_lines,
    end_holders_lines,
    named_holders_lines,
)


def test_a_service_explorer_or_critical_process_is_never_ended() -> None:
    assert ENDABLE_APP_TYPES == (0, 1, 2, 5)
    assert {3, 4, 1000}.isdisjoint(ENDABLE_APP_TYPES)


def test_a_holder_s_start_time_is_two_four_byte_halves_not_an_eight_aligned_long() -> None:
    # Measured on the hub: with a long here every holder after the first
    # read as garbage, a pid of -681976193 among them.
    assert "public uint StartTimeLow;\n            public uint StartTimeHigh;" in HOLDERS_SOURCE
    assert "public long ProcessStartTime;" not in HOLDERS_SOURCE


def test_the_restart_manager_session_is_ended_however_the_query_ends() -> None:
    finally_at = HOLDERS_SOURCE.index("} finally {\n                RmEndSession(session);")

    assert HOLDERS_SOURCE.index("RmGetList(session") < finally_at


def test_restart_manager_is_asked_about_every_file_in_one_session() -> None:
    """A run directory rd left can hold several files (MCPs board task a703ec42)."""
    assert "public static FileHolder[] List(string[] paths) {" in HOLDERS_SOURCE
    assert "session, (uint)paths.Length, paths, 0, null, 0, null);" in HOLDERS_SOURCE


def test_the_path_limit_is_max_path_less_its_terminating_null() -> None:
    # Measured on loki 2026-10-22: 259 characters registered, 260 refused.
    assert RESTART_MANAGER_PATH_LIMIT == 259


def test_the_compile_is_the_guarded_add_type_of_the_holders_source() -> None:
    assert compile_lines() == add_type_lines(HOLDERS_TYPE, HOLDERS_SOURCE)


def test_the_naming_filters_on_start_time_and_names_each_holder_it_still_finds() -> None:
    """Each lookup is a collection the step runs over, so a holder that exits
    after the filter is skipped rather than failing the retire under strict
    mode (sedona, 2026-10-18T20:29:43Z, MCPs board task 3d9eec0d)."""
    assert named_holders_lines(paths_variable="$asked") == (
        "$holders = @([FleetNode.FileHolders]::List([string[]]$asked) | Where-Object { "
        "[FleetNode.FileHolders]::StartTimeOf($_.Pid) -eq $_.StartTime } | ForEach-Object {",
        "    $holder = $_",
        "    foreach ($process in @(Get-CimInstance Win32_Process "
        '-Filter "ProcessId=$($holder.Pid)")) {',
        "        [pscustomobject]@{ Pid = $holder.Pid; AppType = $holder.AppType; "
        'Named = "pid $($holder.Pid) $($process.Name) ($($process.CommandLine))" }',
        "    }",
        "})",
    )


def test_the_ending_names_the_holders_then_refuses_or_ends_each_one() -> None:
    lines = end_holders_lines(path_variable="$Log")
    named = named_holders_lines(paths_variable="$Log")

    assert lines[: len(named)] == named
    assert lines[len(named) :] == (
        "foreach ($holder in $holders) {",
        "    if ($EndableTypes -notcontains $holder.AppType) {",
        f'        throw "{HOLDER_PROTECTED}: $($holder.Named), of Restart Manager type '
        '$($holder.AppType), holds $Log"',
        "    }",
        "    foreach ($running in @(Get-Process | Where-Object { $_.Id -eq $holder.Pid })) {",
        f'        Write-Output "{HOLDER_ENDED}: $($holder.Named) held $Log"',
        "        Stop-Process -InputObject $running -Force",
        "        $running.WaitForExit()",
        "    }",
        "}",
    )


@pytest.mark.parametrize("variable", ["Log", "$Lo g", "$(Get-Item x)", "$"])
def test_a_file_that_is_not_a_plain_variable_is_refused(variable: str) -> None:
    with pytest.raises(ValueError, match="holder's file must be a plain variable"):
        end_holders_lines(path_variable=variable)


@pytest.mark.parametrize("variable", ["asked", "$as ked", "$($left)", "$"])
def test_files_that_are_not_a_plain_variable_are_refused(variable: str) -> None:
    with pytest.raises(ValueError, match="holder's files must be a plain variable"):
        named_holders_lines(paths_variable=variable)
