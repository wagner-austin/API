"""Ending a settled run's transcript holders (fleet.core.windows_holders, MCPs board task e40bca34).

``tests/pester/rendered-dialect-state.Tests.ps1`` executes the committed
retire render against a transcript a real process holds, ending it in one
case and refusing it by name in the other. This pins what that suite cannot
see from outside: which Restart Manager kinds may be ended, the 12-byte
layout of the structure that identifies a holder, that a holder is ended
only after its start time is read again, and what cannot be embedded.
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
    end_holders_lines,
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


def test_the_lines_filter_on_start_time_then_refuse_or_end_each_holder() -> None:
    lines = end_holders_lines(path_variable="$Log")
    compiled = add_type_lines(HOLDERS_TYPE, HOLDERS_SOURCE)

    assert lines[: len(compiled)] == compiled
    assert lines[len(compiled) :] == (
        "$holders = @([FleetNode.FileHolders]::List($Log) | Where-Object { "
        "[FleetNode.FileHolders]::StartTimeOf($_.Pid) -eq $_.StartTime })",
        "foreach ($holder in $holders) {",
        '    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($holder.Pid)"',
        '    $named = "pid $($holder.Pid) $($process.Name) ($($process.CommandLine))"',
        "    if ($EndableTypes -notcontains $holder.AppType) {",
        f'        throw "{HOLDER_PROTECTED}: $named, of Restart Manager type '
        '$($holder.AppType), holds $Log"',
        "    }",
        f'    Write-Output "{HOLDER_ENDED}: $named held $Log"',
        "    $running = Get-Process -Id $holder.Pid",
        "    Stop-Process -InputObject $running -Force",
        "    $running.WaitForExit()",
        "}",
    )


@pytest.mark.parametrize("variable", ["Log", "$Lo g", "$(Get-Item x)", "$"])
def test_a_file_that_is_not_a_plain_variable_is_refused(variable: str) -> None:
    with pytest.raises(ValueError, match="plain variable"):
        end_holders_lines(path_variable=variable)
