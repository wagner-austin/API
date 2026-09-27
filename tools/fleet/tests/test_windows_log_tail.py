"""The Windows transcript tail's text, pinned wherever the suite runs.

The regression is MCPs board task 704f324f: ``Get-Content -Tail 200`` on a
three-megabyte UTF-16 transcript outlived the collector's 120-second deadline
on sedona for an hour, so a finished run was never collected. The script is
run for real, under Windows PowerShell 5.1 against transcripts in each
encoding a node's writers produce, by
``tests/pester/rendered-dialect-tail.Tests.ps1`` over its committed render
(MCPs board task d69786fa); these cases pin what it is handed and that it
never reads the whole file.
"""

from __future__ import annotations

import pytest

from fleet.core.dialect_windows import WindowsDialect
from fleet.core.windows_log_tail import TAIL_BYTES, windows_log_tail_script


class TestScriptText:
    def test_the_dialect_hands_the_bounded_tail_the_transcript_path(self) -> None:
        body = WindowsDialect().log_tail_script("C:/s/run-1", 200)

        assert body == windows_log_tail_script("C:/s/run-1/result.txt.log", 200)

    def test_its_transcript_and_line_count_are_parameters_defaulting_to_the_rendered_ones(
        self,
    ) -> None:
        body = windows_log_tail_script("C:/s/run-1/result.txt.log", 200)

        assert body.startswith(
            "param(\n"
            "    [string]$Log = 'C:/s/run-1/result.txt.log',\n"
            "    [int]$Lines = 200\n"
            ")\n"
            "Set-StrictMode -Version Latest\n"
            "$ErrorActionPreference = 'Stop'\n"
        )
        assert "$parts | Select-Object -Last $Lines" in body

    def test_it_seeks_a_bounded_distance_from_the_end_and_never_reads_the_whole_file(
        self,
    ) -> None:
        body = windows_log_tail_script("C:/s/run-1/result.txt.log", 200)

        assert f"$start = [Math]::Max([long]$skip, $stream.Length - {TAIL_BYTES})" in body
        assert "$null = $stream.Seek($start, 'Begin')" in body
        assert "Get-Content" not in body
        assert TAIL_BYTES == 262_144

    def test_a_path_that_cannot_be_embedded_is_refused(self) -> None:
        with pytest.raises(ValueError, match="log"):
            windows_log_tail_script("C:/s/it's/result.txt.log", 200)
