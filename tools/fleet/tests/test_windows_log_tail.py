"""The Windows transcript tail: its text pinned, and its behaviour run on this hub.

The regression is MCPs board task 704f324f: ``Get-Content -Tail 200`` on a
three-megabyte UTF-16 transcript outlived the collector's 120-second deadline
on sedona for an hour, so a finished run was never collected. The first class
pins the script's text wherever the suite runs; the second runs it under
Windows PowerShell 5.1 against transcripts in each encoding a node's writers
produce, one of them written by the ``*>>`` redirection the build itself uses.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import time

import pytest

from fleet.core.dialect_windows import POWERSHELL_INVOCATION, WindowsDialect
from fleet.core.windows_log_tail import TAIL_BYTES, windows_log_tail_script

#: The UTF-16LE byte-order mark Windows PowerShell 5.1's ``*>>`` writes.
UTF16_MARK = b"\xff\xfe"

#: The UTF-8 byte-order mark ``Set-Content -Encoding utf8`` writes under 5.1.
UTF8_MARK = b"\xef\xbb\xbf"


class TestScriptText:
    def test_the_dialect_hands_the_bounded_tail_the_transcript_path(self) -> None:
        body = WindowsDialect().log_tail_script("C:/s/run-1", 200)

        assert body == windows_log_tail_script("C:/s/run-1/result.txt.log", 200)

    def test_it_seeks_a_bounded_distance_from_the_end_and_never_reads_the_whole_file(
        self,
    ) -> None:
        body = windows_log_tail_script("C:/s/run-1/result.txt.log", 200)

        assert body == (
            "if (Test-Path -LiteralPath 'C:/s/run-1/result.txt.log') {\n"
            "  $stream = [System.IO.File]::Open('C:/s/run-1/result.txt.log',"
            " 'Open', 'Read', 'ReadWrite')\n"
            "  $mark = New-Object byte[] 3\n"
            "  $marked = $stream.Read($mark, 0, 3)\n"
            "  $encoding = [System.Text.Encoding]::UTF8\n"
            "  $skip = 0\n"
            "  if (($marked -ge 2) -and ($mark[0] -eq 255) -and ($mark[1] -eq 254)) {"
            " $encoding = [System.Text.Encoding]::Unicode; $skip = 2 }\n"
            "  if (($marked -eq 3) -and ($mark[0] -eq 239) -and ($mark[1] -eq 187)"
            " -and ($mark[2] -eq 191)) { $skip = 3 }\n"
            "  $start = [Math]::Max([long]$skip, $stream.Length - 262144)\n"
            "  if (($skip -eq 2) -and (($start % 2) -eq 1)) { $start = $start + 1 }\n"
            "  $null = $stream.Seek($start, 'Begin')\n"
            "  $bytes = New-Object byte[] ($stream.Length - $start)\n"
            "  $read = 0\n"
            "  while ($read -lt $bytes.Length) {\n"
            "    $got = $stream.Read($bytes, $read, $bytes.Length - $read)\n"
            "    if ($got -eq 0) { break }\n"
            "    $read = $read + $got\n"
            "  }\n"
            "  $stream.Close()\n"
            '  $text = $encoding.GetString($bytes, 0, $read).TrimEnd([char[]]"`r`n")\n'
            '  $parts = $text -split "`r?`n"\n'
            "  if ($start -gt $skip) { $parts = $parts | Select-Object -Skip 1 }\n"
            "  $parts | Select-Object -Last 200\n"
            "}\n"
        )
        assert "Get-Content" not in body
        assert TAIL_BYTES == 262_144


def _numbered(count: int) -> str:
    """Lines ``line 0000000 ...`` through ``count - 1``, each CRLF-terminated.

    Args:
        count: How many lines.

    Returns:
        The text, as a build transcript's lines would read.
    """
    return "".join(f"line {index:07d} passed in the fixture\r\n" for index in range(count))


@pytest.mark.skipif(sys.platform != "win32", reason="the tail is PowerShell; run it here")
class TestTailForReal:
    """The script run by path under 5.1, the way the collector runs it on a node."""

    def tail(self, tmp_path: pathlib.Path, log: pathlib.Path, lines: int) -> list[str]:
        """Run the tail of ``log`` and return the lines it printed.

        Args:
            tmp_path: Where to write the script.
            log: The transcript.
            lines: How many lines to ask for.

        Returns:
            Every line of its standard output.
        """
        script = tmp_path / "log-tail.ps1"
        script.write_text(windows_log_tail_script(log.as_posix(), lines), encoding="utf-8")
        completed = subprocess.run(
            [*POWERSHELL_INVOCATION, str(script)],
            capture_output=True,
            encoding="utf-8",
            check=False,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stderr == ""
        return completed.stdout.splitlines()

    def test_a_transcript_the_redirection_wrote_reads_back_without_its_mark(
        self, tmp_path: pathlib.Path
    ) -> None:
        log = tmp_path / "result.txt.log"
        writer = tmp_path / "write.ps1"
        writer.write_text(
            f"Write-Output 'first line' *>> '{log.as_posix()}'\n"
            f"Write-Output 'FLEET-CHECK status=passed' *>> '{log.as_posix()}'\n",
            encoding="utf-8",
        )
        subprocess.run([*POWERSHELL_INVOCATION, str(writer)], check=True, timeout=120)

        assert log.read_bytes()[:2] == UTF16_MARK
        assert self.tail(tmp_path, log, 200) == ["first line", "FLEET-CHECK status=passed"]

    def test_a_three_megabyte_utf16_transcript_returns_its_last_lines_quickly(
        self, tmp_path: pathlib.Path
    ) -> None:
        log = tmp_path / "result.txt.log"
        log.write_bytes(UTF16_MARK + _numbered(45_000).encode("utf-16-le"))
        assert log.stat().st_size > 3_000_000

        began = time.monotonic()
        printed = self.tail(tmp_path, log, 3)
        elapsed = time.monotonic() - began

        assert printed == [
            "line 0044997 passed in the fixture",
            "line 0044998 passed in the fixture",
            "line 0044999 passed in the fixture",
        ]
        assert elapsed < 60.0

    def test_a_window_that_starts_mid_line_drops_the_cut_line(self, tmp_path: pathlib.Path) -> None:
        log = tmp_path / "result.txt.log"
        log.write_bytes(UTF16_MARK + _numbered(40_000).encode("utf-16-le"))

        printed = self.tail(tmp_path, log, 100_000)

        # Each line is 36 characters, 72 bytes in UTF-16, and 262144 is not a
        # multiple of 72, so the window opens inside line 36359: dropped, and
        # the 3640 whole lines after it are all that is read.
        assert len(printed) == TAIL_BYTES // 72 == 3640
        assert printed[0] == "line 0036360 passed in the fixture"
        assert printed[-1] == "line 0039999 passed in the fixture"

    def test_an_odd_length_utf16_transcript_still_decodes_on_character_boundaries(
        self, tmp_path: pathlib.Path
    ) -> None:
        log = tmp_path / "result.txt.log"
        log.write_bytes(UTF16_MARK + _numbered(40_000).encode("utf-16-le") + b"\x00")

        printed = self.tail(tmp_path, log, 3)

        assert printed[:2] == [
            "line 0039998 passed in the fixture",
            "line 0039999 passed in the fixture",
        ]

    def test_a_marked_utf8_transcript_reads_back_without_its_mark(
        self, tmp_path: pathlib.Path
    ) -> None:
        log = tmp_path / "result.txt.log"
        log.write_bytes(UTF8_MARK + _numbered(3).encode("utf-8"))

        assert self.tail(tmp_path, log, 200) == [
            "line 0000000 passed in the fixture",
            "line 0000001 passed in the fixture",
            "line 0000002 passed in the fixture",
        ]

    def test_an_unmarked_utf8_transcript_reads_as_utf8(self, tmp_path: pathlib.Path) -> None:
        log = tmp_path / "result.txt.log"
        log.write_bytes("résumé\nFLEET-CHECK status=failed\n".encode())

        assert self.tail(tmp_path, log, 1) == ["FLEET-CHECK status=failed"]

    def test_an_absent_transcript_prints_nothing(self, tmp_path: pathlib.Path) -> None:
        assert self.tail(tmp_path, tmp_path / "result.txt.log", 200) == []
