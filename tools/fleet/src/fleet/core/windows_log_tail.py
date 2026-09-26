"""The last lines of a Windows build transcript, read at a cost that does not grow with it.

WHY NOT ``Get-Content -Tail`` (MCPs board task 704f324f). The Windows build
appends every step to its transcript with ``*>>``, and Windows PowerShell 5.1
writes that redirection as UTF-16LE with a byte-order mark. On a transcript
that size ``Get-Content -Tail 200`` did not return within the collector's
120-second deadline: measured 2026-09-26 on sedona, run
``tools-fleet-1790400468`` finished at 05:31:41Z with a 2,973,062-byte log,
and every collect tick for the next hour died at ``log-tail.ps1`` with
"timed out after 120 s", so the job never closed, its verdict never posted,
and sedona's one run slot stayed held by a run that had long finished.

WHAT IT DOES INSTEAD. It opens the file for shared reading and reads its
first three bytes to learn the encoding: ``FF FE`` is the UTF-16LE mark the
redirection writes, ``EF BB BF`` the UTF-8 mark ``Set-Content -Encoding
utf8`` writes under 5.1, and anything else is read as unmarked UTF-8. It
seeks to at most :data:`TAIL_BYTES` before the end (never into the mark, and
moved forward one byte when a UTF-16 read would otherwise begin
mid-character), reads to the end, decodes, drops the first line when the
read began inside the file because that line is cut, drops the trailing line
break, and prints the last ``lines`` lines. One seek and one bounded read
whatever the transcript's size, so the tail of a two-hundred-megabyte log
costs what the tail of a two-kilobyte one does.
"""

from __future__ import annotations

from typing import Final

#: How many bytes from the end the tail reads. Two hundred lines of pytest's
#: or vitest's summary and coverage table run to a few hundred characters
#: each at most, so 256 KiB (128 Ki UTF-16 characters) holds the lines the
#: verdict parses with room to spare, and stays small enough to cross ssh in
#: well under a second.
TAIL_BYTES: Final[int] = 262_144


def windows_log_tail_script(log: str, lines: int) -> str:
    """The PowerShell that prints a transcript's last lines, or nothing.

    Args:
        log: Absolute path of the transcript on the node. Single-quoted into
            the script; run paths carry no quote by construction.
        lines: How many lines from the end to print.

    Returns:
        The script's text. An absent transcript prints nothing, which the
        verdict reports as a build that wrote no transcript.
    """
    return (
        f"if (Test-Path -LiteralPath '{log}') {{\n"
        f"  $stream = [System.IO.File]::Open('{log}', 'Open', 'Read', 'ReadWrite')\n"
        "  $mark = New-Object byte[] 3\n"
        "  $marked = $stream.Read($mark, 0, 3)\n"
        "  $encoding = [System.Text.Encoding]::UTF8\n"
        "  $skip = 0\n"
        "  if (($marked -ge 2) -and ($mark[0] -eq 255) -and ($mark[1] -eq 254)) {"
        " $encoding = [System.Text.Encoding]::Unicode; $skip = 2 }\n"
        "  if (($marked -eq 3) -and ($mark[0] -eq 239) -and ($mark[1] -eq 187)"
        " -and ($mark[2] -eq 191)) { $skip = 3 }\n"
        f"  $start = [Math]::Max([long]$skip, $stream.Length - {TAIL_BYTES})\n"
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
        f"  $parts | Select-Object -Last {lines}\n"
        "}\n"
    )


__all__ = ["TAIL_BYTES", "windows_log_tail_script"]
