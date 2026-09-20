"""One shape for a process on either platform, and the two readers that fill it.

The reaper's questions -- who is whose child, how old is it, is it burning
CPU, how much commit does it hold -- are the same on both platforms; only
the source differs. Windows answers through the CIM ``Win32_Process`` class,
asked by PowerShell and returned as JSON, because nothing in the standard
library reads a Windows process's parent, command line or CPU time. Linux
answers from ``/proc``. Each reader takes its source through a parameter (a
command runner, a ``/proc`` root) so the parsing is exercised on the
platform that cannot produce the real input.

WHY ``/proc`` AND NOT ``ps``. ``ps`` output is a table for people, with
columns that wrap and a command line that ``ps`` truncates and re-quotes;
``/proc/<pid>/stat`` and ``cmdline`` are the source ``ps`` itself reads.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from pathlib import Path
from typing import Final, TypedDict

from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import (
    JSONValue,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_list,
    require_float,
    require_int,
    require_str,
)

from maketools.commands import RunCapturingProtocol


class ProcessRow(TypedDict):
    """One process as the reaper sees it.

    Attributes:
        pid: The process id.
        parent_pid: Its parent's id at the time of the snapshot.
        name: The executable's basename (``python.exe``, ``python3.11``).
        command_line: The full command line, space-joined; empty when the
            platform would not show it.
        executable: The executable's path; empty when unknown. On Linux this
            is the command line's first word, which is what a venv
            interpreter's path is and is all the project test needs.
        created_unix: When it started, seconds since the epoch; 0 when the
            platform has no answer (the Windows idle process).
        cpu_seconds: User plus kernel CPU time so far.
        commit_mb: Committed memory in MiB (page-file usage on Windows,
            ``VmSize`` on Linux).
    """

    pid: int
    parent_pid: int
    name: str
    command_line: str
    executable: str
    created_unix: float
    cpu_seconds: float
    commit_mb: float


#: Where Linux keeps the process table.
LINUX_PROC_ROOT: Final[Path] = Path("/proc")

#: Ticks per second in ``/proc/<pid>/stat``. ``USER_HZ`` is fixed at 100 by
#: the kernel's userspace ABI on every architecture Linux runs on; it is not
#: the scheduler's ``HZ``. Written as a constant rather than read through
#: ``os.sysconf``, which the type checker on Windows does not know.
LINUX_CLOCK_TICKS: Final[int] = 100

#: The PowerShell program that renders every process as one JSON row in the
#: shape :class:`ProcessRow` expects. Sent encoded so no shell between here
#: and PowerShell reads its quotes. ``-InputObject`` on the array literal
#: keeps a one-process answer an array rather than a bare object. A null
#: ``CreationDate`` (the idle process) becomes 0 rather than an error.
WINDOWS_SNAPSHOT_SCRIPT: Final[str] = (
    "$rows = @(Get-CimInstance Win32_Process | ForEach-Object { [pscustomobject]@{"
    " pid = [int]$_.ProcessId; parent_pid = [int]$_.ParentProcessId;"
    " name = [string]$_.Name; command_line = [string]$_.CommandLine;"
    " executable = [string]$_.ExecutablePath;"
    " created_unix = $(if ($null -eq $_.CreationDate) { 0.0 } else {"
    " [double]((($_.CreationDate).ToUniversalTime() - [datetime]'1970-01-01').TotalSeconds) });"
    " cpu_seconds = [double](([int64]$_.UserModeTime + [int64]$_.KernelModeTime) / 10000000);"
    " commit_mb = [double]($_.PageFileUsage / 1024) } });"
    " ConvertTo-Json -InputObject $rows -Compress"
)


def windows_snapshot_argv() -> Sequence[str]:
    """The command that runs :data:`WINDOWS_SNAPSHOT_SCRIPT`.

    Returns:
        ``powershell -NoProfile -EncodedCommand <base64 of UTF-16LE>``.
    """
    encoded = base64.b64encode(WINDOWS_SNAPSHOT_SCRIPT.encode("utf-16-le")).decode("ascii")
    return ("powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded)


def decode_process_row(value: JSONValue) -> ProcessRow:
    """Narrow one decoded JSON row.

    Args:
        value: The row.

    Returns:
        The row.

    Raises:
        JSONTypeError: On any field missing or of the wrong type.
    """
    row = narrow_json_to_dict(value)
    return ProcessRow(
        pid=require_int(row, "pid"),
        parent_pid=require_int(row, "parent_pid"),
        name=require_str(row, "name"),
        command_line=require_str(row, "command_line"),
        executable=require_str(row, "executable"),
        created_unix=require_float(row, "created_unix"),
        cpu_seconds=require_float(row, "cpu_seconds"),
        commit_mb=require_float(row, "commit_mb"),
    )


def windows_process_table(run: RunCapturingProtocol) -> Sequence[ProcessRow]:
    """Snapshot the Windows process table.

    Args:
        run: The command runner that executes PowerShell.

    Returns:
        Every process the CIM query returned.

    Raises:
        AppError: ``MAKETOOLS_PROCESS_TABLE`` when PowerShell exits non-zero.
        InvalidJsonError: When its answer is not JSON.
        JSONTypeError: When its answer is not an array of rows.
    """
    result = run(windows_snapshot_argv(), cwd=Path.cwd())
    if result["returncode"] != 0:
        raise AppError(
            MaketoolsErrorCode.PROCESS_TABLE,
            f"the process snapshot exited {result['returncode']}: {result['stderr'].strip()}",
        )
    return [decode_process_row(row) for row in narrow_json_to_list(load_json_str(result["stdout"]))]


def linux_boot_unix(proc_root: Path) -> float:
    """Read the boot instant, which ``/proc/<pid>/stat`` start times count from.

    Args:
        proc_root: The ``/proc`` root.

    Returns:
        Seconds since the epoch.

    Raises:
        AppError: ``MAKETOOLS_PROCESS_TABLE`` when ``/proc/stat`` has no
            ``btime`` line.
    """
    for line in (proc_root / "stat").read_text(encoding="utf-8").splitlines():
        if line.startswith("btime "):
            return float(line.split()[1])
    raise AppError(MaketoolsErrorCode.PROCESS_TABLE, f"{proc_root / 'stat'} carries no btime line")


def linux_commit_mb(status_text: str) -> float:
    """Read ``VmSize`` from a ``/proc/<pid>/status`` document.

    Args:
        status_text: The file's contents.

    Returns:
        The size in MiB; 0 for a kernel thread, which has no ``VmSize``.
    """
    for line in status_text.splitlines():
        if line.startswith("VmSize:"):
            return float(line.split()[1]) / 1024
    return 0.0


def linux_process_row(pid: int, proc_root: Path, boot_unix: float) -> ProcessRow:
    """Read one process from its ``/proc`` directory.

    Args:
        pid: The process.
        proc_root: The ``/proc`` root.
        boot_unix: From :func:`linux_boot_unix`.

    Returns:
        The row.

    Raises:
        FileNotFoundError: When the process exited between the listing and
            this read; the caller skips it.
    """
    directory = proc_root / str(pid)
    stat = (directory / "stat").read_text(encoding="utf-8")
    # The command name is parenthesised and may itself contain spaces or
    # parentheses, so the fields are split after the LAST closing paren.
    name = stat[stat.index("(") + 1 : stat.rindex(")")]
    fields = stat[stat.rindex(")") + 2 :].split()
    # After the name: state(0) ppid(1) ... utime(11) stime(12) ... starttime(19).
    parent_pid = int(fields[1])
    cpu_seconds = (int(fields[11]) + int(fields[12])) / LINUX_CLOCK_TICKS
    created_unix = boot_unix + int(fields[19]) / LINUX_CLOCK_TICKS
    command_line = (directory / "cmdline").read_bytes().decode("utf-8", errors="replace")
    words = [word for word in command_line.split("\0") if word != ""]
    return ProcessRow(
        pid=pid,
        parent_pid=parent_pid,
        name=name,
        command_line=" ".join(words),
        executable=words[0] if words else "",
        created_unix=created_unix,
        cpu_seconds=cpu_seconds,
        commit_mb=linux_commit_mb((directory / "status").read_text(encoding="utf-8")),
    )


def linux_process_table(proc_root: Path) -> Sequence[ProcessRow]:
    """Snapshot the Linux process table.

    A process that exits between the directory listing and its own read is
    skipped: it is not in the table any more, which is the answer a snapshot
    taken a moment later would have given.

    Args:
        proc_root: The ``/proc`` root.

    Returns:
        Every process with a numeric directory.
    """
    boot_unix = linux_boot_unix(proc_root)
    rows: list[ProcessRow] = []
    pids = sorted(int(entry.name) for entry in proc_root.iterdir() if entry.name.isdigit())
    for pid in pids:
        try:
            rows.append(linux_process_row(pid, proc_root, boot_unix))
        except FileNotFoundError:
            continue
    return rows


__all__ = [
    "LINUX_CLOCK_TICKS",
    "LINUX_PROC_ROOT",
    "WINDOWS_SNAPSHOT_SCRIPT",
    "ProcessRow",
    "decode_process_row",
    "linux_boot_unix",
    "linux_commit_mb",
    "linux_process_row",
    "linux_process_table",
    "windows_process_table",
    "windows_snapshot_argv",
]
