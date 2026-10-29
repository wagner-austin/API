"""The Linux process rows read by the test launcher and reaper.

WHY ``/proc`` AND NOT ``ps``. ``ps`` output is a table for people, with
columns that wrap and a command line that ``ps`` truncates and re-quotes;
``/proc/<pid>/stat`` and ``cmdline`` are the source ``ps`` itself reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final, Protocol, TypedDict

from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError


class ProcessRow(TypedDict):
    """One process as the reaper sees it.

    Attributes:
        pid: The process id.
        parent_pid: Its parent's id at the time of the snapshot.
        name: The executable's basename (``python3``, ``python3.11``).
        command_line: The full command line, space-joined; empty when the
            platform would not show it.
        executable: The executable's path; empty when unknown. On Linux this
            is the command line's first word, which is what a venv
            interpreter's path is and is all the project test needs.
        created_unix: When it started, seconds since the epoch, derived
            from the boot instant and the process's start ticks.
        cpu_seconds: User plus kernel CPU time so far.
        commit_mb: Virtual memory in MiB, read from Linux ``VmSize``.
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
#: the scheduler's ``HZ``, so it is written as the constant it is.
LINUX_CLOCK_TICKS: Final[int] = 100


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
        FileNotFoundError: When the process exited and was reaped between
            the listing and this read; the caller skips it.
        ProcessLookupError: When the process is being torn down as this
            reads it: its directory is still listed, but the kernel answers
            the read with ESRCH (measured on colossus 2026-10-19, fleet job
            d2454143, MCPs board task cf2e4802). The caller skips it too.
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


class ReadRowProtocol(Protocol):
    """Reads one process's row: :func:`linux_process_row`'s shape."""

    def __call__(self, pid: int, proc_root: Path, boot_unix: float) -> ProcessRow:
        """Read it.

        Args:
            pid: The process.
            proc_root: The ``/proc`` root.
            boot_unix: From :func:`linux_boot_unix`.

        Returns:
            The row.
        """
        ...


def linux_process_table(proc_root: Path, read_row: ReadRowProtocol) -> Sequence[ProcessRow]:
    """Snapshot the Linux process table.

    A process that exits between the directory listing and its own read is
    skipped: it is not in the table any more, which is the answer a snapshot
    taken a moment later would have given. It shows as either of the two
    exceptions :func:`linux_process_row` names, and both are skipped; any
    other failure propagates.

    Args:
        proc_root: The ``/proc`` root.
        read_row: Reads one process; :func:`linux_process_row`, bound in
            ``maketools._test_hooks``, so the two ways a process vanishes
            mid-read are exercised with the kernel's own exception types.

    Returns:
        Every process with a numeric directory that was still there to read.
    """
    boot_unix = linux_boot_unix(proc_root)
    rows: list[ProcessRow] = []
    pids = sorted(int(entry.name) for entry in proc_root.iterdir() if entry.name.isdigit())
    for pid in pids:
        try:
            rows.append(read_row(pid, proc_root, boot_unix))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return rows


__all__ = [
    "LINUX_CLOCK_TICKS",
    "LINUX_PROC_ROOT",
    "ProcessRow",
    "ReadRowProtocol",
    "linux_boot_unix",
    "linux_commit_mb",
    "linux_process_row",
    "linux_process_table",
]
