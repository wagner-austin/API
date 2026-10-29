"""The Linux process-table reader, fed through its proc-root seam."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError

from maketools.processes import (
    LINUX_CLOCK_TICKS,
    ProcessRow,
    linux_boot_unix,
    linux_commit_mb,
    linux_process_row,
    linux_process_table,
)


def write_process(proc: Path, pid: int, *, name: str, ppid: int, cmdline: Sequence[str]) -> None:
    directory = proc / str(pid)
    directory.mkdir()
    # pid (comm) state ppid pgrp session tty tpgid flags minflt cminflt majflt
    # cmajflt utime stime cutime cstime priority nice threads itrealvalue starttime
    tail = f"S {ppid} 1 1 0 -1 4194560 0 0 0 0 250 50 0 0 20 0 1 0 12345"
    (directory / "stat").write_text(f"{pid} ({name}) {tail}\n", encoding="utf-8")
    (directory / "cmdline").write_bytes("\0".join(cmdline).encode("utf-8") + b"\0")
    (directory / "status").write_text(
        f"Name:\t{name}\nVmPeak:\t 100 kB\nVmSize:\t 20480 kB\n", encoding="utf-8"
    )


@pytest.fixture()
def proc(tmp_path: Path) -> Path:
    root = tmp_path / "proc"
    root.mkdir()
    (root / "stat").write_text("cpu 1 2 3\nbtime 1700000000\nprocesses 5\n", encoding="utf-8")
    (root / "self").mkdir()
    (root / "cpuinfo").write_text("processor : 0\n", encoding="utf-8")
    return root


def test_linux_boot_unix_reads_btime(proc: Path) -> None:
    assert linux_boot_unix(proc) == 1700000000.0


def test_linux_boot_unix_refuses_a_stat_without_btime(proc: Path) -> None:
    (proc / "stat").write_text("cpu 1 2 3\n", encoding="utf-8")
    with pytest.raises(AppError) as caught:
        linux_boot_unix(proc)
    assert caught.value.code is MaketoolsErrorCode.PROCESS_TABLE
    assert caught.value.message.endswith("carries no btime line")


def test_linux_commit_mb_reads_vmsize_and_is_zero_for_a_kernel_thread() -> None:
    assert linux_commit_mb("VmSize:\t 20480 kB\n") == 20.0
    assert linux_commit_mb("Name:\tkthreadd\n") == 0.0


def test_linux_process_row_parses_stat_cmdline_and_status(proc: Path) -> None:
    write_process(
        proc, 77, name="python3 (x)", ppid=5, cmdline=["/p/.venv/bin/python", "-m", "pytest"]
    )
    parsed = linux_process_row(77, proc, 1700000000.0)
    assert parsed["pid"] == 77
    assert parsed["parent_pid"] == 5
    assert parsed["name"] == "python3 (x)"
    assert parsed["command_line"] == "/p/.venv/bin/python -m pytest"
    assert parsed["executable"] == "/p/.venv/bin/python"
    assert parsed["cpu_seconds"] == (250 + 50) / LINUX_CLOCK_TICKS
    assert parsed["created_unix"] == 1700000000.0 + 12345 / LINUX_CLOCK_TICKS
    assert parsed["commit_mb"] == 20.0


def test_linux_process_row_with_an_empty_cmdline_has_no_executable(proc: Path) -> None:
    write_process(proc, 2, name="kthreadd", ppid=0, cmdline=[])
    parsed = linux_process_row(2, proc, 0.0)
    assert parsed["command_line"] == ""
    assert parsed["executable"] == ""


def test_linux_process_table_lists_numeric_directories_and_skips_a_vanished_one(
    proc: Path,
) -> None:
    write_process(proc, 3, name="a", ppid=1, cmdline=["a"])
    write_process(proc, 12, name="b", ppid=3, cmdline=["b"])
    (proc / "99").mkdir()  # exited between the listing and the read: no stat
    rows = linux_process_table(proc, linux_process_row)
    assert [(r["pid"], r["parent_pid"], r["name"]) for r in rows] == [(3, 1, "a"), (12, 3, "b")]


def test_linux_process_table_skips_a_process_whose_read_the_kernel_answers_esrch(
    proc: Path,
) -> None:
    # Fleet job d2454143 (MCPs board task cf2e4802): a process being torn
    # down keeps its listed directory, and reading its stat raises
    # ProcessLookupError rather than FileNotFoundError.
    write_process(proc, 3, name="a", ppid=1, cmdline=["a"])
    write_process(proc, 50, name="exiting", ppid=3, cmdline=["exiting"])
    write_process(proc, 12, name="b", ppid=3, cmdline=["b"])
    (proc / "99").mkdir()
    asked: list[int] = []

    def read_row(pid: int, proc_root: Path, boot_unix: float) -> ProcessRow:
        asked.append(pid)
        if pid == 50:
            raise ProcessLookupError(3, "No such process")
        return linux_process_row(pid, proc_root, boot_unix)

    rows = linux_process_table(proc, read_row)

    assert [r["pid"] for r in rows] == [3, 12]
    assert asked == [3, 12, 50, 99]


def test_linux_process_table_lets_any_other_read_failure_propagate(proc: Path) -> None:
    write_process(proc, 3, name="a", ppid=1, cmdline=["a"])

    def read_row(pid: int, proc_root: Path, boot_unix: float) -> ProcessRow:
        raise PermissionError(13, "Permission denied", str(proc_root / str(pid) / "stat"))

    with pytest.raises(PermissionError):
        linux_process_table(proc, read_row)
