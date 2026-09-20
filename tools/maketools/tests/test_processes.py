"""Both process-table readers, fed through their seams."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from pathlib import Path

import pytest
from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue, dump_json_str

from maketools.commands import CommandResult
from maketools.processes import (
    LINUX_CLOCK_TICKS,
    WINDOWS_SNAPSHOT_SCRIPT,
    decode_process_row,
    linux_boot_unix,
    linux_commit_mb,
    linux_process_row,
    linux_process_table,
    windows_process_table,
    windows_snapshot_argv,
)
from tests.conftest import failed, ok

ROW: JSONObject = {
    "pid": 10,
    "parent_pid": 1,
    "name": "python.exe",
    "command_line": "python -m pytest",
    "executable": "C:/p/.venv/Scripts/python.exe",
    "created_unix": 1700000000.5,
    "cpu_seconds": 1.25,
    "commit_mb": 512,
}


def with_fields(**overrides: JSONValue) -> JSONObject:
    merged: JSONObject = dict(ROW)
    merged.update(overrides)
    return merged


class Answering:
    """A capturing runner that answers one scripted result and records the argv."""

    def __init__(self, result: CommandResult) -> None:
        self.result = result
        self.argv: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str], *, cwd: Path) -> CommandResult:
        self.argv.append(tuple(argv))
        return self.result


def test_the_snapshot_argv_encodes_the_script_as_utf16_for_powershell() -> None:
    argv = windows_snapshot_argv()
    assert argv[:4] == ("powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand")
    assert base64.b64decode(argv[4]).decode("utf-16-le") == WINDOWS_SNAPSHOT_SCRIPT


def test_decode_process_row_reads_every_field() -> None:
    decoded = decode_process_row(ROW)
    assert decoded["pid"] == 10
    assert decoded["parent_pid"] == 1
    assert decoded["name"] == "python.exe"
    assert decoded["executable"] == "C:/p/.venv/Scripts/python.exe"
    assert decoded["created_unix"] == 1700000000.5
    assert decoded["cpu_seconds"] == 1.25
    assert decoded["commit_mb"] == 512.0


def test_decode_process_row_names_the_bad_field() -> None:
    with pytest.raises(JSONTypeError, match=r"Field 'pid' must be an integer, got str"):
        decode_process_row(with_fields(pid="10"))


def test_windows_table_runs_the_snapshot_and_decodes_every_row() -> None:
    runner = Answering(ok(dump_json_str([ROW, with_fields(pid=11, parent_pid=10)])))
    rows = windows_process_table(runner)
    assert [r["pid"] for r in rows] == [10, 11]
    assert runner.argv == [tuple(windows_snapshot_argv())]


def test_windows_table_refuses_a_failed_snapshot_with_its_stderr() -> None:
    runner = Answering(failed(1, "Get-CimInstance : access denied"))
    with pytest.raises(AppError) as caught:
        windows_process_table(runner)
    assert caught.value.code is MaketoolsErrorCode.PROCESS_TABLE
    assert caught.value.message == (
        "the process snapshot exited 1: Get-CimInstance : access denied"
    )


def test_windows_table_refuses_an_answer_that_is_not_an_array() -> None:
    with pytest.raises(JSONTypeError, match=r"Expected JSON array, got dict"):
        windows_process_table(Answering(ok(dump_json_str(ROW))))


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
    rows = linux_process_table(proc)
    assert [(r["pid"], r["parent_pid"], r["name"]) for r in rows] == [(3, 1, "a"), (12, 3, "b")]
