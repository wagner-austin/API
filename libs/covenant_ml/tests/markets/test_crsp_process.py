"""The real bounded runner, driven with real child processes, including each way they fail."""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path

import pytest

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.crsp_process import run_bounded


def test_production_hook_is_the_real_runner() -> None:
    assert _test_hooks.bounded_runner is run_bounded


def test_child_writes_relative_to_its_directory_within_the_limit(tmp_path: Path) -> None:
    command = (sys.executable, "-c", "open('out.bin', 'wb').write(b'x' * 64)")
    assert run_bounded(command, tmp_path, seconds=30, max_file_bytes=64) == 0
    assert (tmp_path / "out.bin").read_bytes() == b"x" * 64


def test_nonzero_exit_is_returned_not_raised(tmp_path: Path) -> None:
    command = (sys.executable, "-c", "raise SystemExit(3)")
    assert run_bounded(command, tmp_path, seconds=30, max_file_bytes=64) == 3


def test_writing_past_the_file_limit_kills_the_child(tmp_path: Path) -> None:
    command = ("dd", "if=/dev/zero", "of=out.bin", "bs=1000", "count=1", "status=none")
    assert run_bounded(command, tmp_path, seconds=30, max_file_bytes=100) == -signal.SIGXFSZ
    assert (tmp_path / "out.bin").stat().st_size == 100


def test_the_parent_keeps_no_file_limit(tmp_path: Path) -> None:
    command = ("dd", "if=/dev/zero", "of=small.bin", "bs=10", "count=1", "status=none")
    assert run_bounded(command, tmp_path, seconds=30, max_file_bytes=10) == 0
    (tmp_path / "large.bin").write_bytes(b"y" * 1000)
    assert (tmp_path / "large.bin").stat().st_size == 1000


def test_deadline_kills_the_child_and_raises(tmp_path: Path) -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        run_bounded(("sleep", "30"), tmp_path, seconds=1, max_file_bytes=10)


def test_client_error_is_private_and_not_inherited_by_the_launcher(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    command = (sys.executable, "-c", "import os; os.write(2, b'private SQL'); raise SystemExit(3)")
    assert run_bounded(command, tmp_path, seconds=30, max_file_bytes=64) == 3
    assert capfd.readouterr() == ("", "")
    diagnostics = tmp_path / "client-stderr.txt"
    assert diagnostics.read_bytes() == b"private SQL"
    assert diagnostics.stat().st_mode & 0o777 == 0o600


def test_diagnostic_file_cannot_overwrite_or_follow_an_existing_link(tmp_path: Path) -> None:
    target = tmp_path / "keep.txt"
    target.write_bytes(b"unchanged")
    (tmp_path / "client-stderr.txt").symlink_to(target)
    with pytest.raises(FileExistsError):
        run_bounded(("false",), tmp_path, seconds=30, max_file_bytes=64)
    assert target.read_bytes() == b"unchanged"


def test_diagnostics_have_the_same_kernel_size_bound(tmp_path: Path) -> None:
    command = ("sh", "-c", "exec dd if=/dev/zero bs=1000 count=1 status=none >&2")
    assert run_bounded(command, tmp_path, seconds=30, max_file_bytes=100) == -signal.SIGXFSZ
    assert (tmp_path / "client-stderr.txt").read_bytes() == bytes(100)
