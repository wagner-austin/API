"""Tests for the production hook implementations.

These are the real subprocess and filesystem calls every other test replaces
with a fake. Exercising them here means the seam is verified rather than
merely assumed: a fake that matches a protocol the real implementation does
not satisfy would otherwise pass the whole suite.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

import pytest

from hpc3.cli import _test_hooks as cli_hooks
from hpc3.core import _test_hooks as core_hooks


class TestRealRun:
    def test_it_captures_stdout_and_a_zero_exit(self) -> None:
        result = core_hooks.run([sys.executable, "-c", "print('hello')"])
        assert result["returncode"] == 0
        assert "hello" in result["stdout"]
        assert result["stderr"] == ""

    def test_it_returns_a_non_zero_exit_rather_than_raising(self) -> None:
        """The caller decides what a failure means; the runner does not."""
        result = core_hooks.run([sys.executable, "-c", "raise SystemExit(3)"])
        assert result["returncode"] == 3

    def test_it_captures_stderr(self) -> None:
        result = core_hooks.run([sys.executable, "-c", "import sys; sys.stderr.write('bad')"])
        assert "bad" in result["stderr"]

    def test_it_writes_stdin_bytes_to_the_process(self) -> None:
        result = core_hooks.run(
            [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"],
            stdin_bytes=b"piped",
        )
        assert "piped" in result["stdout"]

    def test_undecodable_output_is_replaced_not_lost(self) -> None:
        """A mangled character in a diagnostic beats losing the diagnostic."""
        result = core_hooks.run(
            [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'\\xff\\xfe')"]
        )
        assert result["returncode"] == 0
        assert result["stdout"] != ""


class TestRealFileHooks:
    def test_read_bytes_returns_exact_bytes(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "f.bin"
        path.write_bytes(b"\x00\x01\r\n")
        assert core_hooks.read_bytes(path) == b"\x00\x01\r\n"

    def test_file_exists_distinguishes_files_from_directories(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "f.txt"
        path.write_text("x", encoding="utf-8")
        assert core_hooks.file_exists(path) is True
        assert core_hooks.file_exists(tmp_path) is False
        assert core_hooks.file_exists(tmp_path / "absent") is False


class TestHookRebinding:
    def test_core_reset_restores_every_default(self) -> None:
        def _other_run(
            argv: Sequence[str], *, stdin_bytes: bytes | None = None
        ) -> core_hooks.CommandResult:
            return core_hooks.CommandResult(returncode=0, stdout="", stderr="")

        core_hooks.run = _other_run
        core_hooks.reset_hooks()
        assert core_hooks.run is not _other_run
        assert core_hooks.run([sys.executable, "-c", "pass"])["returncode"] == 0

    def test_cli_emit_writes_to_stdout_and_resets(self, capsys: pytest.CaptureFixture[str]) -> None:
        cli_hooks.emit("a line")
        assert capsys.readouterr().out == "a line\n"

        captured: list[str] = []
        cli_hooks.emit = captured.append
        cli_hooks.emit("held")
        assert captured == ["held"]

        cli_hooks.reset_hooks()
        cli_hooks.emit("after reset")
        assert capsys.readouterr().out == "after reset\n"
        assert captured == ["held"]
