"""CLI behaviour, driven through the real code with injected I/O hooks.

Output capture uses an explicitly typed context manager rather than a pytest
fixture: ``pytest.fixture`` is an overloaded callable, so decorating with it
produces an expression containing ``Any`` and fails strict type checking.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import TracebackType

import pytest

from rw_bot.harness import _test_hooks
from rw_bot.harness.boot_log import BootLogError
from rw_bot.harness.boot_log_cli import (
    EXIT_BAD_USAGE,
    EXIT_CRASHED,
    EXIT_OK,
    main,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CLEAN_LOG = _PROJECT_ROOT / "wiki" / "sources" / "m0-probe" / "nodisplay-boot.log"
_CRASH_LOG = _PROJECT_ROOT / "wiki" / "sources" / "m1-sandbox" / "sandbox-crash.log"


class _CapturedOutput:
    """Bind the ``write_line`` hook to a recorder for the duration of a block.

    Attributes:
        lines: Every line the CLI wrote while the block was active, in order.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []
        self._original: _test_hooks.WriteLineProto = _test_hooks.write_line

    def __call__(self, text: str) -> None:
        """Record one written line.

        Args:
            text: Line content, without a trailing newline.
        """
        self.lines.append(text)

    def __enter__(self) -> _CapturedOutput:
        """Install this recorder as the output hook.

        Returns:
            This recorder.
        """
        self._original = _test_hooks.write_line
        _test_hooks.write_line = self
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore the original output hook.

        Args:
            exc_type: Exception class raised in the block, if any.
            exc: Exception raised in the block, if any.
            traceback: Traceback of the raised exception, if any.
        """
        _test_hooks.write_line = self._original


def test_clean_log_exits_zero() -> None:
    with _CapturedOutput() as out:
        assert main([str(_CLEAN_LOG)]) == EXIT_OK
    assert "  crashes        0" in out.lines


def test_clean_log_reports_the_build() -> None:
    with _CapturedOutput() as out:
        main([str(_CLEAN_LOG)])
    assert "  build          1.15 (code 176, build #28)" in out.lines


def test_clean_log_reports_the_recovered_class_mapping() -> None:
    with _CapturedOutput() as out:
        main([str(_CLEAN_LOG)])
    assert any("gameEngine -> com.corrodinggames.rts.game.i" in line for line in out.lines)


def test_crash_log_exits_one_and_names_the_frame() -> None:
    with _CapturedOutput() as out:
        assert main([str(_CRASH_LOG)]) == EXIT_CRASHED
    assert any("EnableScissorRegion" in line for line in out.lines)


def test_no_arguments_prints_usage() -> None:
    with _CapturedOutput() as out:
        assert main([]) == EXIT_BAD_USAGE
    assert out.lines == ["usage: rw-boot-log <path-to-engine-log>"]


def test_two_arguments_prints_usage() -> None:
    with _CapturedOutput() as out:
        assert main(["a", "b"]) == EXIT_BAD_USAGE
    assert out.lines == ["usage: rw-boot-log <path-to-engine-log>"]


def test_unparseable_log_propagates_rather_than_returning_a_code(tmp_path: Path) -> None:
    bad = tmp_path / "empty.log"
    bad.write_text("nothing useful here\n", encoding="utf-8")
    with _CapturedOutput() as out, pytest.raises(BootLogError) as caught:
        main([str(bad)])
    assert caught.value.code == "RW-BOOTLOG-001"
    assert out.lines == []


def test_argv_is_read_from_the_hook_when_not_supplied() -> None:
    original = _test_hooks.read_argv
    _test_hooks.read_argv = lambda: [str(_CLEAN_LOG)]
    try:
        with _CapturedOutput() as out:
            assert main(None) == EXIT_OK
    finally:
        _test_hooks.read_argv = original
    assert "  crashes        0" in out.lines


def test_read_text_lines_hook_reads_a_real_file(tmp_path: Path) -> None:
    target = tmp_path / "sample.log"
    target.write_text("first\nsecond\n", encoding="utf-8")
    assert _test_hooks.read_text_lines(target) == ("first", "second")


def test_read_text_lines_hook_rejects_non_utf8_without_softening(tmp_path: Path) -> None:
    target = tmp_path / "latin.log"
    target.write_bytes(b"\xff\xfe invalid")
    with pytest.raises(UnicodeDecodeError):
        _test_hooks.read_text_lines(target)


def test_write_line_hook_writes_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    _test_hooks.write_line("emitted")
    assert capsys.readouterr().out == "emitted\n"


def test_read_argv_hook_excludes_the_program_name() -> None:
    original = sys.argv
    sys.argv = ["rw-boot-log", "first", "second"]
    try:
        assert _test_hooks.read_argv() == ["first", "second"]
    finally:
        sys.argv = original
