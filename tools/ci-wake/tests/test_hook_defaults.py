"""The production hook implementations, exercised directly.

These are the bindings a scheduled run actually uses, so they are tested
against the real filesystem, the real clock and a real subprocess rather
than described. A seam whose production side is only ever replaced by a fake
is a seam whose production side is untested.
"""

from __future__ import annotations

import pathlib
import sys
import time
from collections.abc import Sequence

import pytest

from ci_wake import _test_hooks


class TestFileHooks:
    def test_append_then_read_round_trips_through_a_real_file(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "pushes.jsonl"

        _test_hooks.append_text(path, "one")
        _test_hooks.append_text(path, "two")

        assert _test_hooks.read_text(path) == "one\ntwo\n"

    def test_the_line_ending_is_lf_on_every_platform(self, tmp_path: pathlib.Path) -> None:
        r"""Read back as BYTES. A CRLF written on Windows would still read as
        ``\n`` through the text hook, so a text assertion cannot see it --
        and both records are parsed by other tools."""
        path = tmp_path / "pushes.jsonl"

        _test_hooks.append_text(path, "one")

        assert path.read_bytes() == b"one\n"

    def test_a_missing_parent_directory_is_created(self, tmp_path: pathlib.Path) -> None:
        """The enrolment record's first write is made by a git hook on a
        machine that has never run the bridge, so the ordinary first-run case
        is an absent directory rather than a mistake."""
        path = tmp_path / "runs" / "pushes.jsonl"

        _test_hooks.append_text(path, "one")

        assert _test_hooks.read_text(path) == "one\n"

    def test_file_exists_answers_for_files_directories_and_absence(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A directory is not a file. Answering True for one would make the
        reader try to read it and fail with an OSError instead of treating a
        fresh machine as having enrolled nothing."""
        path = tmp_path / "pushes.jsonl"
        path.write_text("", encoding="utf-8")

        assert _test_hooks.file_exists(path) is True
        assert _test_hooks.file_exists(tmp_path) is False
        assert _test_hooks.file_exists(tmp_path / "absent.jsonl") is False


class TestRunProcess:
    def test_it_captures_a_real_process_s_output_and_status(self) -> None:
        """Against a real subprocess, because the whole ``gh`` boundary is
        three fields off a finished process and a fake proves nothing about
        which three :mod:`subprocess` actually populates."""
        completed = _test_hooks.run_process(
            [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert completed.stdout == "out\n"
        assert completed.stderr == "err\n"
        assert completed.returncode == 0

    def test_a_non_zero_exit_is_returned_rather_than_raised(self) -> None:
        """``check=True`` would raise a ``CalledProcessError`` whose message
        is the argv and nothing else, and the useful half -- the CLI's own
        words on stderr -- would have to be recovered from an attribute."""
        completed = _test_hooks.run_process(
            [sys.executable, "-c", "import sys; sys.exit(7)"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert completed.returncode == 7


class TestReportAndClock:
    def test_emit_writes_one_flushed_line_to_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _test_hooks.emit("a line")

        assert capsys.readouterr().out == "a line\n"

    def test_now_tracks_the_real_clock_in_whole_seconds(self) -> None:
        """The type is mypy's to guarantee and is not re-checked here. What a
        type cannot say is that the value is the CURRENT time rather than a
        constant, a millisecond count, or a monotonic counter -- each of which
        type-checks, and each of which would make every abandonment horizon
        fire immediately or never.
        """
        before = int(time.time())
        stamp = _test_hooks.now()
        after = int(time.time())

        assert before <= stamp <= after


class TestReset:
    def test_reset_restores_every_default(self) -> None:
        """Every hook, not a sample. The autouse fixture calls this between
        tests, so a hook it forgot would leak one test's fake into the next.
        """
        held: list[str] = []

        def _capture(line: str) -> None:
            held.append(line)

        def _frozen() -> int:
            return 0

        def _absent(path: pathlib.Path) -> bool:
            return False

        def _empty(path: pathlib.Path) -> str:
            return ""

        def _swallow(path: pathlib.Path, line: str) -> None:
            held.append(line)

        def _no_process(
            args: Sequence[str], *, capture_output: bool, text: bool, timeout: int
        ) -> _test_hooks.CompletedProto:
            raise AssertionError("should have been reset")

        original_post = _test_hooks.http_post
        _test_hooks.emit = _capture
        _test_hooks.now = _frozen
        _test_hooks.file_exists = _absent
        _test_hooks.read_text = _empty
        _test_hooks.append_text = _swallow
        _test_hooks.run_process = _no_process

        _test_hooks.reset_hooks()

        assert _test_hooks.emit is not _capture
        assert _test_hooks.now is not _frozen
        assert _test_hooks.file_exists is not _absent
        assert _test_hooks.read_text is not _empty
        assert _test_hooks.append_text is not _swallow
        assert _test_hooks.run_process is not _no_process
        assert _test_hooks.http_post is original_post
        assert _test_hooks.now() != 0
