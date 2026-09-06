"""The production hook implementations, exercised directly.

These are the bindings a scheduled run actually uses, so they are tested
against the real filesystem and the real clock rather than described. A seam
whose production side is only ever replaced by a fake is a seam whose
production side is untested.
"""

from __future__ import annotations

import pathlib
import time

import pytest

from fleet_wake import _test_hooks


class TestFileHooks:
    def test_append_then_read_round_trips_through_a_real_file(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "announced.jsonl"

        _test_hooks.append_text(path, "one")
        _test_hooks.append_text(path, "two")

        assert _test_hooks.read_text(path) == "one\ntwo\n"

    def test_the_line_ending_is_lf_on_every_platform(self, tmp_path: pathlib.Path) -> None:
        """Read back as BYTES. A CRLF written on Windows would still read as
        ``\\n`` through the text hook, so a text assertion cannot see it --
        and the file is a record other tools parse."""
        path = tmp_path / "announced.jsonl"

        _test_hooks.append_text(path, "one")

        assert path.read_bytes() == b"one\n"

    def test_a_missing_parent_directory_is_created(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "runs" / "announced.jsonl"

        _test_hooks.append_text(path, "one")

        assert _test_hooks.read_text(path) == "one\n"

    def test_file_exists_answers_for_files_directories_and_absence(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A directory is not a file. Answering True for one would make the
        reader try to read it and fail with an OSError instead of treating a
        fresh workspace as empty."""
        path = tmp_path / "announced.jsonl"
        path.write_text("", encoding="utf-8")

        assert _test_hooks.file_exists(path) is True
        assert _test_hooks.file_exists(tmp_path) is False
        assert _test_hooks.file_exists(tmp_path / "absent.jsonl") is False


class TestReportAndClock:
    def test_emit_writes_one_flushed_line_to_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _test_hooks.emit("a line")

        assert capsys.readouterr().out == "a line\n"

    def test_now_tracks_the_real_clock_in_whole_seconds(self) -> None:
        """The fleet ledger records ``started_unix`` and ``ended_unix`` as
        whole seconds, so a position row has to be comparable against them.

        The type is mypy's to guarantee and is not re-checked here. What a
        type cannot say is that the value is the CURRENT time rather than a
        constant, a millisecond count, or a monotonic counter -- each of which
        type-checks and each of which would put a position row decades from
        the dispatch it records.
        """
        before = int(time.time())
        stamp = _test_hooks.now()
        after = int(time.time())

        assert before <= stamp <= after
        assert stamp == stamp // 1


class TestReset:
    def test_reset_restores_every_default(self) -> None:
        """Every hook, not a sample. The autouse fixture calls this between
        tests, so a hook it forgot would leak one test's fake into the next."""
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

        _test_hooks.emit = _capture
        _test_hooks.now = _frozen
        _test_hooks.file_exists = _absent
        _test_hooks.read_text = _empty
        _test_hooks.append_text = _swallow

        _test_hooks.reset_hooks()

        assert _test_hooks.emit is not _capture
        assert _test_hooks.now is not _frozen
        assert _test_hooks.file_exists is not _absent
        assert _test_hooks.read_text is not _empty
        assert _test_hooks.append_text is not _swallow
        assert _test_hooks.now() != 0
