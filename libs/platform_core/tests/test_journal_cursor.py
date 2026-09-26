"""The journal cursor, against REAL files under ``tmp_path``.

The cases that decide whether a cursor is correct are the ones a journal's
writer makes real: a torn tail mid-append, blank lines, a journal not yet
created, and a position that no longer fits the file.
"""

from __future__ import annotations

import pathlib

import pytest

from platform_core.journal_cursor import (
    cursor_path,
    file_is_present,
    read_complete_lines,
    read_file_bytes,
    read_offset,
    write_file_text,
    write_offset,
)
from platform_core.json_utils import InvalidJsonError, JSONTypeError

LINE_A = '{"ts":"2026-09-26T06:14:13.504Z","kind":"transitions"}\n'
LINE_B = '{"ts":"2026-09-26T06:34:02.117Z","kind":"refused"}\n'


def _stage(tmp_path: pathlib.Path, content: bytes) -> pathlib.Path:
    """Write a journal with exact bytes.

    Args:
        tmp_path: The test's temporary directory.
        content: The journal's full contents, torn tails included.

    Returns:
        The journal's path.
    """
    journal = tmp_path / "events.jsonl"
    journal.write_bytes(content)
    return journal


def _read(journal: pathlib.Path, offset: int) -> tuple[list[str], int]:
    """Read through the production file operations.

    Args:
        journal: The journal's path.
        offset: The offset to read from.

    Returns:
        The lines' texts and the next offset.
    """
    result = read_complete_lines(file_is_present, read_file_bytes, journal, offset)
    return [line["text"] for line in result["lines"]], result["next_offset"]


class TestFileOperations:
    def test_read_file_bytes_reads_raw_bytes(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "journal.jsonl"
        path.write_bytes(b'{"k":1}\n{"torn')
        assert read_file_bytes(path) == b'{"k":1}\n{"torn'

    def test_write_file_text_replaces_and_creates_parents(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "deep" / "offset.json"
        write_file_text(path, '{"offset": 1}\n')
        write_file_text(path, '{"offset": 2}\n')
        assert path.read_bytes() == b'{"offset": 2}\n'

    def test_file_is_present_distinguishes_files_from_absence(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "present.json"
        assert file_is_present(path) is False
        path.write_text("{}", encoding="utf-8")
        assert file_is_present(path) is True
        assert file_is_present(tmp_path) is False


class TestCursorPath:
    def test_derives_beside_the_journal_per_reader(self) -> None:
        journal = pathlib.Path("C:/repo/.fleet-events.jsonl")
        assert cursor_path(journal, "lock-wake") == pathlib.Path(
            "C:/repo/.fleet-events.jsonl.lock-wake-offset.json"
        )
        assert cursor_path(journal, "other") != cursor_path(journal, "lock-wake")


class TestOffset:
    def test_an_absent_file_reads_as_zero(self, tmp_path: pathlib.Path) -> None:
        assert read_offset(file_is_present, read_file_bytes, tmp_path / "never.json") == 0

    def test_round_trips_what_write_offset_wrote(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "offset.json"
        write_offset(write_file_text, path, 16952)
        assert path.read_text(encoding="utf-8") == '{"offset":16952}\n'
        assert read_offset(file_is_present, read_file_bytes, path) == 16952

    def test_a_non_object_is_fatal(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "offset.json"
        path.write_text("42\n", encoding="utf-8")
        with pytest.raises(JSONTypeError, match="is a int, not an object"):
            read_offset(file_is_present, read_file_bytes, path)

    def test_a_negative_offset_is_fatal(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "offset.json"
        path.write_text('{"offset": -3}\n', encoding="utf-8")
        with pytest.raises(JSONTypeError, match="holds offset -3, which is negative"):
            read_offset(file_is_present, read_file_bytes, path)

    def test_a_file_that_is_not_json_is_fatal(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "offset.json"
        path.write_text("{offset", encoding="utf-8")
        with pytest.raises(InvalidJsonError):
            read_offset(file_is_present, read_file_bytes, path)


class TestReadCompleteLines:
    def test_reads_every_complete_line_and_reports_the_next_offset(
        self, tmp_path: pathlib.Path
    ) -> None:
        content = (LINE_A + LINE_B).encode("utf-8")
        assert _read(_stage(tmp_path, content), 0) == (
            [LINE_A.rstrip("\n"), LINE_B.rstrip("\n")],
            len(content),
        )

    def test_starts_at_the_given_offset(self, tmp_path: pathlib.Path) -> None:
        first = LINE_A.encode("utf-8")
        journal = _stage(tmp_path, first + LINE_B.encode("utf-8"))
        assert _read(journal, len(first)) == ([LINE_B.rstrip("\n")], len(first) + len(LINE_B))

    def test_leaves_a_torn_tail_for_the_next_read(self, tmp_path: pathlib.Path) -> None:
        complete = LINE_A.encode("utf-8")
        journal = _stage(tmp_path, complete + b'{"ts":"2026-09-26T06:3')
        assert _read(journal, 0) == ([LINE_A.rstrip("\n")], len(complete))

    def test_a_torn_multibyte_character_is_left_whole_for_the_next_read(
        self, tmp_path: pathlib.Path
    ) -> None:
        complete = LINE_A.encode("utf-8")
        dash = "\u2014".encode()
        journal = _stage(tmp_path, complete + b'{"body":"a ' + dash[:2])
        assert _read(journal, 0) == ([LINE_A.rstrip("\n")], len(complete))

    def test_a_wholly_torn_window_yields_nothing_and_holds_position(
        self, tmp_path: pathlib.Path
    ) -> None:
        complete = LINE_A.encode("utf-8")
        journal = _stage(tmp_path, complete + b'{"ts":"2026')
        assert _read(journal, len(complete)) == ([], len(complete))

    def test_blank_lines_are_skipped_but_counted(self, tmp_path: pathlib.Path) -> None:
        content = (LINE_A + "\n  \n" + LINE_B).encode("utf-8")
        result = read_complete_lines(file_is_present, read_file_bytes, _stage(tmp_path, content), 0)
        assert [line["number"] for line in result["lines"]] == [1, 4]
        assert result["next_offset"] == len(content)

    def test_an_absent_journal_reads_empty_at_offset_zero(self, tmp_path: pathlib.Path) -> None:
        assert read_complete_lines(
            file_is_present, read_file_bytes, tmp_path / "never.jsonl", 0
        ) == {"lines": (), "next_offset": 0}

    def test_an_absent_journal_with_a_position_refuses(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match=r"position says byte 100 of .*the journal is absent"):
            read_complete_lines(file_is_present, read_file_bytes, tmp_path / "never.jsonl", 100)

    def test_a_position_past_the_end_refuses(self, tmp_path: pathlib.Path) -> None:
        journal = _stage(tmp_path, LINE_A.encode("utf-8"))
        with pytest.raises(
            ValueError, match=f"holds only {len(LINE_A)} bytes; it was truncated or replaced"
        ):
            read_complete_lines(file_is_present, read_file_bytes, journal, 10_000)
