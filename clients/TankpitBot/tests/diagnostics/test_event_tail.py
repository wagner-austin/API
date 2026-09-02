"""Following a growing events artifact without re-reading it.

Driven against REAL files on disk. The whole point of this reader is
what it does to a file that another process is appending to, so the
production filesystem seams (``file_marker``, ``read_bytes_from``) are
exactly what these tests exercise -- including the Windows file index
that tells a re-created path from a grown one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.diagnostics.event_tail import EventTail


def _line(second: int, message: str = "tick") -> str:
    """Build one JSONL event line.

    Args:
        second: Seconds field of the timestamp.
        message: Event message.

    Returns:
        One line, without its terminator.
    """
    return (
        f'{{"timestamp":"2026-08-06T20:00:{second:02d}","level":"INFO","logger":"l",'
        f'"mode":"bot","channel":"AI","message":"{message}"}}'
    )


def _append(path: Path, text: str) -> None:
    """Append raw text to an artifact.

    Args:
        path: Artifact path.
        text: Exact bytes to append, terminators included.

    Returns:
        None.
    """
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def test_the_first_read_returns_every_complete_line(tmp_path: Path) -> None:
    """A fresh cursor reads the whole file it is pointed at."""
    path = tmp_path / "latest.events.jsonl"
    _append(path, _line(1) + "\n" + _line(2) + "\n")

    records, restarted = EventTail(path).next_records()

    assert [record["timestamp"] for record in records] == [
        "2026-08-06T20:00:01",
        "2026-08-06T20:00:02",
    ]
    assert restarted is True


def test_a_second_read_returns_only_what_was_appended(tmp_path: Path) -> None:
    """The cursor resumes; it does not re-read what it already folded."""
    path = tmp_path / "latest.events.jsonl"
    _append(path, _line(1) + "\n")
    tail = EventTail(path)
    tail.next_records()

    _append(path, _line(2) + "\n")
    records, restarted = tail.next_records()

    assert [record["timestamp"] for record in records] == ["2026-08-06T20:00:02"]
    assert restarted is False


def test_a_read_with_nothing_new_returns_nothing(tmp_path: Path) -> None:
    """Polling an idle artifact costs a stat and an empty read."""
    path = tmp_path / "latest.events.jsonl"
    _append(path, _line(1) + "\n")
    tail = EventTail(path)
    tail.next_records()

    records, restarted = tail.next_records()

    assert records == []
    assert restarted is False


def test_a_line_still_being_written_is_withheld_until_terminated(
    tmp_path: Path,
) -> None:
    """A poll landing mid-append decodes nothing from the partial line.

    Then the rest arrives and the same line decodes exactly once --
    the half-line is neither dropped nor counted twice.
    """
    path = tmp_path / "latest.events.jsonl"
    complete = _line(5)
    _append(path, complete[: len(complete) // 2])
    tail = EventTail(path)

    mid_write, _ = tail.next_records()
    _append(path, complete[len(complete) // 2 :] + "\n")
    completed, _ = tail.next_records()

    assert mid_write == []
    assert [record["timestamp"] for record in completed] == ["2026-08-06T20:00:05"]


def test_a_multi_byte_character_split_across_reads_survives(tmp_path: Path) -> None:
    """A UTF-8 sequence cut in half by a poll is not corrupted.

    It can only ever sit inside the withheld tail, which is why the
    reader holds back bytes rather than decoding what it has.
    """
    path = tmp_path / "latest.events.jsonl"
    line = _line(6, message="tank ✦ marker").encode("utf-8")
    split = line.index("✦".encode()) + 1
    with path.open("wb") as handle:
        handle.write(line[:split])
    tail = EventTail(path)

    first, _ = tail.next_records()
    with path.open("ab") as handle:
        handle.write(line[split:] + b"\n")
    second, _ = tail.next_records()

    assert first == []
    assert second[0]["message"] == "tank ✦ marker"


def test_a_new_run_under_the_same_path_reports_a_restart(tmp_path: Path) -> None:
    """A replaced artifact is read from its first byte and flagged."""
    path = tmp_path / "latest.events.jsonl"
    _append(path, _line(1) + "\n" + _line(2) + "\n" + _line(3) + "\n")
    tail = EventTail(path)
    tail.next_records()

    path.unlink()
    _append(path, _line(9) + "\n")
    records, restarted = tail.next_records()

    assert restarted is True
    assert [record["timestamp"] for record in records] == ["2026-08-06T20:00:09"]


def test_a_truncated_artifact_is_read_from_the_start(tmp_path: Path) -> None:
    """A file shorter than the cursor cannot be the same run.

    Belt to the file index's braces: a rewrite that reused the same
    file identity would still be caught by the size going backwards.
    """
    path = tmp_path / "latest.events.jsonl"
    _append(path, _line(1) + "\n" + _line(2) + "\n")
    tail = EventTail(path)
    tail.next_records()

    path.write_text(_line(4) + "\n", encoding="utf-8")
    records, restarted = tail.next_records()

    assert restarted is True
    assert [record["timestamp"] for record in records] == ["2026-08-06T20:00:04"]


def test_a_missing_artifact_raises(tmp_path: Path) -> None:
    """An instance that has logged nothing has no file to follow."""
    with pytest.raises(OSError):
        EventTail(tmp_path / "never-written.jsonl").next_records()


def test_a_malformed_line_raises_and_the_cursor_does_not_advance(
    tmp_path: Path,
) -> None:
    """A bad line fails the same way on every poll, never skipped.

    Committing the cursor before the decode would consume the bad line,
    so the next poll would sail past it and fold a run with a hole in
    it -- exactly the silent wrongness the strict decoder exists to
    prevent.
    """
    path = tmp_path / "latest.events.jsonl"
    _append(path, _line(1) + "\n" + '{"not":"an event"}' + "\n")
    tail = EventTail(path)

    with pytest.raises(JSONTypeError):
        tail.next_records()
    with pytest.raises(JSONTypeError):
        tail.next_records()

    # The good line ahead of it is still unread, because nothing was
    # committed: fixing the artifact replays the whole run.
    path.write_text(_line(1) + "\n", encoding="utf-8")
    records, _ = tail.next_records()
    assert [record["timestamp"] for record in records] == ["2026-08-06T20:00:01"]


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    """A stray newline is not an event."""
    path = tmp_path / "latest.events.jsonl"
    _append(path, _line(1) + "\n\n" + _line(2) + "\n")

    records, _ = EventTail(path).next_records()

    assert len(records) == 2
