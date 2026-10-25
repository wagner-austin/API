"""The append-only reader decodes each line once (MCPs board task a1f9d16d).

Every case reads a real file the test writes, through a decoder that records
each line it is handed, so "decoded once" is counted rather than inferred.
"""

from __future__ import annotations

import pathlib
import threading

import pytest
from platform_core.json_utils import InvalidJsonError, load_json_str

from fleet.core import record_tail

_READERS = 8


class _Decoder:
    """Decodes ``{"n": <int>}`` lines and records every call it receives."""

    def __init__(self) -> None:
        """Start with no calls."""
        self.calls: list[tuple[int, str]] = []
        self._lock = threading.Lock()

    def __call__(self, path: pathlib.Path, number: int, line: str) -> int:
        """Decode one line.

        Args:
            path: The file, for the refusal.
            number: The line's number in the file.
            line: The line.

        Returns:
            Its ``n``.

        Raises:
            InvalidJsonError: For a line that is not JSON.
            ValueError: For JSON that is not a ``{"n": <int>}`` object.
        """
        with self._lock:
            self.calls.append((number, line))
        value = load_json_str(line)
        n = value.get("n") if isinstance(value, dict) else None
        if not isinstance(n, int):
            raise ValueError(f"{path} line {number} holds no n")
        return n


def _append(path: pathlib.Path, text: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _rows(count: int, *, start: int = 1) -> str:
    return "".join(f'{{"n": {n}}}\n' for n in range(start, start + count))


def test_an_absent_file_reads_as_empty(tmp_path: pathlib.Path) -> None:
    decoder = _Decoder()

    assert record_tail.RecordTail(decoder).read(tmp_path / "ledger.jsonl") == ()
    assert decoder.calls == []


def test_a_second_read_of_an_unchanged_file_decodes_nothing(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "ledger.jsonl"
    _append(path, _rows(3))
    decoder = _Decoder()
    tail = record_tail.RecordTail(decoder)

    first = tail.read(path)
    second = tail.read(path)

    assert {"first": first, "second": second, "calls": len(decoder.calls)} == {
        "first": (1, 2, 3),
        "second": (1, 2, 3),
        "calls": 3,
    }


def test_only_the_lines_appended_since_the_last_read_are_decoded(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "ledger.jsonl"
    _append(path, _rows(2))
    decoder = _Decoder()
    tail = record_tail.RecordTail(decoder)
    tail.read(path)
    _append(path, _rows(2, start=3))

    assert tail.read(path) == (1, 2, 3, 4)
    assert decoder.calls == [
        (1, '{"n": 1}'),
        (2, '{"n": 2}'),
        (3, '{"n": 3}'),
        (4, '{"n": 4}'),
    ]


def test_a_line_not_yet_ended_is_read_once_its_newline_lands(tmp_path: pathlib.Path) -> None:
    """An appender mid-write leaves a line with no newline; it is not malformed."""
    path = tmp_path / "ledger.jsonl"
    _append(path, _rows(1) + '{"n": ')
    decoder = _Decoder()
    tail = record_tail.RecordTail(decoder)

    before = tail.read(path)
    _append(path, "2}\n")

    assert {"before": before, "after": tail.read(path), "calls": [c[0] for c in decoder.calls]} == {
        "before": (1,),
        "after": (1, 2),
        "calls": [1, 2],
    }


def test_a_file_of_one_unended_line_reads_as_empty(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "ledger.jsonl"
    _append(path, '{"n": 1}')
    decoder = _Decoder()

    assert record_tail.RecordTail(decoder).read(path) == ()
    assert decoder.calls == []


def test_line_numbers_count_blank_lines_across_reads(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "ledger.jsonl"
    _append(path, '{"n": 1}\n\n')
    decoder = _Decoder()
    tail = record_tail.RecordTail(decoder)
    tail.read(path)
    _append(path, '\n{"n": 2}\n')

    assert tail.read(path) == (1, 2)
    assert [number for number, _ in decoder.calls] == [1, 4]


def test_a_replaced_file_is_decoded_whole(tmp_path: pathlib.Path) -> None:
    """Its first row differs, so nothing decoded from the old file is kept."""
    path = tmp_path / "ledger.jsonl"
    _append(path, _rows(2))
    decoder = _Decoder()
    tail = record_tail.RecordTail(decoder)
    tail.read(path)
    path.write_text(_rows(3, start=7), encoding="utf-8")

    assert tail.read(path) == (7, 8, 9)
    assert [line for _, line in decoder.calls[2:]] == ['{"n": 7}', '{"n": 8}', '{"n": 9}']


def test_a_truncated_file_is_decoded_whole(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "ledger.jsonl"
    _append(path, _rows(3))
    decoder = _Decoder()
    tail = record_tail.RecordTail(decoder)
    tail.read(path)
    path.write_text(_rows(1), encoding="utf-8")

    assert tail.read(path) == (1,)
    assert len(decoder.calls) == 4


def test_a_file_whose_last_read_bytes_changed_is_decoded_whole(tmp_path: pathlib.Path) -> None:
    """The same first row and length, a different row where the last read ended."""
    path = tmp_path / "ledger.jsonl"
    _append(path, _rows(2))
    decoder = _Decoder()
    tail = record_tail.RecordTail(decoder)
    tail.read(path)
    path.write_text('{"n": 1}\n{"n": 5}\n{"n": 6}\n', encoding="utf-8")

    assert tail.read(path) == (1, 5, 6)
    assert len(decoder.calls) == 5


def test_a_file_that_disappears_is_forgotten(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "ledger.jsonl"
    _append(path, _rows(2))
    decoder = _Decoder()
    tail = record_tail.RecordTail(decoder)
    tail.read(path)
    path.unlink()
    gone = tail.read(path)
    _append(path, _rows(2))

    assert {"gone": gone, "again": tail.read(path), "calls": len(decoder.calls)} == {
        "gone": (),
        "again": (1, 2),
        "calls": 4,
    }


def test_a_line_that_does_not_decode_keeps_nothing_and_raises_again(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    _append(path, _rows(1))
    decoder = _Decoder()
    tail = record_tail.RecordTail(decoder)
    tail.read(path)
    _append(path, '{"n": 2}\nnot json\n')

    with pytest.raises(InvalidJsonError):
        tail.read(path)
    with pytest.raises(InvalidJsonError):
        tail.read(path)
    assert decoder.calls[1:] == [(2, '{"n": 2}'), (3, "not json"), (2, '{"n": 2}'), (3, "not json")]


def test_a_line_that_is_not_utf8_is_refused(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b'{"n": 1}\n\xff\n')

    with pytest.raises(UnicodeDecodeError):
        record_tail.RecordTail(_Decoder()).read(path)


def test_two_files_are_read_apart(tmp_path: pathlib.Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    feed = tmp_path / "feed.jsonl"
    _append(ledger, _rows(2))
    _append(feed, _rows(1, start=5))
    tail = record_tail.RecordTail(_Decoder())

    assert (tail.read(ledger), tail.read(feed), tail.read(ledger)) == ((1, 2), (5,), (1, 2))


def test_readers_on_many_threads_see_every_row_and_decode_each_once(
    tmp_path: pathlib.Path,
) -> None:
    """The watch's check threads and the serving loop read one ledger at once."""
    path = tmp_path / "ledger.jsonl"
    _append(path, _rows(200))
    decoder = _Decoder()
    tail = record_tail.RecordTail(decoder)
    results: list[tuple[int, ...]] = []
    results_lock = threading.Lock()
    start = threading.Barrier(_READERS)

    def reader() -> None:
        start.wait()
        rows = tail.read(path)
        with results_lock:
            results.append(rows)

    threads = [threading.Thread(target=reader) for _ in range(_READERS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [tuple(range(1, 201))] * _READERS
    assert len(decoder.calls) == 200
