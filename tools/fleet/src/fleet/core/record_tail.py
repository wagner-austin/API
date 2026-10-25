"""Reading an append-only record file, decoding each of its lines once (MCPs board task a1f9d16d).

WHY. Every node runner on the hub asked the ledger which runs are live once
for each run it holds, every ``node_poll_seconds`` (5 s), and once more in
each claim's gate, and each ask decoded the whole file. Measured on
diphtheria on 2026-10-25: ``runs/ledger.jsonl`` held 4.9 MB, 12,210 rows for
6,106 runs, and one :func:`fleet.core.records.latest_rows` took 293 ms of
cpu in the fleet venv; py-spy over all ten node runners for 120 s put
54.9 percent of their 71.3 cpu-seconds in that read, about a third of a core
spent decoding rows that had not changed since the last poll. The files are
appended and never rewritten (:mod:`fleet.core.records`), so all but the
lines appended since the last read decode to what they decoded to then.

HOW. A :class:`RecordTail` keeps, for each file it has read, where its
decoded lines end, a sample of the bytes before that end, and the rows they
decoded to. Each read still reads the whole file, which costs under a
millisecond from the page cache at that size, and when the sample is still
what the file holds there, only the lines after the end are decoded. A file
that is shorter than that end, or whose bytes there differ, was truncated or
replaced, and is decoded whole, as every read did before. Lines are cut by
:func:`platform_core.journal_cursor.complete_lines_after`, the cut the
board's journal bridges make, so a line an appender has begun and not yet
ended with its newline is read on the next pass instead of being refused as
malformed; a complete line that does not decode is still fatal, with its
line number in the file.

WHAT THE SAMPLE DOES NOT SEE. An edit that keeps every byte the sample
covers and the file's length at least what was read changes an earlier
line without being noticed. No writer of these files edits a line: every
one appends in its turn (:data:`fleet.core.records.APPEND_TURN`), and the
sample is the head and the tail of what was read, so a file replaced by
another, which begins with a different row, is decoded whole.

ONE TAIL PER KIND OF RECORD, SHARED BY THE PROCESS. A node runner reads the
ledger from its serving loop and from the watch's check threads at once, so
a tail's state is updated under its lock.
"""

from __future__ import annotations

import pathlib
import threading
from collections.abc import Callable
from typing import Final, Generic, TypeVar

from platform_core.journal_cursor import complete_lines_after
from typing_extensions import TypedDict

from fleet.core import _test_hooks

RowT = TypeVar("RowT")

#: How many bytes at each end of what was read are compared before the
#: rows decoded from it are kept: one row of the ledger is about 400 bytes,
#: so each sample spans several whole rows.
SAMPLE_BYTES: Final[int] = 4096


class _Read(TypedDict, Generic[RowT]):
    """What one file decoded to, as far as its last complete line.

    Attributes:
        end: The byte offset just past the last complete line decoded.
        lines: How many lines, blank ones included, lie before ``end``, so a
            later line's number is its number in the file.
        head: The file's first bytes, at most :data:`SAMPLE_BYTES` of them.
        tail: The bytes just before ``end``, at most :data:`SAMPLE_BYTES`.
        rows: Every row the lines before ``end`` decoded to, in file order.
    """

    end: int
    lines: int
    head: bytes
    tail: bytes
    rows: tuple[RowT, ...]


def _still_holds(data: bytes, held: _Read[RowT]) -> bool:
    """Whether a file's bytes still begin with what was decoded from it.

    Args:
        data: The file's bytes now.
        held: What an earlier read decoded.

    Returns:
        True when the file is at least as long as what was read and holds
        the same bytes at both ends of it.
    """
    end = held["end"]
    return (
        len(data) >= end
        and data[: len(held["head"])] == held["head"]
        and data[end - len(held["tail"]) : end] == held["tail"]
    )


class RecordTail(Generic[RowT]):
    """The rows of append-only files of one kind, each line decoded once.

    Args:
        decode: Turns one complete non-blank line into a row, given the file
            and the line's 1-based number in it for its refusal.
    """

    def __init__(self, decode: Callable[[pathlib.Path, int, str], RowT]) -> None:
        """Start with nothing read.

        Args:
            decode: The line decoder.
        """
        self._decode = decode
        self._lock = threading.Lock()
        self._files: dict[pathlib.Path, _Read[RowT]] = {}

    def read(self, path: pathlib.Path) -> tuple[RowT, ...]:
        """Every row the file holds, in the order it was appended.

        Args:
            path: The record file.

        Returns:
            One row per complete non-blank line. An absent file reads as
            empty: the first append creates it.

        What the decoder raises for a line propagates unchanged, and nothing
        is kept from a read that raised, so the next read decodes the same
        line again and raises again.

        Raises:
            UnicodeDecodeError: A complete line that is not UTF-8.
        """
        if not _test_hooks.file_exists(path):
            with self._lock:
                self._files.pop(path, None)
            return ()
        data = _test_hooks.read_bytes(path)
        with self._lock:
            held = self._files.get(path)
            start, lines, rows = (
                (held["end"], held["lines"], held["rows"])
                if held is not None and _still_holds(data, held)
                else (0, 0, ())
            )
            window = complete_lines_after(data[start:], start)
            end = window["next_offset"]
            if end == start:
                return rows
            rows = rows + tuple(
                self._decode(path, lines + line["number"], line["text"]) for line in window["lines"]
            )
            self._files[path] = _Read(
                end=end,
                lines=lines + len(data[start:end].decode("utf-8").splitlines()),
                head=data[: min(end, SAMPLE_BYTES)],
                tail=data[max(0, end - SAMPLE_BYTES) : end],
                rows=rows,
            )
            return rows


__all__ = ["SAMPLE_BYTES", "RecordTail"]
