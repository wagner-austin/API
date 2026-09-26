"""Following an append-only JSONL journal by byte offset, for a bridge.

Lifted out of ``tools/lock-wake`` (its ``position.py`` and the line reader
inside ``journal.read_journal_slice``) when a second bridge,
``tools/fleet-health-wake``, needed the same cursor over a different journal
(MCPs board task ebc80a03). What stays in each bridge is only what is true
of its journal: the line decoder and the announcement.

THE CONTRACT A JOURNAL OFFERS, in the fleet-lock writer's own words: "a
subscriber keeps a byte offset, reads from it, and cannot miss a transition
at any polling interval, because nothing is ever overwritten."

ONE INTEGER IS THE READER'S ENTIRE MEMORY -- the byte offset of the first
unread journal byte -- so the position is a single small JSON object
rewritten whole, not an append-only log that grows forever saying nothing
its last line does not. It lives BESIDE the journal, derived from its path
and the reader's name, so the two files that describe one stream are not
separately addressable and moving one moves both.

TORN TAILS ARE EXPECTED, NOT ERRORS. The writer may be mid-append when the
reader reads, so the bytes after the last newline are a line that does not
exist yet. :func:`read_complete_lines` returns exactly the complete lines
and the offset just past them; the torn tail is read whole next time. The
reader takes BYTES, not text: a text read would decode a torn multi-byte
character into a replacement char and corrupt the offset arithmetic.

THE BRIDGE WRITES THE OFFSET LAST. It posts, then advances, so a crash
between the two repeats a post rather than losing one: at-least-once, the
family order. Nothing here enforces that; each bridge's cycle does, and
says so.

The file operations are PARAMETERS, as :mod:`platform_core.board` takes its
poster: each bridge binds them in its own ``_test_hooks`` (to the real
implementations below in production), so a test rebinds one seam without
this module keeping any state.
"""

from __future__ import annotations

import pathlib
from typing import Protocol

from typing_extensions import TypedDict

from platform_core.json_utils import JSONTypeError, dump_json_str, load_json_str, require_int


class ReadBytesProtocol(Protocol):
    """Read a whole file as raw bytes."""

    def __call__(self, path: pathlib.Path) -> bytes:
        """Read it.

        Args:
            path: Absolute path to read.

        Returns:
            The file's contents, undecoded.
        """
        ...


class WriteTextProtocol(Protocol):
    """Replace a file's contents with one string."""

    def __call__(self, path: pathlib.Path, content: str) -> None:
        """Write it.

        Args:
            path: Absolute path to write.
            content: The full new contents.
        """
        ...


class FileExistsProtocol(Protocol):
    """Report whether a path is an existing file."""

    def __call__(self, path: pathlib.Path) -> bool:
        """Check it.

        Args:
            path: Absolute path to test.

        Returns:
            True when the file exists.
        """
        ...


def read_file_bytes(path: pathlib.Path) -> bytes:
    """Read a real file's bytes: the production :class:`ReadBytesProtocol`.

    Args:
        path: Absolute path to read.

    Returns:
        The file's contents.

    Raises:
        OSError: The file cannot be read.
    """
    return path.read_bytes()


def write_file_text(path: pathlib.Path, content: str) -> None:
    """Replace a real file's contents: the production :class:`WriteTextProtocol`.

    The parent directory is created if absent: a position file is created
    by its first write, and a machine whose bridge has never run is the
    ordinary first-run case rather than a mistake.

    Args:
        path: Absolute path to write.
        content: The full new contents, written with ``\\n`` line endings.

    Raises:
        OSError: The directory or file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def file_is_present(path: pathlib.Path) -> bool:
    """Test a real path: the production :class:`FileExistsProtocol`.

    Args:
        path: Absolute path to test.

    Returns:
        True when it is an existing file.
    """
    return path.is_file()


class JournalLine(TypedDict):
    """One complete, non-blank journal line.

    Attributes:
        number: Its 1-based line number within the slice read, blank lines
            counted, for a decoder's refusal to name.
        text: The line, without its newline.
    """

    number: int
    text: str


class LineSlice(TypedDict):
    """What one read of a journal yielded.

    Attributes:
        lines: Every complete non-blank line at or past the offset, in file
            order.
        next_offset: The byte offset just past the last complete line --
            what the position file records once the lines are announced.
    """

    lines: tuple[JournalLine, ...]
    next_offset: int


def cursor_path(journal: pathlib.Path, reader: str) -> pathlib.Path:
    """Where one reader's offset lives for a given journal.

    Args:
        journal: The journal's path.
        reader: The reading bridge's name, so two bridges following one
            journal keep separate positions.

    Returns:
        ``<journal>.<reader>-offset.json``, beside the journal.
    """
    return journal.parent / f"{journal.name}.{reader}-offset.json"


def read_offset(
    file_exists: FileExistsProtocol, read_bytes: ReadBytesProtocol, path: pathlib.Path
) -> int:
    """Read the first unread byte's offset.

    Args:
        file_exists: The existence test.
        read_bytes: The byte reader.
        path: The position file's path, from :func:`cursor_path`.

    Returns:
        The offset. An absent file reads as 0 rather than raising: a
        machine whose bridge has never run has announced nothing, and
        refusing the first cycle for having no history would make the
        bridge impossible to start.

    Raises:
        InvalidJsonError: A position file that is not JSON at all.
        JSONTypeError: A position file that is JSON but not an offset
            object, or an offset that is negative -- NOT defaulted, because
            a misread position either re-announces history or skips it,
            and both wear the costume of a working bridge.
        OSError: A position file that exists but cannot be read.
    """
    if not file_exists(path):
        return 0
    value = load_json_str(read_bytes(path).decode("utf-8"))
    if not isinstance(value, dict):
        raise JSONTypeError(
            f"{path} is a {type(value).__name__}, not an object; a position that "
            f"cannot be read either re-announces history or skips it"
        )
    offset = require_int(value, "offset")
    if offset < 0:
        raise JSONTypeError(f"{path} holds offset {offset}, which is negative")
    return offset


def write_offset(write_text: WriteTextProtocol, path: pathlib.Path, offset: int) -> None:
    """Record the first unread byte's offset.

    Args:
        write_text: The whole-file writer.
        path: The position file's path, from :func:`cursor_path`.
        offset: The offset just past the last announced line.

    Raises:
        OSError: The position file cannot be written.
    """
    write_text(path, dump_json_str({"offset": offset}) + "\n")


def read_complete_lines(
    file_exists: FileExistsProtocol,
    read_bytes: ReadBytesProtocol,
    journal: pathlib.Path,
    offset: int,
) -> LineSlice:
    """Read every complete journal line at or past a byte offset.

    Args:
        file_exists: The existence test.
        read_bytes: The byte reader.
        journal: The journal's path.
        offset: Byte offset of the first unread byte, from the position
            file; 0 for a bridge that has never run.

    Returns:
        The complete non-blank lines and the offset just past the last
        complete line. An absent journal reads as empty at offset 0 rather
        than raising: a journal's writer creates it on its first event, and
        refusing the first cycle before then would make the bridge
        impossible to start.

    Raises:
        ValueError: A position into a journal that is absent, or past its
            end -- it was deleted, truncated or replaced, and silently
            rewinding would re-announce every event in its history; the
            operator decides, not this reader.
        UnicodeDecodeError: Complete lines that are not UTF-8.
        OSError: A journal that exists but cannot be read.
    """
    if not file_exists(journal):
        if offset != 0:
            raise ValueError(
                f"position says byte {offset} of {journal}, but the journal is absent; "
                f"it was deleted or moved, and rewinding silently would re-announce "
                f"every event in its history"
            )
        return LineSlice(lines=(), next_offset=0)
    data = read_bytes(journal)
    if offset > len(data):
        raise ValueError(
            f"position says byte {offset} of {journal}, but the journal holds only "
            f"{len(data)} bytes; it was truncated or replaced, and rewinding silently "
            f"would re-announce every event in its history"
        )
    window = data[offset:]
    last_newline = window.rfind(b"\n")
    if last_newline == -1:
        return LineSlice(lines=(), next_offset=offset)
    complete = window[: last_newline + 1].decode("utf-8")
    lines = tuple(
        JournalLine(number=number, text=text)
        for number, text in enumerate(complete.splitlines(), start=1)
        if text.strip() != ""
    )
    return LineSlice(lines=lines, next_offset=offset + last_newline + 1)


__all__ = [
    "FileExistsProtocol",
    "JournalLine",
    "LineSlice",
    "ReadBytesProtocol",
    "WriteTextProtocol",
    "cursor_path",
    "file_is_present",
    "read_complete_lines",
    "read_file_bytes",
    "read_offset",
    "write_file_text",
    "write_offset",
]
