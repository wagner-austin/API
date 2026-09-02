"""Follow a growing JSONL events artifact without re-reading it.

A live bot appends to ``runs/bot/<instance>/latest.events.jsonl`` for
the whole session, so any reader that polls it -- the fleet control
page does, every second -- must choose between re-reading megabytes it
has already seen and remembering where it stopped. This reader
remembers.

Two things make that safe rather than merely fast:

* **Line boundaries.** A poll can land mid-append, so the trailing
  bytes after the last newline are held back until the rest of the
  line arrives. A partial line is never decoded, and a multi-byte
  UTF-8 character split across two polls stays intact because it can
  only ever sit inside that withheld tail.
* **Run identity.** A new session re-creates the same path, which
  would otherwise leave the reader's byte offset pointing into the
  middle of a different run. The file's identity (its filesystem file
  number) is compared on every read, so a replaced file is REPORTED as
  a restart rather than silently mixed with the previous run's
  records.
"""

from __future__ import annotations

from pathlib import Path

from tankpit_bot import _test_hooks
from tankpit_bot.diagnostics.event_stream import decode_event_lines
from tankpit_bot.runtime_records import RuntimeEventRecordDict

_LINE_SEPARATOR = b"\n"


class EventTail:
    """A resumable cursor over one events artifact.

    One instance follows one path. Construct it once per artifact and
    call :meth:`next_records` as often as wanted; each call returns
    only what arrived since the previous one.
    """

    def __init__(self, path: Path) -> None:
        """Start a cursor positioned before the file's first byte.

        Args:
            path: Events artifact to follow. It need not exist yet;
                the first :meth:`next_records` call is what touches
                the filesystem.
        """
        self._path = path
        self._identity = -1
        self._offset = 0
        self._partial = b""

    def next_records(self) -> tuple[list[RuntimeEventRecordDict], bool]:
        """Decode the complete lines appended since the last call.

        Args:
            None.

        Returns:
            ``(records, restarted)``. ``restarted`` is True when the
            artifact was replaced since the previous call -- a new run
            under the same path -- in which case ``records`` holds
            that new run's events from its first line, and any state
            the caller folded from earlier records is stale and must
            be discarded.

        Raises:
            OSError: If the artifact does not exist.
            JSONTypeError: When a complete line fails strict event
                decoding.
        """
        identity, size = _test_hooks.file_marker(self._path)
        restarted = identity != self._identity or size < self._offset
        offset = 0 if restarted else self._offset
        withheld = b"" if restarted else self._partial

        chunk = _test_hooks.read_bytes_from(self._path, offset)
        complete = (withheld + chunk).split(_LINE_SEPARATOR)
        # The final element is whatever follows the last newline: an
        # unfinished line mid-append, or empty when the file ends on a
        # newline. Either way it is not decodable yet.
        partial = complete.pop()
        records = decode_event_lines([line.decode("utf-8") for line in complete])

        # Committed only once the decode has succeeded. Advancing the
        # cursor first would consume a malformed line, so the NEXT
        # poll would sail past it and fold a run with a hole in it --
        # the artifact's own contract is that a bad line is surfaced,
        # not silently dropped.
        self._identity = identity
        self._offset = offset + len(chunk)
        self._partial = partial
        return records, restarted


__all__ = [
    "EventTail",
]
