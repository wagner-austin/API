"""Reading and appending the two append-only files.

The ledger and the feed share every mechanical property -- one JSON object per
line, appended and never rewritten, read whole -- and differ only in what a
line means and who reads it. So the mechanics live here once and the two
contracts stay separate, which is the DRY split: shared plumbing, distinct
vocabulary.

A MALFORMED LINE IS FATAL, NOT SKIPPED. It would be easy to ignore a line that
does not decode and carry on with the rest; that is the "best effort" this
codebase does not do, and here it has a specific cost. The ledger's live rows
are what a capacity check subtracts from a node's free memory, so a line
skipped is a running dispatch made invisible, and the next dispatch is admitted
onto a node that is already full -- the exact failure the package exists to
prevent. The reader raises and names the line number.
"""

from __future__ import annotations

import pathlib

from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str

from fleet.contracts.feed import FeedEvent, decode_feed_event, encode_feed_event
from fleet.contracts.ledger import LedgerEntry, decode_ledger_entry, encode_ledger_entry, is_live
from fleet.core import _test_hooks


def _read_lines(path: pathlib.Path) -> tuple[tuple[int, JSONValue], ...]:
    """Read a JSONL file into numbered, parsed lines.

    Args:
        path: The file to read.

    Returns:
        ``(line_number, value)`` for every non-blank line, one-indexed. An
        absent file reads as empty rather than raising: no dispatch has run in
        this workspace yet, and refusing the first one for having no history
        would make the package impossible to start using.

    Raises:
        JSONTypeError: If a line is not valid JSON, naming the line number.
            Blank lines are skipped because a trailing newline is normal;
            nothing else is.
    """
    if not _test_hooks.file_exists(path):
        return ()
    numbered: list[tuple[int, JSONValue]] = []
    for index, line in enumerate(_test_hooks.read_text(path).splitlines(), start=1):
        if not line.strip():
            continue
        numbered.append((index, load_json_str(line)))
    return tuple(numbered)


def read_ledger(path: pathlib.Path) -> tuple[LedgerEntry, ...]:
    """Read every dispatch this workspace has recorded.

    Args:
        path: The ledger file.

    Returns:
        Every row, in the order it was appended.

    Raises:
        AppError: With ``LEDGER_ROW_UNPARSABLE`` if a line does not decode,
            naming the line. See the module docstring for why this is fatal.
    """
    rows: list[LedgerEntry] = []
    for line_number, value in _read_lines(path):
        if not isinstance(value, dict):
            raise AppError(
                FleetErrorCode.LEDGER_ROW_UNPARSABLE,
                f"{path} line {line_number} is a {type(value).__name__}, not an object; a "
                "line that cannot be read is a running dispatch made invisible, and the next "
                "capacity check would admit work onto a node that is already full",
            )
        rows.append(decode_ledger_entry(value))
    return tuple(rows)


def latest_rows(path: pathlib.Path) -> tuple[LedgerEntry, ...]:
    """Read the CURRENT row for each dispatch the ledger knows about.

    THE LEDGER IS APPEND-ONLY, SO A FINISHED RUN STILL HAS A RUNNING ROW IN
    IT. Anything asking what is happening now must therefore reduce to the
    last row per id, and the first version of :func:`live_runs` did not:
    it counted every row whose outcome was ``running``, including the ones
    already superseded. Measured 2026-09-04 on the second real dispatch --
    sedona declares ``max_concurrent_runs: 1``, and having run and CANCELLED
    exactly one dispatch it refused every future one as already full. A
    node's live count could only ever go up.

    Args:
        path: The ledger file.

    Returns:
        One row per dispatch, each the most recently appended, in the order
        the dispatches first appear.

    Raises:
        AppError: With ``LEDGER_ROW_UNPARSABLE`` if a line does not decode.
    """
    latest: dict[str, LedgerEntry] = {}
    for row in read_ledger(path):
        latest[row["run_id"]] = row
    return tuple(latest.values())


def live_runs(path: pathlib.Path, *, node: str) -> int:
    """Count the dispatches still holding resources on one node.

    Args:
        path: The ledger file.
        node: The node's workspace name.

    Returns:
        How many dispatches on that node are CURRENTLY running -- see
        :func:`latest_rows` for why the distinction is load-bearing.
    """
    return sum(1 for row in latest_rows(path) if row["node"] == node and is_live(row))


def append_ledger(path: pathlib.Path, entry: LedgerEntry) -> None:
    """Append one dispatch row.

    Args:
        path: The ledger file.
        entry: The row to record.
    """
    _test_hooks.append_text(path, dump_json_str(encode_ledger_entry(entry)))


def read_feed(path: pathlib.Path) -> tuple[FeedEvent, ...]:
    """Read every event this workspace has emitted.

    Args:
        path: The feed file.

    Returns:
        Every event, in the order it was appended.

    Raises:
        AppError: With ``FEED_EVENT_UNPARSABLE`` if a line is not an object.
        JSONTypeError: If a line is an object that does not decode.
    """
    events: list[FeedEvent] = []
    for line_number, value in _read_lines(path):
        if not isinstance(value, dict):
            raise AppError(
                FleetErrorCode.FEED_EVENT_UNPARSABLE,
                f"{path} line {line_number} is a {type(value).__name__}, not an object",
            )
        events.append(decode_feed_event(value))
    return tuple(events)


def append_feed(path: pathlib.Path, event: FeedEvent) -> None:
    """Append one event to the stream subscribers tail.

    Args:
        path: The feed file.
        event: The event to emit.
    """
    _test_hooks.append_text(path, dump_json_str(encode_feed_event(event)))


__all__ = [
    "append_feed",
    "append_ledger",
    "latest_rows",
    "live_runs",
    "read_feed",
    "read_ledger",
]
