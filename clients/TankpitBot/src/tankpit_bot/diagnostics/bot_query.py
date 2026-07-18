"""Named queries over ``runs/bot/latest.events.jsonl``.

Four canned queries, each printed to stdout in a stable format that
``awk``/``jq`` pipelines can post-process:

  - ``timeline``         -- one line per STATE / WIRE /
    DIAGNOSTIC event, in file order. The smallest useful "what
    happened?" view.
  - ``stalls``           -- ``action_outcome`` events with
    ``signal=stall_timeout``, including the surrounding ``tick_n`` and
    ``bot_state`` (Tier 3.2 fields).
  - ``action-spans``     -- pairs WIRE dispatch events with their
    matching ``action_outcome`` events; one line per action lifecycle with
    duration_ms and signal.
  - ``target-decisions`` -- HUNT score events with the target tile and
    score so reviewers can scan acquisition history without grepping.

The CLI itself is a thin dispatcher; each query is a pure function that
takes the records list and writes to a stream. File reads go through
:mod:`tankpit_bot._test_hooks` so tests inject fakes.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from platform_core.json_utils import (
    JSONObject,
    load_json_str,
    narrow_json_to_dict,
    optional_int,
    optional_str,
    require_str,
)

from tankpit_bot import _test_hooks

#: Default JSONL path the queries read when none is passed in.
DEFAULT_EVENTS_PATH: Path = Path("runs/bot/latest.events.jsonl")

#: Channels surfaced by the ``timeline`` query.
_TIMELINE_CHANNELS: frozenset[str] = frozenset({"STATE", "WIRE", "DIAGNOSTIC"})

# Reserved record-level keys stripped from a record's ``fields`` view.
_RESERVED_RECORD_KEYS: frozenset[str] = frozenset(
    {"timestamp", "level", "logger", "mode", "channel", "message"}
)


class BotQueryRecord:
    """One parsed event record from ``latest.events.jsonl``.

    Attributes:
        timestamp: ISO timestamp string written by the runtime logger.
        channel: Event channel (``AI`` / ``WIRE`` /
            ``STATE`` / ``DIAGNOSTIC`` / ``SYNC`` / ``WORLD``).
        message: Human-readable message body.
        fields: Structured-field view with reserved record-level keys
            stripped (e.g. ``tick_n``, ``bot_state``, ``signal``,
            ``action_kind``, ``combat_target_x`` ...).
    """

    __slots__ = ("channel", "fields", "message", "timestamp")

    def __init__(
        self,
        *,
        timestamp: str,
        channel: str,
        message: str,
        fields: JSONObject,
    ) -> None:
        """Initialise from validated record fields.

        Args:
            timestamp: ISO timestamp string.
            channel: Event channel string.
            message: Event message string.
            fields: Structured fields (reserved keys already stripped).
        """
        self.timestamp = timestamp
        self.channel = channel
        self.message = message
        self.fields = fields


def _decode_record(parsed: JSONObject) -> BotQueryRecord:
    """Decode a parsed JSONObject into a :class:`BotQueryRecord`.

    Args:
        parsed: Parsed JSON object for the record.

    Returns:
        Validated record.

    Raises:
        JSONTypeError: If a required field is missing or has the
            wrong type. Propagated unchanged so the CLI surfaces it
            with full context.
    """
    return BotQueryRecord(
        timestamp=require_str(parsed, "timestamp"),
        channel=require_str(parsed, "channel"),
        message=require_str(parsed, "message"),
        fields={k: v for k, v in parsed.items() if k not in _RESERVED_RECORD_KEYS},
    )


class _MissingEventsFileError(SystemExit):
    """Raised by :func:`load_records` when the JSONL file is missing.

    Mirrors the smoke script's behaviour: exits the CLI with status 1
    and carries a diagnostic message the wrapper main() can surface.
    """

    def __init__(self, path: Path) -> None:
        """Initialise with a diagnostic CLI message.

        Args:
            path: Path the loader expected to find.
        """
        super().__init__(1)
        self.path = path
        self.message = (
            f"bot-query: {path} does not exist -- did you run `poetry run tankpit-bot` first?"
        )


def load_records(path: Path) -> list[BotQueryRecord]:
    """Read every JSONL line into a list of :class:`BotQueryRecord`.

    Args:
        path: Path to the events JSONL file.

    Returns:
        Parsed records in file order.

    Raises:
        _MissingEventsFileError: If the file does not exist.
        JSONTypeError: If a record's top-level value is not an object
            or a required field is missing.
    """
    if not _test_hooks.path_exists(path):
        raise _MissingEventsFileError(path)
    text = _test_hooks.read_text(path)
    records: list[BotQueryRecord] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parsed = narrow_json_to_dict(load_json_str(line))
        records.append(_decode_record(parsed))
    return records


# Stream signature used by each query helper. ``Callable[[str], int]``
# matches both ``sys.stdout.write`` and the test recorder's write.
StreamWrite = Callable[[str], int]


def query_timeline(records: list[BotQueryRecord], write: StreamWrite) -> None:
    """Print one line per STATE / WIRE / DIAGNOSTIC event.

    Each line carries the timestamp, channel, the active ``tick_n`` (or
    ``-`` when none was set), and the message body. Reviewers can pipe
    through ``less`` or ``grep`` for narrow exploration.

    Args:
        records: All loaded events.
        write: Stream writer.
    """
    for rec in records:
        if rec.channel not in _TIMELINE_CHANNELS:
            continue
        tick = optional_int(rec.fields, "tick_n")
        tick_str = str(tick) if tick is not None else "-"
        write(f"{rec.timestamp}\ttick={tick_str}\t{rec.channel}\t{rec.message}\n")


def _is_action_outcome(rec: BotQueryRecord) -> bool:
    """Report whether a record is an ``action_outcome`` diagnostic.

    Args:
        rec: Loaded event record.

    Returns:
        True for DIAGNOSTIC records carrying the unified outcome kind.
    """
    return (
        rec.channel == "DIAGNOSTIC"
        and optional_str(rec.fields, "diagnostic_kind") == "action_outcome"
    )


def query_stalls(records: list[BotQueryRecord], write: StreamWrite) -> None:
    """Print every ``action_outcome`` event with ``outcome=stall_timeout``.

    Each line carries the timestamp, the action_kind that stalled, its
    duration_ms, the tick_n and bot_state for context, and the message
    body. Empty when no stalls occurred (the desired steady state).

    Args:
        records: All loaded events.
        write: Stream writer.
    """
    for rec in records:
        if not _is_action_outcome(rec):
            continue
        if optional_str(rec.fields, "outcome") != "stall_timeout":
            continue
        tick = optional_int(rec.fields, "tick_n")
        tick_str = str(tick) if tick is not None else "-"
        action_kind = optional_str(rec.fields, "action_kind") or "-"
        duration_ms = optional_int(rec.fields, "duration_ms")
        duration_str = str(duration_ms) if duration_ms is not None else "-"
        bot_state = optional_str(rec.fields, "bot_state") or "-"
        write(
            f"{rec.timestamp}\ttick={tick_str}\taction={action_kind}\t"
            f"duration_ms={duration_str}\tstate={bot_state}\t{rec.message}\n"
        )


def query_action_spans(records: list[BotQueryRecord], write: StreamWrite) -> None:
    """Pair WIRE dispatch events with their ``action_outcome`` resolutions.

    Args:
        records: All loaded events.
        write: Stream writer.

    Notes:
        - A WIRE event with an ``action_kind`` field opens a span.
        - The next ``action_outcome`` event with the same
          ``action_kind`` closes it. If a WIRE event opens a span and
          a second WIRE event of the same kind starts before the first
          resolves, the new dispatch overrides the open span -- the
          bot only ever has one in-flight action of a given kind at a
          time.
        - An ``action_outcome`` without a matching WIRE is printed as
          a ``(orphan)`` line so reviewers see it instead of a silent
          drop. Executor discards resolve pre-dispatch, so they are
          expected orphans.
    """
    open_spans: dict[str, BotQueryRecord] = {}
    for rec in records:
        if rec.channel == "WIRE":
            action_kind = optional_str(rec.fields, "action_kind")
            if action_kind is not None:
                open_spans[action_kind] = rec
            continue
        if not _is_action_outcome(rec):
            continue
        action_kind = optional_str(rec.fields, "action_kind")
        if action_kind is None:
            continue
        outcome = optional_str(rec.fields, "outcome") or "-"
        duration_ms = optional_int(rec.fields, "duration_ms")
        duration_str = str(duration_ms) if duration_ms is not None else "-"
        opener = open_spans.pop(action_kind, None)
        if opener is None:
            write(
                f"{rec.timestamp}\t(orphan)\taction={action_kind}\t"
                f"outcome={outcome}\tduration_ms={duration_str}\n"
            )
            continue
        write(
            f"{opener.timestamp}\t->\t{rec.timestamp}\taction={action_kind}\t"
            f"outcome={outcome}\tduration_ms={duration_str}\n"
        )


def query_target_decisions(records: list[BotQueryRecord], write: StreamWrite) -> None:
    """Print every HUNT score event with the target tile and score.

    Args:
        records: All loaded events.
        write: Stream writer.

    Lines look like::

        2026-06-20T15:02:13  tick=7  target=(131,124)  HUNT score=0.8 target=(131,124)

    Records missing both ``combat_target_x`` and ``combat_target_y``
    print ``target=-`` so a "decided IDLE" tick is still visible.
    """
    for rec in records:
        if rec.channel != "AI":
            continue
        if not rec.message.startswith("HUNT score="):
            continue
        tick = optional_int(rec.fields, "tick_n")
        tick_str = str(tick) if tick is not None else "-"
        target_x = optional_int(rec.fields, "combat_target_x")
        target_y = optional_int(rec.fields, "combat_target_y")
        if target_x is None and target_y is None:
            target_str = "-"
        else:
            tx = str(target_x) if target_x is not None else "?"
            ty = str(target_y) if target_y is not None else "?"
            target_str = f"({tx},{ty})"
        write(f"{rec.timestamp}\ttick={tick_str}\ttarget={target_str}\t{rec.message}\n")


_USAGE_BLOCK = (
    "usage: bot-query <timeline | stalls | action-spans | target-decisions> [PATH]\n"
    "  timeline           Print every STATE / WIRE / DIAGNOSTIC line.\n"
    "  stalls             Print every stall_timeout action_outcome.\n"
    "  action-spans       Pair WIRE dispatches with action_outcome.\n"
    "  target-decisions   Print every HUNT score event with target tile.\n"
    "  PATH               Optional events JSONL path (defaults to runs/bot/latest.events.jsonl).\n"
)


_QUERIES: dict[str, Callable[[list[BotQueryRecord], StreamWrite], None]] = {
    "timeline": query_timeline,
    "stalls": query_stalls,
    "action-spans": query_action_spans,
    "target-decisions": query_target_decisions,
}


def run(argv: list[str]) -> int:
    """Dispatch to a query based on ``argv``.

    Args:
        argv: Argument vector excluding the program name (i.e.
            ``sys.argv[1:]``).

    Returns:
        ``0`` on success, ``1`` on usage error or missing input.
    """
    if len(argv) == 0 or len(argv) > 2:
        sys.stderr.write(_USAGE_BLOCK)
        return 1
    query_name = argv[0]
    if query_name not in _QUERIES:
        sys.stderr.write(_USAGE_BLOCK)
        return 1
    path = Path(argv[1]) if len(argv) == 2 else DEFAULT_EVENTS_PATH
    records = load_records(path)
    writer: StreamWrite = sys.stdout.write
    _QUERIES[query_name](records, writer)
    return 0


def main() -> None:
    """Entry point for the ``tankpit-bot-query`` console script."""
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
