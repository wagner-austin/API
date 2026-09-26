"""Reading the fleet health journal from a byte offset, strictly.

The journal (MCPs ``fleet-mcp/state/health-events.jsonl``) is appended by
the fleet audit, one JSON object per run whose outcome is news, encoded by
``encodeHealthEvent`` in ``fleet-mcp/src/health-journal.ts``: ``at``,
``kind``, ``key`` and ``body``, the body the note exactly as it is to be
posted. That file is the writer's contract and this decoder mirrors it
field for field; a kind added there is a loud refusal here until this
package learns it, never a line posted as if it were understood.

The cursor (complete lines from an offset, torn tails left for the next
cycle) is :func:`platform_core.journal_cursor.read_complete_lines`.
"""

from __future__ import annotations

import pathlib
from typing import Final, Literal

from platform_core.journal_cursor import read_complete_lines
from platform_core.json_utils import JSONTypeError, JSONValue, load_json_str, require_str
from typing_extensions import TypedDict

from fleet_health_wake import _test_hooks

#: The kinds the audit writes, as ``HEALTH_EVENT_KINDS`` in fleet-mcp.
EVENT_KINDS: Final = ("transitions", "baseline", "refused")


class HealthEvent(TypedDict):
    """One journal line, as the fleet audit wrote it.

    Attributes:
        at: When the run that wrote it happened, ISO-8601 UTC.
        kind: ``transitions`` (rows or nodes changed), ``baseline`` (the
            previous snapshot could not be read, so nothing was compared)
            or ``refused`` (the audit's build is stale and it did not run).
        key: What makes the line unique to its writer.
        body: The note, rendered by the audit.
    """

    at: str
    kind: Literal["transitions", "baseline", "refused"]
    key: str
    body: str


#: Each declared kind's text to its narrowed value, so a kind is narrowed by
#: lookup and one missing here is refused like an unknown one.
_KIND_BY_NAME: Final[dict[str, Literal["transitions", "baseline", "refused"]]] = {
    "transitions": "transitions",
    "baseline": "baseline",
    "refused": "refused",
}


class HealthSlice(TypedDict):
    """What one read of the journal yielded.

    Attributes:
        events: Every complete line's event, in file order.
        next_offset: The byte offset just past the last complete line.
    """

    events: tuple[HealthEvent, ...]
    next_offset: int


def decode_health_event(value: JSONValue, line_number: int) -> HealthEvent:
    """Decode and validate one journal line's object.

    Args:
        value: Value produced by the JSON loader.
        line_number: 1-based line number within the slice, for refusals.

    Returns:
        The validated event.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, or the kind is not one the audit is known to write.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(
            f"health journal line {line_number} is a {type(value).__name__}, not an "
            f"object; a line that cannot be read is a fleet change nobody is told about"
        )
    kind_text = require_str(value, "kind")
    kind = _KIND_BY_NAME.get(kind_text)
    if kind is None:
        raise JSONTypeError(
            f"health journal line {line_number} has kind {kind_text!r}, which this "
            f"package does not know; the declared kinds are {', '.join(EVENT_KINDS)}, "
            f"and an unknown one means fleet-mcp's health journal grew a kind this "
            f"bridge would post without understanding"
        )
    return HealthEvent(
        at=require_str(value, "at"),
        kind=kind,
        key=require_str(value, "key"),
        body=require_str(value, "body"),
    )


def read_health_slice(journal: pathlib.Path, offset: int) -> HealthSlice:
    """Read and decode every complete journal line at or past a byte offset.

    Args:
        journal: The journal's path.
        offset: Byte offset of the first unread byte, from the position
            file; 0 for a bridge that has never run.

    Returns:
        The decoded events and the offset just past the last complete
        line. An absent journal (an audit that has never had news) reads as
        empty at offset 0.

    Raises:
        JSONTypeError: A complete line that is not a valid event.
        InvalidJsonError: A complete line that is not JSON at all.
        ValueError: A position into an absent journal or past its end.
        OSError: A journal that exists but cannot be read.
    """
    lines = read_complete_lines(_test_hooks.file_exists, _test_hooks.read_bytes, journal, offset)
    return HealthSlice(
        events=tuple(
            decode_health_event(load_json_str(line["text"]), line["number"])
            for line in lines["lines"]
        ),
        next_offset=lines["next_offset"],
    )


__all__ = [
    "EVENT_KINDS",
    "HealthEvent",
    "HealthSlice",
    "decode_health_event",
    "read_health_slice",
]
