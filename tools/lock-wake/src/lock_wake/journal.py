"""Reading the fleet-lock journal from a byte offset, strictly.

The journal (``.fleet-events.jsonl`` in the MCPs repo root) is written by
``scripts/with-fleet-lock.ps1``: one JSON object per transition, appended
with ``FileShare ReadWrite`` so readers never block a cascade. Its own
docstring states the subscription contract this module implements: "a
subscriber keeps a byte offset, reads from it, and cannot miss a transition
at any polling interval, because nothing is ever overwritten."

TORN TAILS ARE EXPECTED, NOT ERRORS. The writer may be mid-append when this
process reads, so the bytes after the last newline are a line that does not
exist yet. The reader consumes exactly through the last complete line and
reports the offset just past it; the torn tail is read whole on the next
cycle. That is cursor discipline, not tolerance -- a COMPLETE line that
fails to decode is fatal and names itself, because a skipped transition is
a cascade nobody was told about, which is the blindness the journal exists
to remove.

ONE FIELD IS ABSENT FROM HISTORY. ``agent`` joined the journal on
2026-09-09 (MCPs ``66b85d32``) so a publisher can @mention whoever started
a cascade. Rows written before that commit are immutable facts without the
field, and this decoder reads them as ``agent=""`` -- the same explicit
"nobody recorded one" the hpc3 ledger uses for its pre-field rows. New rows
always carry the key, empty when the invoking shell exported no label.
"""

from __future__ import annotations

import pathlib
from typing import Final, Literal

from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    load_json_str,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from lock_wake import _test_hooks

#: Every transition kind the lock wrapper writes, in lifecycle order.
#: Pinned as data so a new kind in the journal is a loud decode refusal
#: here rather than a silently mis-summarised cascade.
EVENT_KINDS: Final = ("requested", "waiting", "acquired", "step", "released", "failed", "timeout")


class LockEvent(TypedDict):
    """One journal transition, as the lock wrapper wrote it.

    Attributes:
        ts: UTC timestamp, ISO-8601 with a trailing ``Z``.
        kind: The transition, one of :data:`EVENT_KINDS`.
        holder_pid: The lock-taking process, the journal's ``pid`` field --
            renamed here because ``pid`` shadows a builtin-adjacent name in
            half the linters that read it.
        label: The make target's label (``up-transcriber``, ``deploy-ts``).
        op: The wrapper operation (``service-up``, ``bases``, ...).
        only: The language scope the target declared.
        detail: Kind-specific text -- the step command, the failure note,
            the holder stamp a waiter saw.
        agent: The session label behind the invocation, ``""`` when none
            was exported or the row predates the field.
    """

    ts: str
    kind: Literal["requested", "waiting", "acquired", "step", "released", "failed", "timeout"]
    holder_pid: int
    label: str
    op: str
    only: str
    detail: str
    agent: str


class JournalSlice(TypedDict):
    """What one read of the journal yielded.

    Attributes:
        events: Every complete line's event, in file order.
        next_offset: The byte offset just past the last complete line --
            what the position file records once the events are announced.
    """

    events: tuple[LockEvent, ...]
    next_offset: int


def _decode_kind(
    value: str, line_number: int
) -> Literal["requested", "waiting", "acquired", "step", "released", "failed", "timeout"]:
    """Narrow a kind string to the declared set.

    Args:
        value: The journal row's ``kind`` field.
        line_number: 1-based line number, for the refusal.

    Returns:
        The narrowed kind.

    Raises:
        JSONTypeError: For a kind this package does not know -- summarising
            an unknown transition as if it were understood would report a
            cascade story that never happened.
    """
    if value == "requested":
        return "requested"
    if value == "waiting":
        return "waiting"
    if value == "acquired":
        return "acquired"
    if value == "step":
        return "step"
    if value == "released":
        return "released"
    if value == "failed":
        return "failed"
    if value == "timeout":
        return "timeout"
    raise JSONTypeError(
        f"journal line {line_number} has kind {value!r}, which this package does not "
        f"know; the declared kinds are {', '.join(EVENT_KINDS)} and an unknown one "
        f"means the lock wrapper grew a transition lock-wake would mis-summarise"
    )


def decode_lock_event(value: JSONValue, line_number: int) -> LockEvent:
    """Decode and validate one journal line's object.

    Args:
        value: Value produced by the JSON loader.
        line_number: 1-based line number within the slice, for refusals.

    Returns:
        The validated event.

    Raises:
        JSONTypeError: If the value is not an object, a required field is
            missing or mistyped, or the kind is undeclared.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(
            f"journal line {line_number} is a {type(value).__name__}, not an object; "
            f"a line that cannot be read is a cascade transition nobody is told about"
        )
    agent_raw = value.get("agent")
    if agent_raw is None:
        agent = ""
    elif isinstance(agent_raw, str):
        agent = agent_raw
    else:
        raise JSONTypeError(
            f"journal line {line_number} field 'agent' is a "
            f"{type(agent_raw).__name__}, not a string"
        )
    return LockEvent(
        ts=require_str(value, "ts"),
        kind=_decode_kind(require_str(value, "kind"), line_number),
        holder_pid=require_int(value, "pid"),
        label=require_str(value, "label"),
        op=require_str(value, "op"),
        only=require_str(value, "only"),
        detail=require_str(value, "detail"),
        agent=agent,
    )


def read_journal_slice(journal: pathlib.Path, offset: int) -> JournalSlice:
    """Read every complete journal line at or past a byte offset.

    Args:
        journal: The journal's path.
        offset: Byte offset of the first unread byte, from the position
            file; 0 for a bridge that has never run.

    Returns:
        The decoded events and the offset just past the last complete
        line. An absent journal reads as empty at offset 0 rather than
        raising: a machine whose fleet has never taken a lock has no
        journal, and refusing the first cycle for that would make the
        bridge impossible to start.

    Raises:
        JSONTypeError: A complete line that is not a valid event.
        InvalidJsonError: A complete line that is not JSON at all.
        ValueError: An offset past the end of the journal -- the journal
            was truncated or replaced, and silently rewinding would
            re-announce history; the operator decides, not this reader.
        OSError: A journal that exists but cannot be read.
    """
    if not _test_hooks.file_exists(journal):
        if offset != 0:
            raise ValueError(
                f"position says byte {offset} of {journal}, but the journal is absent; "
                f"it was deleted or moved, and rewinding silently would re-announce "
                f"every cascade in history"
            )
        return JournalSlice(events=(), next_offset=0)
    data = _test_hooks.read_bytes(journal)
    if offset > len(data):
        raise ValueError(
            f"position says byte {offset} of {journal}, but the journal holds only "
            f"{len(data)} bytes; it was truncated or replaced, and rewinding silently "
            f"would re-announce every cascade in history"
        )
    window = data[offset:]
    last_newline = window.rfind(b"\n")
    if last_newline == -1:
        return JournalSlice(events=(), next_offset=offset)
    complete = window[: last_newline + 1]
    events: list[LockEvent] = []
    for line_number, raw in enumerate(complete.decode("utf-8").splitlines(), start=1):
        if raw.strip() == "":
            continue
        events.append(decode_lock_event(load_json_str(raw), line_number))
    return JournalSlice(events=tuple(events), next_offset=offset + last_newline + 1)


__all__ = [
    "EVENT_KINDS",
    "JournalSlice",
    "LockEvent",
    "decode_lock_event",
    "read_journal_slice",
]
