"""Reading the fleet-lock journal from a byte offset, strictly.

The journal (``.fleet-events.jsonl`` in the MCPs repo root) is written by
``scripts/with-fleet-lock.ps1``: one JSON object per transition, appended
with ``FileShare ReadWrite`` so readers never block a cascade. Its own
docstring states the subscription contract this module implements: "a
subscriber keeps a byte offset, reads from it, and cannot miss a transition
at any polling interval, because nothing is ever overwritten."

THE CURSOR IS SHARED; THE DECODE IS THIS PACKAGE'S. Reading complete lines
from a byte offset, torn tails left for the next cycle, is
:func:`platform_core.journal_cursor.read_complete_lines`, lifted out of
this module (MCPs board task ebc80a03). What stays here is the fleet
journal's own line shape. A COMPLETE line that fails to decode is fatal and
names itself, because a skipped transition is a cascade nobody was told
about, which is the blindness the journal exists to remove.

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

from platform_core.journal_cursor import read_complete_lines
from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    load_json_str,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from lock_wake import _test_hooks

#: Every transition kind the fleet's writers emit, in lifecycle order.
#: Pinned as data so a new kind in the journal is a loud decode refusal
#: here rather than a silently mis-summarised cascade.
#:
#: ``gate-blocked`` and ``refused`` happen BEFORE any hold exists:
#: the freshness gate declining a rebuild-triggering target, and the
#: lock wrapper declining an unlabelled acquire (MCPs board task
#: 07fcc6af -- until 2026-09-11 both refusals left no record at all,
#: so "how many sessions tried and were told no" was unanswerable
#: from the fleet record).
#:
#: ``checked`` is not a lock transition at all: it is one finished
#: ``make test`` run under the per-package check lock (MCPs
#: ``packages/maketools`` ``check_lock``), written to this journal so check
#: results ride the same stream as rebuilds (MCPs board task ea2ea29c, the
#: operator's one channel "for make chdck for rwbuilds"). Its ``label`` is
#: the package, its ``op`` is ``check-lock`` and its ``detail`` carries the
#: exit code, the seconds, the HEAD and the tree.
EVENT_KINDS: Final = (
    "gate-blocked",
    "refused",
    "requested",
    "waiting",
    "acquired",
    "step",
    "released",
    "failed",
    "timeout",
    "checked",
)


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
    kind: Literal[
        "gate-blocked",
        "refused",
        "requested",
        "waiting",
        "acquired",
        "step",
        "released",
        "failed",
        "timeout",
        "checked",
    ]
    holder_pid: int
    label: str
    op: str
    only: str
    detail: str
    agent: str


#: Each declared kind's text to its narrowed value: the one table
#: :func:`_decode_kind` reads, so a kind is narrowed by lookup rather than by
#: an arm per kind, and a kind missing here is refused like an unknown one.
_KIND_BY_NAME: Final[
    dict[
        str,
        Literal[
            "gate-blocked",
            "refused",
            "requested",
            "waiting",
            "acquired",
            "step",
            "released",
            "failed",
            "timeout",
            "checked",
        ],
    ]
] = {
    "gate-blocked": "gate-blocked",
    "refused": "refused",
    "requested": "requested",
    "waiting": "waiting",
    "acquired": "acquired",
    "step": "step",
    "released": "released",
    "failed": "failed",
    "timeout": "timeout",
    "checked": "checked",
}


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
) -> Literal[
    "gate-blocked",
    "refused",
    "requested",
    "waiting",
    "acquired",
    "step",
    "released",
    "failed",
    "timeout",
    "checked",
]:
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
    kind = _KIND_BY_NAME.get(value)
    if kind is not None:
        return kind
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
    """Read and decode every complete journal line at or past a byte offset.

    Args:
        journal: The journal's path.
        offset: Byte offset of the first unread byte, from the position
            file; 0 for a bridge that has never run.

    Returns:
        The decoded events and the offset just past the last complete line,
        per :func:`platform_core.journal_cursor.read_complete_lines`: an
        absent journal (a machine whose fleet has never taken a lock) reads
        as empty at offset 0.

    Raises:
        JSONTypeError: A complete line that is not a valid event.
        InvalidJsonError: A complete line that is not JSON at all.
        ValueError: A position into an absent journal or past its end --
            the journal was deleted, truncated or replaced, and the
            operator decides, not this reader.
        OSError: A journal that exists but cannot be read.
    """
    lines = read_complete_lines(_test_hooks.file_exists, _test_hooks.read_bytes, journal, offset)
    return JournalSlice(
        events=tuple(
            decode_lock_event(load_json_str(line["text"]), line["number"])
            for line in lines["lines"]
        ),
        next_offset=lines["next_offset"],
    )


def require_check_rows(events: tuple[LockEvent, ...], journal: pathlib.Path) -> None:
    """Refuse a check journal slice holding anything but finished runs.

    The check lock writes its rows to their own file (MCPs
    ``packages/maketools`` ``check_lock``, ``.check-events.jsonl``) because
    every checkout's older reader of the fleet journal refuses a kind it does
    not declare. A lock transition in that file is a writer pointed at the
    wrong journal, and announcing it as a check run would report something
    that never happened.

    Args:
        events: The check journal slice's events.
        journal: The check journal's path, for the refusal.

    Raises:
        JSONTypeError: When any event's kind is not ``checked``.
    """
    strays = sorted({event["kind"] for event in events if event["kind"] != "checked"})
    if strays:
        raise JSONTypeError(
            f"{journal} holds {', '.join(strays)} rows; the check journal carries only "
            f"checked rows, and a lock transition here means a writer chose the wrong file"
        )


__all__ = [
    "EVENT_KINDS",
    "JournalSlice",
    "LockEvent",
    "decode_lock_event",
    "read_journal_slice",
    "require_check_rows",
]
