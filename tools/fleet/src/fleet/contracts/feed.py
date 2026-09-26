"""The event stream a session subscribes to instead of polling.

WHY A FILE AND NOT A SOCKET. Any number of Claude sessions must be able to
watch, they start and stop without telling each other, and none of them can be
the server. An append-only JSONL file that every session tails satisfies all
three with nothing running: a subscriber is ``Monitor({command: "fleet-watch
--follow"})`` and an unsubscribe is closing it. A socket would need a broker
that outlives every session, which is one more thing to wedge.

THE FILTER RULE, AND IT IS THE WHOLE REASON THE KINDS ARE AN ENUM. The Monitor
tool's own guidance is that silence is not success: a watcher that greps only
for the success marker stays quiet through a crash, a hang, and an unexpected
exit, and quiet looks exactly like still-running. So every terminal state is a
kind here, failures included, and :data:`TERMINAL_KINDS` names them together
so a subscriber cannot enumerate the happy ones and think it is done. This is
not hypothetical on this box -- on 2026-09-04 two suites sat wedged for
twenty-nine minutes holding 77.9 GB, emitting nothing, and looking from
outside exactly like work in progress.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_int,
    require_str,
)
from platform_core.members import find_member
from typing_extensions import TypedDict


class FeedKind(StrEnum):
    """What a feed line can say; each value is the word a record spells it with.

    ``lost`` is the one that does not correspond to anything the run does. It
    is emitted by a watcher that finds a dispatch whose lease has expired with
    no terminal event, which is the observable signature of a wedge -- the run
    cannot report its own death, so something else has to.

    ``refused`` is terminal too, and deliberately on the same feed as the
    rest: a dispatch that never started because the node had no room is an
    outcome a subscriber is waiting for, and putting refusals on a separate
    channel is how a caller ends up waiting forever for a run that was never
    going to exist.

    The enum is the one place a kind is added: its members are both the
    membership test a decode applies and the narrowed value it returns, so a
    kind that is not a member does not exist in either sense.
    """

    REFUSED = "refused"
    LEASED = "leased"
    STAGED = "staged"
    STARTED = "started"
    PHASE = "phase"
    HEARTBEAT = "heartbeat"
    PASSED = "passed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LOST = "lost"


TERMINAL_KINDS: Final[frozenset[FeedKind]] = frozenset(
    {FeedKind.REFUSED, FeedKind.PASSED, FeedKind.FAILED, FeedKind.CANCELLED, FeedKind.LOST}
)
"""Kinds after which no further event will arrive for a run.

Spelled out rather than derived from :class:`FeedKind`, because terminality
is not a property of being a kind -- it is a claim about each one, and the
five here are a deliberate subset of the ten. A derivation would have to
encode the same judgement somewhere else.

A frozenset because the only question anyone asks of it is membership, and a
named constant rather than a literal at each call site so a new terminal kind
cannot be added without every reader learning about it.
"""


class FeedEvent(TypedDict):
    """One line of the feed.

    Attributes:
        at_unix: When it happened, whole seconds since the epoch.
        run_id: The dispatch it belongs to. Present on every event including
            ``refused``, so a subscriber filtering for its own run does not
            miss the reason its run never began.
        node: The node's SSH alias.
        project: Repo-relative project path.
        kind: What happened.
        detail: Human-readable specifics -- the phase name, the exit code,
            the refusal's reason. Free text on purpose: this is the half a
            person reads, and the half a program reads is ``kind``. A
            program that parses ``detail`` is a program that will break, and
            keeping the two separate is what lets the wording improve without
            breaking a subscriber.
    """

    at_unix: int
    run_id: str
    node: str
    project: str
    kind: FeedKind
    detail: str


def is_terminal(event: FeedEvent) -> bool:
    """Whether this event ends its run's story.

    Args:
        event: The event to judge.

    Returns:
        True when no further event will arrive for that run.
    """
    return event["kind"] in TERMINAL_KINDS


def render_feed_line(event: FeedEvent) -> str:
    """Render an event as the one line a Monitor subscriber is notified with.

    Args:
        event: The event to render.

    Returns:
        A single line, kind first. Kind leads because a subscriber's grep is
        written against it and a leading timestamp would make every
        alternation start with a wildcard.
    """
    return (
        f"{event['kind'].upper()} {event['project']} on {event['node']} "
        f"[{event['run_id']}] {event['detail']}"
    )


def encode_feed_event(event: FeedEvent) -> JSONObject:
    """Encode one feed event.

    Args:
        event: The event to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "at_unix": event["at_unix"],
        "run_id": event["run_id"],
        "node": event["node"],
        "project": event["project"],
        "kind": event["kind"].value,
        "detail": event["detail"],
    }


def decode_feed_event(value: JSONValue) -> FeedEvent:
    """Decode and validate one feed event.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated event.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, or the kind is not one this version knows. An unknown
            kind is refused rather than passed through: a subscriber deciding
            whether a run has ended reads ``kind``, and a kind it cannot
            classify would be treated as non-terminal, which is the failure
            where a caller waits forever on a run that already finished.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"feed event must be a JSON object, got {type(value).__name__}")
    spelling = require_str(value, "kind")
    kind = find_member(spelling, FeedKind)
    if kind is None:
        raise JSONTypeError(
            f"feed event kind {spelling!r} is not one of "
            f"{', '.join(sorted(FeedKind))}; an unrecognised kind would be read as "
            "non-terminal and leave a subscriber waiting on a run that has already ended"
        )
    return FeedEvent(
        at_unix=require_int(value, "at_unix"),
        run_id=require_str(value, "run_id"),
        node=require_str(value, "node"),
        project=require_str(value, "project"),
        kind=kind,
        detail=require_str(value, "detail"),
    )


__all__ = [
    "TERMINAL_KINDS",
    "FeedEvent",
    "FeedKind",
    "decode_feed_event",
    "encode_feed_event",
    "is_terminal",
    "render_feed_line",
]
