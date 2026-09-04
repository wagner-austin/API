"""The durable record of every dispatch this machine has made.

One JSON object per line, appended and never rewritten. The feed is the
subscribable stream and this is the record: a subscriber tails the feed and
forgets, and an auditor reads this and does not.

WHO DISPATCHED IS A FIELD, and it is the field this record exists for. The
incident behind the package was two sessions colliding in one project with no
way for either to know the other was there. A record that says a suite ran but
not which session ran it answers the easy half of the question. The identity
is the board's -- agent label plus session UUID -- so a ledger row and a board
post can be matched by a reader who has both.

WHY THE OUTCOME IS A SEPARATE FIELD FROM THE EXIT CODE. A dispatch that was
refused before it started has no exit code, and a dispatch whose lease expired
with no result has none either. Spelling those as exit code -1 would make them
arithmetic on a number that means something else, and every reader would need
to know the convention.
"""

from __future__ import annotations

from typing import Final, Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

LedgerOutcome = Literal["refused", "passed", "failed", "cancelled", "lost", "running"]
"""How a dispatch ended, or that it has not.

``running`` is the only non-terminal value and exists because the row is
written when the dispatch STARTS. A ledger that only recorded finished work
could not answer "what is live on this node right now", which is exactly what
a capacity check needs -- and a capacity check that could not see running work
would admit a second dispatch onto a node the first had already filled.
"""

OUTCOME_BY_NAME: Final[dict[str, LedgerOutcome]] = {
    "refused": "refused",
    "passed": "passed",
    "failed": "failed",
    "cancelled": "cancelled",
    "lost": "lost",
    "running": "running",
}
"""Every outcome, keyed by the string a row spells it with.

Same shape as :data:`~fleet.contracts.feed.KIND_BY_NAME` and for the same
reason: the key set IS the membership test and the value IS the narrowed
literal, so an outcome absent here does not exist in either sense. Typed
``dict[str, LedgerOutcome]``, so mypy checks each value against the Literal
once, at definition, which is what makes the narrowing sound without a cast.
"""

#: The one outcome that means a dispatch still holds resources.
RUNNING: Final[str] = "running"

#: Recorded in ``exit_code`` when a dispatch produced no exit status at all.
#:
#: Distinct from any real status. A refused dispatch never ran, and a lost one
#: never reported; both are outcomes rather than results, and the outcome
#: field is where a reader learns which.
NO_EXIT_CODE: Final[int] = -1


class LedgerEntry(TypedDict):
    """One dispatch, as it will be read months later.

    Attributes:
        run_id: The dispatch's identity, unique across the ledger.
        node: The node's workspace name.
        host: The SSH alias actually used. Both, because a workspace may
            rename a node and a row has to stay readable against the ssh
            config of the day.
        project: Repo-relative project path.
        agent: Board label of the dispatching session.
        session_id: That session's UUID.
        started_unix: When the dispatch began, whole seconds since the epoch.
        ended_unix: When it reached a terminal outcome, or the same value as
            ``started_unix`` while it is still running. Never null: a
            nullable timestamp makes every reader branch, and the outcome
            field already says whether the run has ended.
        outcome: How it ended, or ``running``.
        exit_code: The recipe's exit status, or :data:`NO_EXIT_CODE` when
            there was none.
        workers: How many test workers the dispatch was granted. Recorded
            because it is the number the capacity arithmetic produced, and a
            wedge is diagnosed by comparing it against what the node could
            actually hold.
        detail: Human-readable specifics -- a refusal's reason, a failure's
            summary. Free text; a program reads ``outcome``.
    """

    run_id: str
    node: str
    host: str
    project: str
    agent: str
    session_id: str
    started_unix: int
    ended_unix: int
    outcome: LedgerOutcome
    exit_code: int
    workers: int
    detail: str


def is_live(entry: LedgerEntry) -> bool:
    """Whether this dispatch still holds resources on its node.

    Args:
        entry: The row to judge.

    Returns:
        True while the outcome is ``running``.
    """
    return entry["outcome"] == RUNNING


def encode_ledger_entry(entry: LedgerEntry) -> JSONObject:
    """Encode one dispatch row.

    Args:
        entry: The row to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "run_id": entry["run_id"],
        "node": entry["node"],
        "host": entry["host"],
        "project": entry["project"],
        "agent": entry["agent"],
        "session_id": entry["session_id"],
        "started_unix": entry["started_unix"],
        "ended_unix": entry["ended_unix"],
        "outcome": entry["outcome"],
        "exit_code": entry["exit_code"],
        "workers": entry["workers"],
        "detail": entry["detail"],
    }


def decode_ledger_entry(value: JSONValue) -> LedgerEntry:
    """Decode and validate one dispatch row.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated row.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, the outcome is unknown, the row ends before it starts,
            or it was granted no workers. An unknown outcome is refused
            because a capacity check asks whether a row is ``running`` and
            would read anything it cannot classify as finished -- admitting a
            dispatch onto a node that is already full, which is the failure
            the whole package exists to prevent.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"ledger entry must be a JSON object, got {type(value).__name__}")
    spelling = require_str(value, "outcome")
    outcome = OUTCOME_BY_NAME.get(spelling)
    if outcome is None:
        raise JSONTypeError(
            f"ledger outcome {spelling!r} is not one of "
            f"{', '.join(sorted(OUTCOME_BY_NAME))}; an unrecognised outcome would be read as "
            "finished and let a second dispatch onto a node the first still holds"
        )
    started_unix = require_int(value, "started_unix")
    ended_unix = require_int(value, "ended_unix")
    if ended_unix < started_unix:
        raise JSONTypeError(
            f"ledger row ends at {ended_unix} and starts at {started_unix}; a dispatch cannot "
            "finish before it begins, and a duration derived from this row would be negative"
        )
    workers = require_int(value, "workers")
    if workers < 0:
        raise JSONTypeError(
            f"workers must not be negative, got {workers}; a refused dispatch is recorded with "
            "zero, which is the smallest truthful number"
        )
    return LedgerEntry(
        run_id=require_str(value, "run_id"),
        node=require_str(value, "node"),
        host=require_str(value, "host"),
        project=require_str(value, "project"),
        agent=require_str(value, "agent"),
        session_id=require_str(value, "session_id"),
        started_unix=started_unix,
        ended_unix=ended_unix,
        outcome=outcome,
        exit_code=require_int(value, "exit_code"),
        workers=workers,
        detail=require_str(value, "detail"),
    )


__all__ = [
    "NO_EXIT_CODE",
    "OUTCOME_BY_NAME",
    "RUNNING",
    "LedgerEntry",
    "LedgerOutcome",
    "decode_ledger_entry",
    "encode_ledger_entry",
    "is_live",
]
