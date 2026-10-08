"""The record of every retire a node runner could not finish when it settled a run.

One JSON object per line beside the ledger, appended and never rewritten,
like the ledger and the feed (:mod:`fleet.core.records`). A settle closes the
queue job and the ledger row before it retires the run's directory
(:mod:`fleet.cli.node_settle`), so a retire the node did not answer has no
live row left to be settled again: this record is what a later pass reads to
retire it (:mod:`fleet.core.retire_owed`, MCPs board task 8776b828).

A run's CURRENT state is its last line. ``owed`` is the only state a pass
acts on; ``retired`` and ``failed`` end it, so no directory is retired twice
and a retire the node refused is not sent again.
"""

from __future__ import annotations

from enum import StrEnum

from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue, require_int, require_str
from platform_core.members import require_member
from typing_extensions import TypedDict


class RetireState(StrEnum):
    """Where one run's retire stands; each value is a line's word.

    ``owed`` is written when the node did not answer the retire, ``retired``
    when a later pass's retire was answered, and ``failed`` when the node
    answered and the retire failed there, which is raised by name and never
    retried, as a settle's failed retire never was.
    """

    OWED = "owed"
    RETIRED = "retired"
    FAILED = "failed"


class RetireRecord(TypedDict):
    """One change in a run's retire.

    Attributes:
        run_id: The dispatch whose directory it is.
        node: The node's workspace name, so a runner retries only its own.
        state: What happened.
        at_unix: When, whole seconds since the epoch.
        detail: Why: the transport's message for ``owed`` and ``failed``,
            where the transcript is kept for ``retired``.
    """

    run_id: str
    node: str
    state: RetireState
    at_unix: int
    detail: str


def encode_retire_record(record: RetireRecord) -> JSONObject:
    """Encode one retire line.

    Args:
        record: The line to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "run_id": record["run_id"],
        "node": record["node"],
        "state": record["state"].value,
        "at_unix": record["at_unix"],
        "detail": record["detail"],
    }


def decode_retire_record(value: JSONValue) -> RetireRecord:
    """Decode and validate one retire line.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated line.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, or the state is not a :class:`RetireState` word. An
            unknown state is refused rather than read as done, since a run
            read as retired is a directory nothing removes again.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"retire record must be a JSON object, got {type(value).__name__}")
    return RetireRecord(
        run_id=require_str(value, "run_id"),
        node=require_str(value, "node"),
        state=require_member(value, "state", RetireState),
        at_unix=require_int(value, "at_unix"),
        detail=require_str(value, "detail"),
    )


__all__ = [
    "RetireRecord",
    "RetireState",
    "decode_retire_record",
    "encode_retire_record",
]
