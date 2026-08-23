"""Remembering that a job finished, so accounting need never be asked again.

``unaccounted`` is the finding that catches a job which was submitted, got an
id, and does not exist -- the one condition no cluster-side query can detect,
because the evidence is the absence of a cluster-side record.

It has a shelf life. ``sacct`` retention is finite: a cluster keeps completed
jobs for a while and then forgets them. Once that window passes, a job that ran
perfectly a month ago is a ledger entry with no accounting row, which is
character-for-character the same observation as a job that never existed. The
tool would report it forever, the finding count would climb without bound, and
``hpc3-triage`` would exit non-zero permanently -- which is the same as having
no triage at all, because nobody reads a board that is always red.

So the moment accounting reports a job in a terminal state, that fact is
written down locally. It is the one observation that cannot be recovered later,
and it is cheap to keep.

A closure is not a duplicate of the ledger entry. The ledger says what was
submitted; a closure says how it ended. They are separate files for the same
reason both are append-only: neither is ever rewritten, so a crash truncates at
a line boundary and loses at most the record being written.

What this deliberately does NOT do is close a job the tool never saw finish. A
job that vanished before any triage run stays unaccounted forever, and that is
correct -- it is exactly the case the finding exists for.
"""

from __future__ import annotations

from platform_core.json_utils import JSONTypeError, JSONValue, require_str
from typing_extensions import TypedDict

from hpc3.contracts.status import JobState, require_state


class Closure(TypedDict):
    """One job observed to have ended, and how.

    Attributes:
        job_id: The job that ended.
        state: The terminal state accounting reported. Kept rather than
            reduced to a boolean because ``COMPLETED`` and ``OUT_OF_MEMORY``
            are both closures and only one of them is good news.
        closed_at: ISO-8601 timestamp of the observation, supplied by the
            caller. Not when the job ended -- when this tool noticed, which is
            the only thing it can honestly claim.
    """

    job_id: str
    state: JobState
    closed_at: str


def encode_closure(closure: Closure) -> dict[str, JSONValue]:
    """Encode a closure to a JSON object.

    Args:
        closure: Closure to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "job_id": closure["job_id"],
        "state": closure["state"],
        "closed_at": closure["closed_at"],
    }


def decode_closure(value: JSONValue) -> Closure:
    """Decode and validate a JSON value into a closure.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        Validated closure.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            empty, or the state is not one this package recognises. A closure
            that cannot be read is a job that will be reported as unaccounted
            forever, so it fails the read rather than being skipped.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"closure must be a JSON object, got {type(value).__name__}")

    job_id = require_str(value, "job_id")
    if job_id == "":
        raise JSONTypeError("Field 'job_id' must not be empty")
    closed_at = require_str(value, "closed_at")
    if closed_at == "":
        raise JSONTypeError("Field 'closed_at' must not be empty")

    return Closure(job_id=job_id, state=require_state(value, "state"), closed_at=closed_at)


__all__ = ["Closure", "decode_closure", "encode_closure"]
