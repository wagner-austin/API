"""What the cluster is holding under this account, whoever put it there.

Every other query in this package starts from something we already know: a
job id we recorded, a run document we resolved. This one starts from the
account, and asks the cluster to enumerate itself. That is the only question
whose answer can contain a job the ledger has never heard of.

WHY IT EXISTS. The ledger's claim is "every job this machine submitted", and
:func:`~hpc3.core.triage.unaccounted_jobs` checks one direction of it -- we
recorded it, accounting never heard of it. Nothing checked the other: a job
running under the account that no ledger row claims. That is exactly the
trace a raw ``ssh <host> sbatch`` leaves, which is how the image builds were
started for twenty-one versions and how any future bypass would look. A
record that only proves its own entries are real, and cannot say whether it
holds all of them, is half a record.

``squeue --me`` rather than ``squeue -u <name>``: the workspace declares an
SSH destination and no username, and ``--me`` resolves to whoever the key
authenticated as. A username in the workspace would be a second place for
the account's identity to live and a second place for it to be wrong.

WHAT THIS CANNOT SEE, stated because the gap is real: a bypassed job that
already finished. ``squeue`` holds a job for minutes after it ends and then
forgets it, so this catches an unrecorded job while it is running -- which is
when it is costing something and when cancelling it is still possible -- and
never afterwards. Catching the finished ones would mean an ``sacct`` sweep
over a time window, which reports every interactive shell the account has
ever opened and would drown the signal it was added for.
"""

from __future__ import annotations

from platform_core.json_utils import JSONTypeError, JSONValue, require_str
from typing_extensions import TypedDict


class AccountJob(TypedDict):
    """One job the cluster currently holds for this account.

    Deliberately three fields and not the accounting row's nine. This is the
    output of an enumeration, not of a lookup: it answers "what is there",
    and every question about a job it names can be asked of that job by id
    afterwards. Carrying the allocation here would mean a second parser to
    keep in step with ``sacct``'s for no question this query is asked.

    Attributes:
        job_id: Slurm job id.
        name: Job name, verbatim. Not required to carry a project prefix --
            a job that bypassed this package is under no obligation to be
            named the way this package names things, and demanding the
            prefix is what would make the bypass invisible.
        state: Slurm state, verbatim from ``squeue``'s ``%T``. Reported so a
            finding can say whether the unrecorded job is running now or
            still queued, which is the difference between something to stop
            and something to cancel before it starts.
    """

    job_id: str
    name: str
    state: str


def _require_nonempty_str(obj: dict[str, JSONValue], key: str) -> str:
    """Read a required string field that must not be empty.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not a string, or empty.
    """
    value = require_str(obj, key)
    if value == "":
        raise JSONTypeError(f"Field '{key}' must not be empty")
    return value


def encode_account_job(job: AccountJob) -> dict[str, JSONValue]:
    """Encode an account job to a JSON object.

    Args:
        job: Job to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {"job_id": job["job_id"], "name": job["name"], "state": job["state"]}


def decode_account_job(value: JSONValue) -> AccountJob:
    """Decode and validate a JSON value into an account job.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        Validated account job.

    Raises:
        JSONTypeError: If the value is not an object, or any field is
            missing, mistyped or empty. Every field is required non-empty
            here, unlike a pending reason: this row exists because the
            cluster volunteered it, so a blank id or state means the parse
            is wrong rather than that the scheduler has not looked yet.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"account job must be a JSON object, got {type(value).__name__}")
    return AccountJob(
        job_id=_require_nonempty_str(value, "job_id"),
        name=_require_nonempty_str(value, "name"),
        state=_require_nonempty_str(value, "state"),
    )


__all__ = ["AccountJob", "decode_account_job", "encode_account_job"]
