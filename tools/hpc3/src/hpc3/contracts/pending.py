"""Why a job is pending, and whether waiting will ever help.

Measured on HPC3 on 2026-08-22: of 621 pending GPU jobs, **261 were
``DependencyNeverSatisfied``** and only 3 were waiting on ``Resources``. Those
261 are somebody's dead workflow. They will never run. They will sit in the
queue until a human notices, and meanwhile every count of "how busy is the
cluster" includes them.

That is the failure this module exists to prevent on our side. A job left
pending looks identical to a job that is merely waiting -- ``squeue`` shows
``PENDING`` for both -- and the difference is entirely in the reason column.
So reasons are classified rather than displayed:

* **Transient** -- the queue will resolve this. Waiting is correct.
* **Blocked** -- nothing will resolve this without intervention. Waiting is
  the mistake, and the job should be cancelled and resubmitted or fixed.

An unrecognised reason is treated as blocked, not transient. A new reason we
have never seen is exactly the case where assuming "it'll sort itself out" is
how a job sits for a week.
"""

from __future__ import annotations

from platform_core.json_utils import JSONTypeError, JSONValue, require_str
from typing_extensions import TypedDict

TRANSIENT_REASONS: frozenset[str] = frozenset(
    {
        # The scheduler is doing its job; the job is simply in line.
        "Resources",
        "Priority",
        "None",
        "",
        # Our own limits, which our own finishing jobs release.
        "QOSMaxJobsPerUserLimit",
        "QOSMaxGRESPerUser",
        "QOSGrpCpuLimit",
        "QOSGrpGRES",
        "AssocMaxJobsLimit",
        "JobArrayTaskLimit",
        "MaxGRESPerAccount",
        "AssocGrpBillingMinutes",
        # A dependency that can still be satisfied, unlike the one below.
        "Dependency",
        # Transient cluster states that clear on their own.
        "ReqNodeNotAvail",
        "Reservation",
        "BeginTime",
    }
)
"""Reasons a job will leave on its own. Waiting is the correct response."""


class PendingJob(TypedDict):
    """One pending job and the reason the scheduler gives for it.

    Attributes:
        job_id: Slurm job id.
        name: Job name as submitted.
        reason: The scheduler's reason, verbatim from ``squeue``'s ``%r``.
    """

    job_id: str
    name: str
    reason: str


def is_blocked(reason: str) -> bool:
    """Report whether a pending reason will resolve without intervention.

    Args:
        reason: The scheduler's reason, verbatim.

    Returns:
        True when nothing will clear this on its own. Unrecognised reasons
        return True deliberately: a reason we have never seen is precisely
        where assuming patience would be rewarded is how a job sits for a
        week. A false alarm costs a glance; a missed block costs the run.
    """
    return reason.strip() not in TRANSIENT_REASONS


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


def encode_pending_job(job: PendingJob) -> dict[str, JSONValue]:
    """Encode a pending job to a JSON object.

    Args:
        job: Job to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {"job_id": job["job_id"], "name": job["name"], "reason": job["reason"]}


def decode_pending_job(value: JSONValue) -> PendingJob:
    """Decode and validate a JSON value into a pending job.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        Validated pending job.

    Raises:
        JSONTypeError: If the value is not an object, or the id or name is
            missing, mistyped or empty. The reason may be empty: ``squeue``
            reports ``None`` or blank for a job the scheduler has just not
            looked at yet, and that is a real transient state.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"pending job must be a JSON object, got {type(value).__name__}")
    return PendingJob(
        job_id=_require_nonempty_str(value, "job_id"),
        name=_require_nonempty_str(value, "name"),
        reason=require_str(value, "reason"),
    )


__all__ = [
    "TRANSIENT_REASONS",
    "PendingJob",
    "decode_pending_job",
    "encode_pending_job",
    "is_blocked",
]
