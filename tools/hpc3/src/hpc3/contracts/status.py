"""The status contract: what a submitted job is doing and what it has cost.

Slurm reports a job's charge as a ``billing`` figure inside ``AllocTRES``,
which is a rate rather than a total: the cost is that figure multiplied by
elapsed time, and then by the partition's usage factor. On this cluster the
factor is the whole story -- ``free-gpu`` runs at 0.000000, so a job there
reports ``billing=4`` and still costs nothing, while the identically-shaped
job on ``free-gpu32`` costs 4 service units per hour.

Reading ``billing`` as a cost is therefore wrong in both directions: it
overstates free work and, because ``sbank`` rounds to whole units, understates
short billed work as zero. :func:`service_units` applies the factor so callers
get the number that will eventually appear on the balance.
"""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from hpc3.contracts.cluster import ClusterFacts, partition_facts, require_partition

JobState = Literal[
    "PENDING",
    "RUNNING",
    "SUSPENDED",
    "COMPLETING",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "PREEMPTED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "REQUEUED",
]
"""Job states this package recognises, as ``sacct`` spells them."""

JOB_STATES: tuple[JobState, ...] = (
    "PENDING",
    "RUNNING",
    "SUSPENDED",
    "COMPLETING",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "PREEMPTED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "REQUEUED",
)

TERMINAL_STATES: frozenset[JobState] = frozenset(
    {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "PREEMPTED", "NODE_FAIL", "OUT_OF_MEMORY"}
)
"""States from which a job will not advance on its own.

``REQUEUED`` is deliberately absent: a requeued job is going back to the
queue, which is the protection working rather than the run ending.
"""

SECONDS_PER_HOUR = 3600


class JobStatus(TypedDict):
    """One job's accounting row.

    Attributes:
        job_id: Slurm job id.
        name: Job name as submitted.
        partition: Partition the job occupies.
        state: Current state.
        elapsed_seconds: Wall clock consumed so far.
        billing_tres: The ``billing`` figure from ``AllocTRES``. A rate per
            hour, not a total, and not yet adjusted by the usage factor.
        gpu_count: GPUs the allocation holds, from ``AllocTRES``'s
            ``gres/gpu``. Zero while pending. This is what GPU-hours are
            computed from, and it is not derivable from ``billing``: the two
            are separate TRES with unrelated weights.
        node_list: Nodes assigned, or an empty string while pending.
    """

    job_id: str
    name: str
    partition: str
    state: JobState
    elapsed_seconds: int
    billing_tres: int
    gpu_count: int
    node_list: str


def gpu_hours(status: JobStatus) -> float:
    """Compute the GPU-hours a job has consumed.

    Args:
        status: The job's accounting row.

    Returns:
        GPUs held times elapsed hours. Unlike service units this is charged
        against nobody -- it is the measure of our share of a shared machine,
        which is a courtesy question rather than a billing one, and the free
        partitions make it the only measure that exists.
    """
    return status["gpu_count"] * status["elapsed_seconds"] / SECONDS_PER_HOUR


def service_units(status: JobStatus, cluster: ClusterFacts) -> float:
    """Compute what a job has actually charged.

    Args:
        status: The job's accounting row.
        cluster: The cluster whose measured usage factors apply.

    Returns:
        Service units consumed: the billing rate times elapsed hours, times
        the partition's usage factor. Zero on a zero-factor partition however
        long the job ran, which is why the raw ``billing`` figure must not be
        read as a cost -- and correct on a site that charges a fraction,
        because the measured factor is applied rather than a yes/no.

    Raises:
        AppError: With ``PARTITION_UNKNOWN`` if the row names a partition this
            cluster does not have.
    """
    factor = partition_facts(cluster, status["partition"])["usage_factor"]
    return factor * status["billing_tres"] * status["elapsed_seconds"] / SECONDS_PER_HOUR


def is_terminal(state: JobState) -> bool:
    """Report whether a job in this state has stopped for good.

    Args:
        state: State to classify.

    Returns:
        True when the job will not advance without a new submission.
        ``REQUEUED`` returns False: the job is going back to the queue.
    """
    return state in TERMINAL_STATES


def _require_object(value: JSONValue, what: str) -> dict[str, JSONValue]:
    """Narrow a decoded JSON value to an object.

    Args:
        value: Value produced by the JSON loader.
        what: Name of the thing being decoded, used in the error message.

    Returns:
        The value as a JSON object.

    Raises:
        JSONTypeError: If the value is not a JSON object.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"{what} must be a JSON object, got {type(value).__name__}")
    return value


def require_state(obj: dict[str, JSONValue], key: str) -> JobState:
    """Read and narrow a required job-state field.

    Shared with :mod:`hpc3.contracts.closure`, which records the terminal
    state a job ended in and must narrow it the same way -- a closure holding
    a state this package does not recognise would be a job it can neither
    report on nor stop reporting on.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The named state.

    Raises:
        JSONTypeError: If the field is missing, not a string, or names a
            state this package does not recognise. An unrecognised state must
            not be treated as terminal or non-terminal by guess.
    """
    raw = require_str(obj, key)
    for candidate in JOB_STATES:
        if raw == candidate:
            return candidate
    raise JSONTypeError(f"Field '{key}' must name one of {list(JOB_STATES)}, got {raw!r}")


def _require_nonnegative(obj: dict[str, JSONValue], key: str) -> int:
    """Read a required integer field that cannot be negative.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not an integer, or negative.
    """
    value = require_int(obj, key)
    if value < 0:
        raise JSONTypeError(f"Field '{key}' must not be negative, got {value}")
    return value


def encode_job_status(status: JobStatus) -> dict[str, JSONValue]:
    """Encode a job status to a JSON object.

    Args:
        status: Status to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "job_id": status["job_id"],
        "name": status["name"],
        "partition": status["partition"],
        "state": status["state"],
        "elapsed_seconds": status["elapsed_seconds"],
        "billing_tres": status["billing_tres"],
        "gpu_count": status["gpu_count"],
        "node_list": status["node_list"],
    }


def decode_job_status(value: JSONValue, cluster: ClusterFacts) -> JobStatus:
    """Decode and validate a JSON value into a job status.

    Args:
        value: Value produced by the JSON loader.
        cluster: The cluster whose partitions the row is checked against.

    Returns:
        Validated status.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, the id or name is empty, the state is unrecognised, or
            a count is negative.
        AppError: With ``PARTITION_UNKNOWN`` if the row names a partition this
            cluster does not have.
    """
    obj = _require_object(value, "job status")
    job_id = require_str(obj, "job_id")
    if job_id == "":
        raise JSONTypeError("Field 'job_id' must not be empty")
    name = require_str(obj, "name")
    if name == "":
        raise JSONTypeError("Field 'name' must not be empty")
    return JobStatus(
        job_id=job_id,
        name=name,
        partition=require_partition(cluster, obj, "partition"),
        state=require_state(obj, "state"),
        elapsed_seconds=_require_nonnegative(obj, "elapsed_seconds"),
        billing_tres=_require_nonnegative(obj, "billing_tres"),
        gpu_count=_require_nonnegative(obj, "gpu_count"),
        node_list=require_str(obj, "node_list"),
    )


__all__ = [
    "JOB_STATES",
    "SECONDS_PER_HOUR",
    "TERMINAL_STATES",
    "JobState",
    "JobStatus",
    "decode_job_status",
    "encode_job_status",
    "gpu_hours",
    "is_terminal",
    "require_state",
    "service_units",
]
