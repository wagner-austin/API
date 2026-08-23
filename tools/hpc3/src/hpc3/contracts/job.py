"""The job contract: what a submission must state before it can be rendered.

The five rules below cost real time to learn, so they are enforced when a spec
is decoded rather than documented for a reader to remember. A spec that
violates one cannot be constructed from JSON at all, which means an invalid
job is caught at author time instead of an hour into the queue or ten hours
into a run.

1. The GPU model is named, and the cluster carries it. A bare ``--gres=gpu:1``
   on a mixed partition is a coin flip over generations; where the pinned torch
   does not target the card that comes up, the failure reads like a bug in the
   training code.
2. A billing partition requires explicit consent. A partition's name is not
   evidence about its cost -- HPC3's ``free-gpu32`` bills at ``UsageFactor``
   1.0 -- so silence must mean no.
3. A preemptible run long enough to matter carries requeue and checkpointing.
   Under ``PreemptMode=CANCEL`` an eviction destroys unsaved work.
4. The wall clock fits the partition. Slurm rejects the rest at submission.
5. The partition exists on the cluster the workspace selected.

Every rule is asked of a :class:`~hpc3.contracts.cluster.ClusterFacts` rather
than of a constant, so the same code enforces a different machine's real
limits without a branch anywhere in it.
"""

from __future__ import annotations

from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    require_bool,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from hpc3.contracts.cluster import (
    ClusterFacts,
    partition_bills,
    partition_facts,
    require_gpu_type,
    require_partition,
)
from hpc3.contracts.layout import require_project

PREEMPTION_PROTECTION_THRESHOLD_MINUTES = 60
"""Above this, a preemptible job must carry requeue and checkpointing.

Below it, re-running a lost job costs less than the checkpoint machinery, and
on a zero-usage-factor partition a re-run costs nothing at all. Above it, an
unprotected job is a bet that nothing else wants the node for hours.
"""

MINUTES_PER_HOUR = 60


class JobSpec(TypedDict):
    """One submission, fully specified.

    Attributes:
        project: Which body of work this belongs to. Prefixes the job name so
            ``squeue`` is self-describing on a shared machine, and names the
            directory the job's scripts and logs live in so two projects
            cannot scatter into each other.
        name: The job's own name within its project.
        partition: Partition to submit to. Validated against the selected
            cluster, not against a fixed list.
        gpu: GPU model to pin. Never generic -- see rule 1.
        gpu_count: GPUs requested on the node.
        cpus: CPU cores requested. Where a partition bills, this is usually
            the whole charge: billing tracks cores, not GPUs or memory.
        mem_gb: Host memory requested, in GiB.
        minutes: Wall-clock limit.
        requeue: Whether Slurm should resubmit the job after a preemption.
        checkpoint_steps: Training steps between checkpoints; 0 means none.
        accept_billing: Explicit consent to spend service units. Must be True
            for any partition whose QOS charges.
        env_path: Absolute path on the cluster to a directory with a ``bin``
            holding the payload's interpreter or binary.
        command: Payload to run, executed with that ``bin`` already on PATH.
    """

    project: str
    name: str
    partition: str
    gpu: str
    gpu_count: int
    cpus: int
    mem_gb: int
    minutes: int
    requeue: bool
    checkpoint_steps: int
    accept_billing: bool
    env_path: str
    command: str


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


def _require_positive(obj: dict[str, JSONValue], key: str) -> int:
    """Read a required integer field that must be at least one.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not an integer, or below one.
            A job asking for zero cores or zero minutes describes no work.
    """
    value = require_int(obj, key)
    if value < 1:
        raise JSONTypeError(f"Field '{key}' must be at least 1, got {value}")
    return value


def _check_partition_carries_gpu(cluster: ClusterFacts, partition: str, gpu: str) -> None:
    """Reject a job asking a partition for a GPU it does not hold.

    Args:
        cluster: The selected cluster.
        partition: Target partition.
        gpu: Requested GPU model.

    Raises:
        AppError: With ``PARTITION_GPU_MISMATCH`` if the partition's nodes do
            not carry the model. Slurm would leave the job pending forever
            rather than reject it.
    """
    if gpu not in partition_facts(cluster, partition)["gpus"]:
        raise AppError(
            Hpc3ErrorCode.PARTITION_GPU_MISMATCH,
            f"Partition {partition!r} on {cluster['slug']!r} carries no {gpu} GPUs; "
            "the job would pend forever.",
        )


def _check_billing_consent(cluster: ClusterFacts, partition: str, accept_billing: bool) -> None:
    """Reject a billing job that did not say so.

    Args:
        cluster: The selected cluster.
        partition: Target partition.
        accept_billing: Whether the caller consented to spend service units.

    Raises:
        AppError: With ``PARTITION_BILLS_WITHOUT_CONSENT`` if the partition
            charges and consent was not given. The message names the measured
            usage factor, because a partition's name is not evidence about
            its cost.
    """
    if partition_bills(cluster, partition) and not accept_billing:
        factor = partition_facts(cluster, partition)["usage_factor"]
        raise AppError(
            Hpc3ErrorCode.PARTITION_BILLS_WITHOUT_CONSENT,
            f"Partition {partition!r} on {cluster['slug']!r} charges service units "
            f"(UsageFactor {factor}). Set 'accept_billing' to true to spend them.",
        )


def _check_preemption_protection(
    cluster: ClusterFacts, partition: str, minutes: int, requeue: bool, checkpoint_steps: int
) -> None:
    """Reject a long preemptible job that would lose everything if evicted.

    Args:
        cluster: The selected cluster.
        partition: Target partition.
        minutes: Requested wall clock.
        requeue: Whether Slurm should resubmit after preemption.
        checkpoint_steps: Steps between checkpoints; 0 means none.

    Raises:
        AppError: With ``PREEMPTIBLE_RUN_UNPROTECTED`` if the job is
            preemptible, longer than
            :data:`PREEMPTION_PROTECTION_THRESHOLD_MINUTES`, and lacks either
            requeue or checkpointing. Requeue without checkpoints restarts
            from step zero, which is not protection.
    """
    if not partition_facts(cluster, partition)["preemptible"]:
        return
    if minutes <= PREEMPTION_PROTECTION_THRESHOLD_MINUTES:
        return
    if requeue and checkpoint_steps > 0:
        return
    raise AppError(
        Hpc3ErrorCode.PREEMPTIBLE_RUN_UNPROTECTED,
        f"A {minutes}-minute job on preemptible {partition!r} needs both "
        f"'requeue' and a positive 'checkpoint_steps'; got requeue={requeue}, "
        f"checkpoint_steps={checkpoint_steps}. Preemption cancels the job.",
    )


def _check_time_limit(cluster: ClusterFacts, partition: str, minutes: int) -> None:
    """Reject a job asking for more wall clock than its partition allows.

    Args:
        cluster: The selected cluster.
        partition: Target partition.
        minutes: Requested wall clock.

    Raises:
        AppError: With ``TIME_LIMIT_EXCEEDS_PARTITION`` if the request exceeds
            the partition ceiling.
    """
    limit = partition_facts(cluster, partition)["max_hours"] * MINUTES_PER_HOUR
    if minutes > limit:
        raise AppError(
            Hpc3ErrorCode.TIME_LIMIT_EXCEEDS_PARTITION,
            f"Partition {partition!r} on {cluster['slug']!r} allows {limit} minutes, "
            f"job asked for {minutes}.",
        )


def encode_job_spec(spec: JobSpec) -> dict[str, JSONValue]:
    """Encode a job spec to a JSON object.

    Args:
        spec: Spec to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "project": spec["project"],
        "name": spec["name"],
        "partition": spec["partition"],
        "gpu": spec["gpu"],
        "gpu_count": spec["gpu_count"],
        "cpus": spec["cpus"],
        "mem_gb": spec["mem_gb"],
        "minutes": spec["minutes"],
        "requeue": spec["requeue"],
        "checkpoint_steps": spec["checkpoint_steps"],
        "accept_billing": spec["accept_billing"],
        "env_path": spec["env_path"],
        "command": spec["command"],
    }


def decode_job_spec(value: JSONValue, cluster: ClusterFacts) -> JobSpec:
    """Decode and validate a JSON value into a job spec.

    Args:
        value: Value produced by the JSON loader.
        cluster: The cluster whose measured limits the rules are checked
            against.

    Returns:
        A spec that satisfies every submission rule on that cluster.

    Raises:
        JSONTypeError: If the value is not an object, or a field is missing,
            mistyped, empty, or non-positive.
        AppError: If the spec names no GPU the cluster carries, targets a
            partition it does not have, targets one that does not carry the
            model, bills without consent, leaves a long preemptible run
            unprotected, or exceeds the partition's ceiling. The code
            identifies which.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"job spec must be a JSON object, got {type(value).__name__}")

    partition = require_partition(cluster, value, "partition")
    gpu = require_gpu_type(cluster, value, "gpu")
    minutes = _require_positive(value, "minutes")
    requeue = require_bool(value, "requeue")
    accept_billing = require_bool(value, "accept_billing")

    checkpoint_steps = require_int(value, "checkpoint_steps")
    if checkpoint_steps < 0:
        raise JSONTypeError(
            f"Field 'checkpoint_steps' must not be negative, got {checkpoint_steps}"
        )

    _check_partition_carries_gpu(cluster, partition, gpu)
    _check_billing_consent(cluster, partition, accept_billing)
    _check_time_limit(cluster, partition, minutes)
    _check_preemption_protection(cluster, partition, minutes, requeue, checkpoint_steps)

    return JobSpec(
        project=require_project(value, "project"),
        name=_require_nonempty_str(value, "name"),
        partition=partition,
        gpu=gpu,
        gpu_count=_require_positive(value, "gpu_count"),
        cpus=_require_positive(value, "cpus"),
        mem_gb=_require_positive(value, "mem_gb"),
        minutes=minutes,
        requeue=requeue,
        checkpoint_steps=checkpoint_steps,
        accept_billing=accept_billing,
        env_path=_require_nonempty_str(value, "env_path"),
        command=_require_nonempty_str(value, "command"),
    )


__all__ = [
    "MINUTES_PER_HOUR",
    "PREEMPTION_PROTECTION_THRESHOLD_MINUTES",
    "JobSpec",
    "decode_job_spec",
    "encode_job_spec",
]
