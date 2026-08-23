"""What a Slurm cluster is, as a shape the rules can be checked against.

This module holds no facts about any particular machine. It defines what a
cluster has to tell this package before the package will submit to it, and the
one function that reads a partition's facts. The measured facts themselves live
in :mod:`hpc3.clusters`, one module per machine.

The split exists because the facts are the enforcement. Whether a partition
bills, whether it preempts, how many GPUs one user may hold -- those are what
turn a rule from advice into a refusal, and they are different on every
cluster. Keeping them as data in this package (rather than as fields in the
user's workspace document) is what stops a caller declaring
``max_gpus_per_user: 999`` and disabling the check instead of raising the
ceiling. Adding a cluster means measuring one and committing a module; it is
never something a run can assert about itself.

Partition and GPU names are plain strings here rather than ``Literal`` types,
because the valid set is a property of the selected cluster and not of this
package. What that costs is small and worth naming: a first-party call that
hard-codes a misspelled partition is now caught when the value is decoded
rather than by mypy. What it buys is that the refusal message names *your*
cluster's partitions instead of some other machine's.
"""

from __future__ import annotations

from collections.abc import Mapping

from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue, require_int, require_str
from typing_extensions import TypedDict


class GpuRequest(TypedDict):
    """A job's demand for GPUs, when it has one.

    Kept as one nullable object rather than a model field beside a count,
    because the two must agree and a flat pair permits states that mean
    nothing: a named model with a count of zero, or a count of three with no
    model. Neither is expressible here -- a job either asks for GPUs, naming
    which and how many, or it asks for none by writing null.

    Attributes:
        model: GPU model to pin, by Slurm GRES name. Never generic: a bare
            request lands wherever the scheduler chooses, which is how a run
            reaches a card the pinned torch cannot drive.
        count: GPUs requested on the node. At least one -- a request for zero
            is spelled by omitting the request.
    """

    model: str
    count: int


class PartitionFacts(TypedDict):
    """What one partition costs, allows, and risks.

    Attributes:
        usage_factor: The QOS ``UsageFactor``. Service units are the billing
            rate times elapsed hours times this. Stored as the measured number
            rather than a bills/does-not-bill flag because Slurm permits any
            non-negative value, and a site that charges half rate is neither
            free nor full price.
        preemptible: Whether a job here can be cancelled to make room for a
            higher-tier job. Where the cluster runs ``PreemptMode=CANCEL``,
            a preemption destroys unsaved work outright.
        max_hours: Wall-clock ceiling the partition enforces.
        gpus: GPU models physically present, by Slurm GRES name. Must be a
            subset of the cluster's own ``gpus``. **Empty means this is a CPU
            partition**, and a job asking it for a GPU is refused.
        max_gpus_per_user: QOS ``MaxTRESPU`` for ``gres/gpu``, or None where
            the QOS declares no such cap. A sweep that asks for more does not
            have the excess rejected -- those jobs sit pending against a limit
            rather than a resource, which reads as contention and is not.
        max_cpus_per_user: QOS ``MaxTRESPU`` for ``cpu``, or None where the
            QOS declares no such cap. The twin of the field above for CPU
            work: a wide CPU sweep pends against cores, and nothing else here
            would predict it.
        max_jobs_per_user: QOS ``MaxJobsPU``: concurrently running jobs.

    None on either ceiling means **measured absence, not unknown**. HPC3's
    ``free-gpu-part`` caps ``gres/gpu`` and says nothing about cores;
    ``free-part`` caps ``cpu`` and says nothing about GPUs. Writing a large
    number instead of None would be inventing a limit the QOS does not
    declare, and the check it fed would be fiction.
    """

    usage_factor: float
    preemptible: bool
    max_hours: int
    gpus: tuple[str, ...]
    max_gpus_per_user: int | None
    max_cpus_per_user: int | None
    max_jobs_per_user: int


class ClusterFacts(TypedDict):
    """One measured machine.

    Attributes:
        slug: Short name a workspace selects this cluster by.
        description: One line naming the machine and when it was measured, so
            a stale module is visible without a git blame.
        gpus: Every GPU model the cluster carries, by Slurm GRES name.
        partitions: Facts per partition, keyed by partition name.
    """

    slug: str
    description: str
    gpus: tuple[str, ...]
    partitions: Mapping[str, PartitionFacts]


def partition_facts(cluster: ClusterFacts, partition: str) -> PartitionFacts:
    """Read one partition's measured facts.

    This is the only accessor. The six one-line readers it replaced were
    wrappers over a dictionary lookup, and callers read the fields directly.

    Args:
        cluster: The selected cluster.
        partition: Partition name.

    Returns:
        That partition's facts.

    Raises:
        AppError: With ``PARTITION_UNKNOWN`` if the cluster has no such
            partition. The message lists the ones it does have, because the
            usual cause is a workspace written for a different machine.
    """
    facts = cluster["partitions"].get(partition)
    if facts is None:
        known = sorted(cluster["partitions"])
        raise AppError(
            Hpc3ErrorCode.PARTITION_UNKNOWN,
            f"Cluster {cluster['slug']!r} has no partition {partition!r}; it has {known}.",
        )
    return facts


def partition_bills(cluster: ClusterFacts, partition: str) -> bool:
    """Report whether a partition charges anything at all.

    Args:
        cluster: The selected cluster.
        partition: Partition name.

    Returns:
        True when the usage factor is above zero. Kept as a named question
        rather than inlined at each call site because "does this spend money"
        is the thing consent is asked about, and a bare ``> 0.0`` comparison
        scattered across the package would be the same rule written five
        times.

    Raises:
        AppError: With ``PARTITION_UNKNOWN`` if the cluster has no such
            partition.
    """
    return partition_facts(cluster, partition)["usage_factor"] > 0.0


def partition_names(cluster: ClusterFacts) -> tuple[str, ...]:
    """List a cluster's partitions in a stable order.

    Args:
        cluster: The selected cluster.

    Returns:
        Partition names, sorted, for error messages and validation.
    """
    return tuple(sorted(cluster["partitions"]))


def require_partition(cluster: ClusterFacts, obj: dict[str, JSONValue], key: str) -> str:
    """Read a required field naming a partition of this cluster.

    Args:
        cluster: The selected cluster.
        obj: Object being decoded.
        key: Field name.

    Returns:
        The partition name.

    Raises:
        JSONTypeError: If the field is missing or not a string.
        AppError: With ``PARTITION_UNKNOWN`` if the cluster has no such
            partition.
    """
    name = require_str(obj, key)
    partition_facts(cluster, name)
    return name


def require_gpu_type(cluster: ClusterFacts, obj: dict[str, JSONValue], key: str) -> str:
    """Read a required field naming a GPU model this cluster carries.

    Args:
        cluster: The selected cluster.
        obj: Object being decoded.
        key: Field name.

    Returns:
        The GPU model.

    Raises:
        JSONTypeError: If the field is missing or not a string.
        AppError: With ``GPU_TYPE_UNPINNED`` if the value names no GPU this
            cluster carries. A generic ``gpu`` lands wherever the scheduler
            chooses, which is how a run reaches a card the pinned torch cannot
            drive.
    """
    raw = require_str(obj, key)
    if raw not in cluster["gpus"]:
        raise AppError(
            Hpc3ErrorCode.GPU_TYPE_UNPINNED,
            f"Field {key!r} must name a GPU cluster {cluster['slug']!r} carries "
            f"({list(cluster['gpus'])}), got {raw!r}. "
            "A generic GPU request can land on a card the pinned torch cannot drive.",
        )
    return raw


def decode_gpu_request(cluster: ClusterFacts, value: JSONValue, key: str) -> GpuRequest | None:
    """Decode a job's GPU request, which may be absent.

    Args:
        cluster: The selected cluster.
        value: The field's value. ``None`` means the job wants no GPU.
        key: Field name, used in error messages.

    Returns:
        The validated request, or None for a CPU-only job.

    Raises:
        JSONTypeError: If the value is neither null nor an object, or the
            count is missing, mistyped, or below one. There is no zero-GPU
            request: a job wanting no GPU writes null, and ``{"count": 0}``
            would be a second spelling of the same state.
        AppError: With ``GPU_TYPE_UNPINNED`` if the model names no GPU the
            cluster carries.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise JSONTypeError(
            f"Field {key!r} must be a GPU request object or null, got {type(value).__name__}"
        )
    count = require_int(value, "count")
    if count < 1:
        raise JSONTypeError(
            f"Field {key!r} must ask for at least 1 GPU, got {count}. "
            "A job wanting no GPU states null."
        )
    return GpuRequest(model=require_gpu_type(cluster, value, "model"), count=count)


def encode_gpu_request(request: GpuRequest | None) -> JSONValue:
    """Encode a GPU request back to JSON.

    Args:
        request: The request, or None for a CPU-only job.

    Returns:
        An object carrying the model and count, or null.
    """
    if request is None:
        return None
    return {"model": request["model"], "count": request["count"]}


def describe_gpu_request(request: GpuRequest | None) -> str:
    """Render a GPU request for a human reading a queue row or a console line.

    One definition rather than three, because this string appears in the job
    comment, in the submit confirmation and in the sweep summary, and an
    operator comparing what they were told against what ``sacct`` shows should
    not have to notice that two of them spell it differently.

    Args:
        request: The request, or None for a CPU-only job.

    Returns:
        ``"<model>x<count>"``, or ``"cpu-only"`` when no GPU was asked for --
        never an empty string, because a blank in this position reads as
        missing information rather than as a deliberate absence.
    """
    if request is None:
        return "cpu-only"
    return f"{request['model']}x{request['count']}"


def gpu_count(request: GpuRequest | None) -> int:
    """Report how many GPUs a request asks for, counting absence as zero.

    Every ceiling and projection in this package multiplies by this number,
    and a CPU job contributes nothing to any of them. Narrowing the optional
    once here keeps that from becoming the same None-check written at six
    call sites.

    Args:
        request: The request, or None for a CPU-only job.

    Returns:
        The requested count, or 0 when no GPU was asked for.
    """
    return 0 if request is None else request["count"]


__all__ = [
    "ClusterFacts",
    "GpuRequest",
    "PartitionFacts",
    "decode_gpu_request",
    "describe_gpu_request",
    "encode_gpu_request",
    "gpu_count",
    "partition_bills",
    "partition_facts",
    "partition_names",
    "require_gpu_type",
    "require_partition",
]
