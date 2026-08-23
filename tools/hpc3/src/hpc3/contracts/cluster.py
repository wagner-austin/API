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
from platform_core.json_utils import JSONValue, require_str
from typing_extensions import TypedDict


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
            subset of the cluster's own ``gpus``.
        max_gpus_per_user: QOS ``MaxTRESPU`` for ``gres/gpu``. A sweep that
            asks for more does not have the excess rejected -- those jobs sit
            pending against a limit rather than a resource, which reads as
            contention and is not.
        max_jobs_per_user: QOS ``MaxJobsPU``: concurrently running jobs.
    """

    usage_factor: float
    preemptible: bool
    max_hours: int
    gpus: tuple[str, ...]
    max_gpus_per_user: int
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


__all__ = [
    "ClusterFacts",
    "PartitionFacts",
    "partition_bills",
    "partition_facts",
    "partition_names",
    "require_gpu_type",
    "require_partition",
]
