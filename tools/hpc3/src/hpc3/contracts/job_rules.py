"""The submission rules a job spec is checked against, cluster facts in hand.

Split from :mod:`hpc3.contracts.job` at the 600-line ceiling: that module
says what a job IS (the spec's shape, decode and encode); this one says what
a cluster will TOLERATE -- partition/GPU fit, funding, wall-clock ceilings
and preemption protection. Each rule refuses with the code that names it,
and :func:`~hpc3.contracts.job.decode_job_spec` runs them all, so nothing
importable here changes what is enforced -- only where it is read.
"""

from __future__ import annotations

from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.cluster import (
    ClusterFacts,
    GpuRequest,
    partition_bills,
    partition_facts,
    partition_names,
)

PREEMPTION_PROTECTION_THRESHOLD_MINUTES = 60
"""Above this, a preemptible job must carry requeue and checkpointing.

Below it, re-running a lost job costs less than the checkpoint machinery, and
on a zero-usage-factor partition a re-run costs nothing at all. Above it, an
unprotected job is a bet that nothing else wants the node for hours.
"""

MINUTES_PER_HOUR = 60


def _check_partition_carries_gpu(
    cluster: ClusterFacts, partition: str, gpu: GpuRequest | None
) -> None:
    """Reject a job whose GPU request does not match its partition.

    Both directions are refused, and the second is the reason this is not
    simply a membership test. Asking a CPU partition for a GPU leaves the job
    pending forever. Asking a GPU partition for no GPU is *accepted* by Slurm
    and runs -- occupying a GPU node to do CPU work, which is why it has to be
    caught here rather than left to the scheduler.

    Args:
        cluster: The selected cluster.
        partition: Target partition.
        gpu: The job's GPU request, or None for a CPU-only job.

    Raises:
        AppError: With ``PARTITION_GPU_MISMATCH`` when the partition carries
            no GPUs but one was asked for, when it carries GPUs but none was
            asked for, or when it does not hold the model requested.
    """
    available = partition_facts(cluster, partition)["gpus"]

    if gpu is None:
        if available != ():
            raise AppError(
                Hpc3ErrorCode.PARTITION_GPU_MISMATCH,
                f"Partition {partition!r} on {cluster['slug']!r} is a GPU partition "
                f"({list(available)}) and this job asks for no GPU. It would run, "
                "holding a GPU node to do CPU work. Use a CPU partition.",
            )
        return

    if available == ():
        raise AppError(
            Hpc3ErrorCode.PARTITION_GPU_MISMATCH,
            f"Partition {partition!r} on {cluster['slug']!r} is a CPU partition and "
            f"carries no GPUs, but this job asks for {gpu['count']}x {gpu['model']}; "
            "the job would pend forever.",
        )

    if gpu["model"] not in available:
        raise AppError(
            Hpc3ErrorCode.PARTITION_GPU_MISMATCH,
            f"Partition {partition!r} on {cluster['slug']!r} carries no "
            f"{gpu['model']} GPUs ({list(available)}); the job would pend forever.",
        )


def _check_partition_is_funded(
    cluster: ClusterFacts, partition: str, max_service_units: float
) -> None:
    """Reject a billed partition when the workspace has declared no budget for it.

    This refusal used to be unconditional, on the reasoning that an
    ``accept_billing`` field would make the limit something a run could turn
    off -- the same shape as declaring ``max_gpus_per_user: 999`` to raise a
    ceiling, which disables a check instead of changing the fact.

    That argument still holds, and this is not that. The allowance is not a
    per-run flag: it is the workspace's declared service-unit budget, the same
    number :func:`~hpc3.core.budget.check_projection` enforces the size of the
    spend against. A workspace that has declared none still cannot submit
    billed work, and the refusal now says so in terms of the budget rather
    than as a property of the package. Raising it is a deliberate edit to a
    declared cap, and the cap then binds how much may be spent -- which is
    changing the fact, not turning off the check.

    Args:
        cluster: The selected cluster.
        partition: Target partition.
        max_service_units: The workspace's declared service-unit cap. Zero
            means free work only.

    Raises:
        AppError: With ``PARTITION_BILLS`` if the partition's usage factor is
            above zero and no budget has been declared. The message names the
            measured factor and lists the free partitions, because the useful
            next step is usually which partition to use instead.
    """
    if not partition_bills(cluster, partition):
        return
    if max_service_units > 0.0:
        return
    factor = partition_facts(cluster, partition)["usage_factor"]
    free = [name for name in partition_names(cluster) if not partition_bills(cluster, name)]
    raise AppError(
        Hpc3ErrorCode.PARTITION_BILLS,
        f"Partition {partition!r} on {cluster['slug']!r} charges service units "
        f"(UsageFactor {factor}), and this workspace declares a service-unit "
        f"budget of 0. Free partitions on this cluster: {free}. To spend, raise "
        f"'max_service_units' in the workspace budget deliberately.",
    )


def _check_preemption_protection(
    cluster: ClusterFacts,
    partition: str,
    minutes: int,
    requeue: bool,
    checkpoint_steps: int,
    deterministic: bool,
) -> None:
    """Reject a long preemptible job that would lose everything if evicted.

    Args:
        cluster: The selected cluster.
        partition: Target partition.
        minutes: Requested wall clock.
        requeue: Whether Slurm should resubmit after preemption.
        checkpoint_steps: Steps between checkpoints; 0 means none.
        deterministic: Whether the workload replays identically from the
            start. For such a job requeue alone IS protection: a preempted
            run resubmits, replays, and produces the same result -- the
            whole run is a checkpoint at step zero. Rusted's pinned-regime
            matches are the workload this clause was measured against
            (replicated seed-for-seed across independent submissions,
            2026-09-01); a stochastic trainer restarting from step zero is
            not protected, which is what the checkpoint half still refuses.

    Raises:
        AppError: With ``PREEMPTIBLE_RUN_UNPROTECTED`` if the job is
            preemptible, longer than
            :data:`PREEMPTION_PROTECTION_THRESHOLD_MINUTES`, and lacks
            requeue paired with either checkpointing or deterministic
            replay. Requeue without either restarts a stochastic run from
            step zero as a DIFFERENT run, which is not protection.
    """
    if not partition_facts(cluster, partition)["preemptible"]:
        return
    if minutes <= PREEMPTION_PROTECTION_THRESHOLD_MINUTES:
        return
    if requeue and (checkpoint_steps > 0 or deterministic):
        return
    raise AppError(
        Hpc3ErrorCode.PREEMPTIBLE_RUN_UNPROTECTED,
        f"A {minutes}-minute job on preemptible {partition!r} needs 'requeue' "
        "paired with a positive 'checkpoint_steps' or with 'deterministic' "
        f"replay; got requeue={requeue}, checkpoint_steps={checkpoint_steps}, "
        f"deterministic={deterministic}. Preemption cancels the job.",
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


__all__ = [
    "MINUTES_PER_HOUR",
    "PREEMPTION_PROTECTION_THRESHOLD_MINUTES",
    "_check_partition_carries_gpu",
    "_check_partition_is_funded",
    "_check_preemption_protection",
    "_check_time_limit",
]
