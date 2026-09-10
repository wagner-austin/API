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

REPLAY_AFFORDABLE_CEILING_MINUTES = 240
"""Above this, DETERMINISM STOPS COUNTING AS PROTECTION and only a checkpoint does.

THE BYPASS THIS CLOSES, MEASURED. On 2026-09-10 a 600-minute question-set
benchmark was admitted to ``free-gpu`` -- ``PreemptMode=CANCEL``, verified by
``scontrol`` -- and submitted, because ``hpc3-mi.json`` declares
``deterministic: true`` for the whole project. ``survives_eviction`` was
therefore true and the rule returned before it could refuse anything. Nothing
was wrong with the code; the escape simply had no ceiling.

WHY DETERMINISM IS WEAKER PROTECTION THAN IT LOOKS, which is the whole reason
a ceiling belongs here. A checkpoint makes PARTIAL WORK survive eviction.
Determinism does not: it makes the RESULT reproducible, so an evicted run has
lost no science and every minute it had already spent. Its protection IS the
re-run -- and :data:`PREEMPTION_PROTECTION_THRESHOLD_MINUTES` exists precisely
because a re-run stops being cheap at some length. Letting determinism satisfy
the rule without a ceiling says a re-run is always affordable in the same
module that declares it is not affordable past an hour.

THE NUMBER IS A JUDGEMENT AND IS ANCHORED RATHER THAN INVENTED, in the way the
60-minute threshold above it is. 240 is the largest ``minutes`` any workspace
in this repository declares as its project default, so it is this workspace's
own existing statement of a normal unit of work it is willing to spend. A
replay costing one such unit is affordable; the run that provoked this was 2.5
of them. It wants re-measuring against a real preemption rate on ``free-gpu``,
which nobody has measured, and until then it is a bound and not a finding.

AND THE REPLAY MAY NOT EVEN RETURN THE SAME NUMBER, which makes this ceiling
a floor on the problem rather than a full answer. Measured 2026-09-10 by
another session over one 3,042-item set under full determinism pins: a repeat
run and an IMAGE CHANGE spanning a refactor of the training path each differed
in 0 of 12,168 score elements, while changing the training CARD differed in
100% of them and moved accuracy by -0.0986 and +2.6298 points. Determinism is
a property of the payload AND the hardware.

Scoped honestly, because that measurement compared two card MODELS and this
cluster does not let a job leave its model unpinned: ``GPU_TYPE_UNPINNED``
already refuses a generic request, so a replay here returns to the model it
named. What is NOT established either way is whether two cards of the SAME
model agree. So the hardware axis is guarded by a different rule than this
one, and ``deterministic`` is not what is holding it.

WHAT IT DOES NOT DO. It does not demand ``requeue`` on a CANCEL partition,
where that flag is inert -- that reasoning is unchanged and correct. A job
under this ceiling is unaffected, which is deliberate: the deterministic array
workloads this repository already runs on ``free-gpu`` sit well below it and
are resubmitted by ``hpc3-campaign``, and breaking them to close a hole they
are not in would be a worse trade than the hole.
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
    resumes_from_checkpoint: bool,
    deterministic: bool,
) -> None:
    """Reject a long preemptible job that would lose everything if evicted.

    Args:
        cluster: The selected cluster.
        partition: Target partition.
        minutes: Requested wall clock.
        requeue: Whether the script carries ``--requeue``. Only protection
            where the partition's mode is ``REQUEUE``, because that is the
            only mode in which Slurm resubmits anything. Under ``CANCEL``
            the flag is inert -- measured 2026-09-02, when a wave took 22
            array tasks carrying it straight to terminal PREEMPTED -- so it
            is not demanded there.
        resumes_from_checkpoint: Whether the payload checkpoints and resumes
            from one. Asserted by the operator; nothing HERE can check it,
            and that is the weak point of this whole rule -- a declaration
            is exactly as good as the payload behind it, and a false one
            buys the run past this refusal with a sentence.
            So it is checked where it can be: the image spec carries a smoke
            command that exercises the resume INSIDE the built image, and it
            fails the build rather than the run. That is the order the mi
            documents follow -- code, then an image that proves it, then the
            declaration -- and it is the reason a document may not turn this
            flag on in the same breath as the payload learning to checkpoint.
        deterministic: Whether the workload replays identically from the
            start. For such a job the whole run is a checkpoint at step
            zero: resubmit it and the same result comes back. Rusted's
            pinned-regime matches are the workload this clause was measured
            against (replicated seed-for-seed across independent
            submissions, 2026-09-01); a stochastic trainer restarting from
            step zero is not protected, which is what the checkpoint half
            still refuses.

    The check is in two parts because eviction poses two questions. Does the
    work survive it -- a checkpoint to resume from, or a replay that returns
    the same answer? And does anything resubmit? Slurm answers the second
    only under ``REQUEUE``; under ``CANCEL`` the resubmission comes from a
    campaign or a person, outside anything this guard can inspect, so it
    checks the first question alone rather than demanding a flag that does
    nothing.

    Raises:
        AppError: With ``REPLAY_EXCEEDS_AFFORDABLE_LOSS`` if the partition
            preempts, the job is longer than
            :data:`REPLAY_AFFORDABLE_CEILING_MINUTES`, and it carries no
            checkpoint -- whatever it declares about determinism, because
            replay protects the result and not the hours.
        AppError: With ``PREEMPTIBLE_RUN_UNPROTECTED`` if the partition
            preempts, the job is longer than
            :data:`PREEMPTION_PROTECTION_THRESHOLD_MINUTES`, and the work
            would not survive eviction -- or, on a ``REQUEUE`` partition,
            if it would survive but nothing asked Slurm to bring it back.
    """
    mode = partition_facts(cluster, partition)["preempt_mode"]
    if mode == "OFF":
        return
    if minutes <= PREEMPTION_PROTECTION_THRESHOLD_MINUTES:
        return
    replays_instead_of_resuming = deterministic and not resumes_from_checkpoint
    if replays_instead_of_resuming and minutes > REPLAY_AFFORDABLE_CEILING_MINUTES:
        # NARROWED TO THE RUNS THE DETERMINISM ESCAPE WOULD HAVE ADMITTED,
        # which the existing suite is what forced. The first version asked
        # only `not resumes_from_checkpoint`, so it also caught runs
        # declaring NEITHER protection -- and handed them a message about an
        # unaffordable replay when their real defect is that nothing would
        # survive eviction at all. Four tests failed on exactly that, and
        # they were right to: a refusal that names the wrong defect sends
        # the submitter to fix the wrong thing.
        #
        # So this fires only where `deterministic` is doing the admitting.
        # Below, the job cannot survive at all and keeps its own refusal.
        raise AppError(
            Hpc3ErrorCode.REPLAY_EXCEEDS_AFFORDABLE_LOSS,
            f"A {minutes}-minute job on {partition!r} (PreemptMode={mode}) has no "
            f"checkpoint, so eviction costs the whole run and it starts again at "
            f"zero. deterministic={deterministic} makes the RESULT reproducible, "
            f"not the elapsed time recoverable, and past "
            f"{REPLAY_AFFORDABLE_CEILING_MINUTES} minutes that replay is the "
            f"expense this rule exists to refuse. It needs "
            f"'resumes_from_checkpoint' -- a payload that writes progress and "
            f"picks it up on restart -- or a wall clock at or under "
            f"{REPLAY_AFFORDABLE_CEILING_MINUTES} minutes, or a partition that "
            f"does not preempt.",
        )
    survives_eviction = resumes_from_checkpoint or deterministic
    if survives_eviction and (mode == "CANCEL" or requeue):
        return
    if not survives_eviction:
        raise AppError(
            Hpc3ErrorCode.PREEMPTIBLE_RUN_UNPROTECTED,
            f"A {minutes}-minute job on {partition!r} (PreemptMode={mode}) would "
            "lose everything if evicted; got "
            f"resumes_from_checkpoint={resumes_from_checkpoint}, "
            f"deterministic={deterministic}. It needs 'resumes_from_checkpoint' "
            "so a restart picks up where it stopped, or 'deterministic' replay "
            "so a resubmission returns the same result.",
        )
    raise AppError(
        Hpc3ErrorCode.PREEMPTIBLE_RUN_UNPROTECTED,
        f"A {minutes}-minute job on {partition!r} survives eviction but nothing "
        f"would bring it back: PreemptMode={mode} means Slurm resubmits a "
        "preempted job, and this one does not carry 'requeue'.",
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
