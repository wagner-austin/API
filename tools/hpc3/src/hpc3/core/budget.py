"""Enforcing our declared share, before submission and while running.

Projection and observation answer different questions and neither replaces
the other. A projection is arithmetic over what was asked for; it catches a
flood before it starts and is the only check available at submission time.
An observation is arithmetic over what Slurm reports; it catches a projection
that was wrong -- a job that ran longer than its estimate, a requeue that
doubled the cost, a member submitted outside the sweep.

Projection uses the requested wall clock, not an expected runtime. A job that
finishes early costs less than projected and that is fine; a projection built
on optimism would admit a sweep that cannot fit its own cap.
"""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.budget import Budget, Consumption
from hpc3.contracts.cluster import ClusterFacts, gpu_count
from hpc3.contracts.job import MINUTES_PER_HOUR, JobSpec
from hpc3.contracts.status import JobStatus, gpu_hours, service_units


def project(specs: Sequence[JobSpec], cluster: ClusterFacts) -> Consumption:
    """Compute what a set of specs would consume if each ran to its limit.

    Args:
        specs: Specs to total.
        cluster: The cluster whose measured usage factors apply.

    Returns:
        Projected consumption. Service units are always zero and that is
        structural, not a simplification: every spec here has passed
        :func:`~hpc3.contracts.job.decode_job_spec`, which refuses any
        partition with a non-zero usage factor. Multiplying by a factor that
        is provably zero, and then comparing the product against a cap, would
        be a check that cannot fail -- and a check that cannot fail still
        reads as protection.

        The service-unit cap is enforced against OBSERVED usage instead, where
        it is a real tripwire: a non-zero reading there means a partition this
        package believes is free is not. That has happened once already --
        ``free-gpu32`` was recorded as billing for a day when it does not --
        so the direction it guards is worth guarding.
    """
    projected_gpu_hours = 0.0
    for spec in specs:
        projected_gpu_hours += gpu_count(spec["gpu"]) * spec["minutes"] / MINUTES_PER_HOUR
    return Consumption(
        gpu_hours=projected_gpu_hours,
        service_units=0.0,
        jobs=len(specs),
    )


def observe(statuses: Sequence[JobStatus], cluster: ClusterFacts) -> Consumption:
    """Compute what a set of jobs has actually consumed.

    Args:
        statuses: Accounting rows to total.
        cluster: The cluster whose measured usage factors apply.

    Returns:
        Observed consumption so far. For running jobs this grows on every
        query; it is a reading, not a final figure.
    """
    return Consumption(
        gpu_hours=sum(gpu_hours(status) for status in statuses),
        service_units=sum(service_units(status, cluster) for status in statuses),
        jobs=len(statuses),
    )


def check_projection(
    budget: Budget, specs: Sequence[JobSpec], cluster: ClusterFacts
) -> Consumption:
    """Refuse a set of specs that would exceed the declared budget.

    Args:
        budget: The caps to enforce.
        specs: Specs about to be submitted.
        cluster: The cluster whose measured usage factors apply.

    Returns:
        The projection, so a caller that passes can report it rather than
        recomputing.

    Raises:
        AppError: With
            :attr:`~platform_core.errors.Hpc3ErrorCode.BUDGET_PROJECTION_EXCEEDED`
            if the GPU-hour cap would be broken. Raised before anything is
            submitted, which is the whole point: a flood that has started is
            no longer a budget question.

            The service-unit cap is not checked here -- see :func:`project`
            for why it cannot be exceeded by anything this package will
            submit. :func:`check_consumption` enforces it against what
            actually ran.
    """
    projected = project(specs, cluster)
    if projected["gpu_hours"] > budget["max_gpu_hours"]:
        raise AppError(
            Hpc3ErrorCode.BUDGET_PROJECTION_EXCEEDED,
            f"{projected['jobs']} job(s) would use {projected['gpu_hours']:.1f} GPU-hours, "
            f"over the declared cap of {budget['max_gpu_hours']:.1f}. "
            "Nothing was submitted.",
        )
    return projected


def check_consumption(
    budget: Budget, statuses: Sequence[JobStatus], cluster: ClusterFacts
) -> Consumption:
    """Report that running jobs have passed the declared budget.

    Args:
        budget: The caps to enforce.
        statuses: Accounting rows for the jobs to total.
        cluster: The cluster whose measured usage factors apply.

    Returns:
        The observed consumption, so a caller that passes can report it.

    Raises:
        AppError: With
            :attr:`~platform_core.errors.Hpc3ErrorCode.BUDGET_CONSUMPTION_EXCEEDED`
            if either cap has been passed. This does NOT cancel anything --
            stopping work is a decision with its own consequences and belongs
            to the operator, not to a reporting call. It fails loudly so the
            overrun cannot be scrolled past.
    """
    observed = observe(statuses, cluster)
    # Unlike the projection, BOTH caps are live here. A non-zero service-unit
    # reading means a partition this package admitted as free is charging,
    # which is a wrong fact in the cluster module rather than an overspend by
    # the operator -- and the only place that can ever surface.
    if observed["gpu_hours"] > budget["max_gpu_hours"]:
        raise AppError(
            Hpc3ErrorCode.BUDGET_CONSUMPTION_EXCEEDED,
            f"{observed['jobs']} job(s) have used {observed['gpu_hours']:.1f} GPU-hours, "
            f"over the declared cap of {budget['max_gpu_hours']:.1f}. "
            "Nothing was cancelled; that is your call.",
        )
    if observed["service_units"] > budget["max_service_units"]:
        raise AppError(
            Hpc3ErrorCode.BUDGET_CONSUMPTION_EXCEEDED,
            f"{observed['jobs']} job(s) have spent {observed['service_units']:.1f} SU, "
            f"over the declared cap of {budget['max_service_units']:.1f}. "
            "Nothing was cancelled; that is your call.",
        )
    return observed


__all__ = ["check_consumption", "check_projection", "observe", "project"]
