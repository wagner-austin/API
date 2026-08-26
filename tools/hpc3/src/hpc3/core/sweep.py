"""Submitting many jobs from one template.

A sweep is submitted member by member, in declaration order, and stops on the
first failure. It does not roll back: the members already submitted are real
jobs holding real nodes, and cancelling them because a later one failed would
throw away work that is fine. The caller is told which member stopped the run
and which ids are already live, so the remainder can be submitted separately
once the cause is fixed.

The size check happens in the contract, before any of this runs, so a sweep
that would pend against the QOS never reaches the cluster at all.
"""

from __future__ import annotations

import pathlib

from hpc3.contracts.cluster import ClusterFacts
from hpc3.contracts.layout import qualified_name
from hpc3.contracts.sweep import SweepSpec, expand_sweep
from hpc3.core import audit, submit


class SubmittedMember:
    """One member of a sweep, and the id it was given.

    Attributes:
        name: The member's QUALIFIED job name -- the same string ``squeue``
            shows, so what the operator reads here is what they will search
            for.
        job_id: Id Slurm assigned.
    """

    __slots__ = ("job_id", "name")

    def __init__(self, name: str, job_id: str) -> None:
        """Record a submitted member.

        Args:
            name: The member's qualified job name.
            job_id: Id Slurm assigned.
        """
        self.name = name
        self.job_id = job_id


def submit_sweep(
    spec: SweepSpec,
    *,
    host: str,
    script_dir: str,
    log_dir: str,
    ledger_path: pathlib.Path,
    submitted_at: str,
    cluster: ClusterFacts,
    charge_account: str,
) -> list[SubmittedMember]:
    """Submit every member of a sweep.

    Args:
        spec: A sweep already validated by
            :func:`~hpc3.contracts.sweep.decode_sweep_spec`, so its template
            satisfies every submission rule and its size fits the QOS.
        host: SSH destination.
        script_dir: Absolute cluster directory to hold the batch scripts.
        log_dir: Absolute cluster directory for the jobs' output.
        ledger_path: Local append-only record. Each member is written as it
            is submitted, so a sweep that dies on member four leaves three
            findable jobs rather than three orphans.
        submitted_at: ISO-8601 timestamp for the records.
        cluster: The cluster whose measured limits each member is validated
            against.

    Returns:
        One record per member, in declaration order.

    Raises:
        AppError: With ``REMOTE_COMMAND_FAILED`` on the first member that
            could not be submitted. Members submitted before it stay running;
            nothing is rolled back, because a live job that is fine should not
            be cancelled for a later job's failure. They are already in the
            ledger, so they remain findable.
    """
    submitted: list[SubmittedMember] = []
    for member_spec in expand_sweep(spec):
        job_id = submit.submit(
            member_spec,
            host=host,
            script_dir=script_dir,
            log_dir=log_dir,
            ledger_path=ledger_path,
            submitted_at=submitted_at,
            cluster=cluster,
            charge_account=charge_account,
        )
        label = qualified_name(member_spec["project"], member_spec["name"])
        submitted.append(SubmittedMember(label, job_id))

    audit.sweep_submitted(
        host=host,
        project=spec["base"]["project"],
        base_name=spec["base"]["name"],
        job_ids=[member.job_id for member in submitted],
    )
    return submitted


__all__ = ["SubmittedMember", "submit_sweep"]
