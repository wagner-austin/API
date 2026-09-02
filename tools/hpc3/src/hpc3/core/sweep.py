"""Submitting many jobs from one template, as one array call.

A sweep used to be submitted member by member -- three SSH round trips each,
~13 seconds apiece, measured against a cluster that scheduled the jobs
instantly (rusted ab48, 2026-09-01). It is now one call: the whole member
table renders into one script, ``sbatch --array`` runs every document
position, and the ledger records each task individually. The mechanics live
in :mod:`hpc3.core.array_submit`; this module is the sweep-shaped door.

The size check happens in the contract, before any of this runs, so a sweep
that would pend against the QOS never reaches the cluster at all. Failure
atomicity is now stronger than the old loop's, not weaker: the loop could
die on member four leaving three live, while the array either submits as a
whole or refuses as a whole -- and every refusal happens before ``sbatch``.
"""

from __future__ import annotations

import pathlib

from hpc3.contracts.cluster import ClusterFacts
from hpc3.contracts.sweep import SweepSpec
from hpc3.core.array_submit import SubmittedMember, submit_array


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
    """Submit every member of a sweep as one job array.

    Args:
        spec: A sweep already validated by
            :func:`~hpc3.contracts.sweep.decode_sweep_spec`, so its template
            satisfies every submission rule and its size fits the QOS.
        host: SSH destination.
        script_dir: Absolute cluster directory to hold the batch script.
        log_dir: Absolute cluster directory for the tasks' output.
        ledger_path: Local append-only record, one entry per member.
        submitted_at: ISO-8601 timestamp for the records.
        cluster: The cluster whose measured limits each member is validated
            against.
        charge_account: Slurm account to bill, or empty for none.

    Returns:
        One record per member, in declaration order, each carrying its array
        task id.

    Raises:
        AppError: Through :func:`~hpc3.core.array_submit.submit_array` --
            the artifact-race refusal, the preflight codes, or
            ``REMOTE_COMMAND_FAILED``. Nothing has been submitted when any
            of them is raised.
    """
    return submit_array(
        spec,
        tuple(range(len(spec["members"]))),
        host=host,
        script_dir=script_dir,
        log_dir=log_dir,
        ledger_path=ledger_path,
        submitted_at=submitted_at,
        cluster=cluster,
        charge_account=charge_account,
    )


__all__ = ["SubmittedMember", "submit_sweep"]
