"""Structured audit events for everything this package sends to the cluster.

Separate from the CLI's report, and not a duplicate of it. The report tells
the operator what just happened; this is the durable record of what was sent,
which matters because a job runs for hours after the process that submitted it
has exited, and because a billed submission is a spending decision that should
be traceable without re-deriving it from Slurm.

Every event carries the fields a later reader needs to reconstruct the
decision: which partition, which GPU, whether billing applied, and the id
Slurm assigned. Emission goes through the core hook rather than a module-level
logger, so a test asserts on the event name and its exact fields instead of
scraping captured output.
"""

from __future__ import annotations

from hpc3.contracts.cluster import ClusterFacts, partition_bills
from hpc3.contracts.job import JobSpec
from hpc3.contracts.layout import qualified_name
from hpc3.core import _test_hooks

JOB_SUBMITTED = "hpc3_job_submitted"
SWEEP_SUBMITTED = "hpc3_sweep_submitted"
FILES_STAGED = "hpc3_files_staged"


def job_submitted(spec: JobSpec, *, host: str, job_id: str, cluster: ClusterFacts) -> None:
    """Record a successful submission.

    Args:
        spec: The spec that was submitted.
        host: SSH destination it went to.
        job_id: Id Slurm assigned.
        cluster: The cluster it went to. Recorded by slug and consulted for
            whether the partition bills, so a later reader can tell a free
            submission from a spending one without re-deriving it.
    """
    _test_hooks.log_event(
        JOB_SUBMITTED,
        {
            "job_id": job_id,
            # The qualified label, not the bare name: this is the string that
            # appears in `squeue`, in the log filenames and in the ledger, so
            # a reader holding only this event can find all three.
            "job_name": qualified_name(spec["project"], spec["name"]),
            "project": spec["project"],
            "host": host,
            "cluster": cluster["slug"],
            "partition": spec["partition"],
            "gpu": spec["gpu"],
            "gpu_count": spec["gpu_count"],
            "cpus": spec["cpus"],
            "minutes": spec["minutes"],
            "bills": partition_bills(cluster, spec["partition"]),
            "requeue": spec["requeue"],
            "checkpoint_steps": spec["checkpoint_steps"],
        },
    )


def sweep_submitted(*, host: str, project: str, base_name: str, job_ids: list[str]) -> None:
    """Record a completed sweep.

    Args:
        host: SSH destination the members went to.
        project: Body of work every member belongs to.
        base_name: Template name the members were derived from.
        job_ids: Ids Slurm assigned, in member order.
    """
    _test_hooks.log_event(
        SWEEP_SUBMITTED,
        {
            "host": host,
            "project": project,
            "base_name": qualified_name(project, base_name),
            "members": len(job_ids),
            "job_ids": ",".join(job_ids),
        },
    )


def files_staged(*, host: str, destination: str, count: int, provenance: str) -> None:
    """Record a staging operation whose files all verified on the cluster.

    Args:
        host: SSH destination.
        destination: Directory the files were placed in.
        count: Number of files placed and verified.
        provenance: The manifest's record of where the bytes came from,
            already rendered. Carried into the event because "what was staged
            here" is a question asked months later, when the manifest may have
            been regenerated and the log is what is left.
    """
    _test_hooks.log_event(
        FILES_STAGED,
        {"host": host, "destination": destination, "files": count, "provenance": provenance},
    )


__all__ = [
    "FILES_STAGED",
    "JOB_SUBMITTED",
    "SWEEP_SUBMITTED",
    "files_staged",
    "job_submitted",
    "sweep_submitted",
]
