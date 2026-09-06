"""Submitting a pipeline, wiring each stage to the id of the one before it.

Stages go out in order. Every stage after the first carries
``--dependency=afterok:<previous id>``, so nothing downstream of a failure
computes a result on top of it.

Nothing is rolled back on a mid-chain failure, for the same reason a sweep
does not roll back: the stages already submitted are real jobs, and the ones
that have not run yet are blocked behind a predecessor that is fine. What the
caller gets back is which stages exist and which id each got, so the remainder
can be submitted against the last good id once the cause is fixed.

The dependency is built from an id this package received from ``sbatch`` a
moment earlier, not from anything a document said, so it is not re-validated.
That is the reason ``depends_on`` is refused in a chain document at all: the
only ids that reach a chain's ``--dependency`` are ones Slurm just issued.
"""

from __future__ import annotations

import pathlib

from hpc3.contracts.chain import ChainSpec
from hpc3.contracts.cluster import ClusterFacts
from hpc3.contracts.dependency import AFTER_OK, Dependency
from hpc3.contracts.job import JobSpec
from hpc3.contracts.layout import qualified_name
from hpc3.core import audit, submit
from hpc3.core.array_submit import SubmittedMember


def _waiting_on(previous: str) -> Dependency:
    """Build the dependency one stage places on the stage before it.

    Args:
        previous: Id Slurm assigned the preceding stage.

    Returns:
        An ``afterok`` dependency on exactly that job. Never ``afterany``: a
        stage that runs after its input failed produces a second wrong answer
        and costs the wall clock to do it.
    """
    return Dependency(kind=AFTER_OK, job_ids=[previous])


def submit_chain(
    spec: ChainSpec,
    *,
    host: str,
    script_dir: str,
    log_dir: str,
    ledger_path: pathlib.Path,
    submitted_at: str,
    submitter: str,
    cluster: ClusterFacts,
    charge_account: str,
) -> list[SubmittedMember]:
    """Submit every stage of a chain, in order, each waiting on the last.

    Args:
        spec: A chain already validated by
            :func:`~hpc3.contracts.chain.decode_chain_spec`, so every stage
            satisfies every submission rule before the first one is sent.
        host: SSH destination.
        script_dir: Absolute cluster directory to hold the batch scripts.
        log_dir: Absolute cluster directory for the stages' output.
        ledger_path: Local append-only record, written per stage as it is
            submitted.
        submitted_at: ISO-8601 timestamp for the records.
        submitter: The submitting session's agent-board label, or ``""``
            when it declared none. One value for every stage: a chain has
            one submitter.
        cluster: The cluster whose measured limits each stage is validated
            against.

    Returns:
        One record per stage, in execution order.

    Raises:
        AppError: With ``REMOTE_COMMAND_FAILED`` on the first stage that could
            not be submitted. Earlier stages stay queued and are already in
            the ledger; later ones were never sent.
    """
    submitted: list[SubmittedMember] = []
    previous: str | None = None

    for stage in spec["stages"]:
        waiting: JobSpec = (
            stage if previous is None else JobSpec({**stage, "depends_on": _waiting_on(previous)})
        )
        job_id = submit.submit(
            waiting,
            host=host,
            script_dir=script_dir,
            log_dir=log_dir,
            ledger_path=ledger_path,
            submitted_at=submitted_at,
            submitter=submitter,
            cluster=cluster,
            charge_account=charge_account,
        )
        submitted.append(SubmittedMember(qualified_name(stage["project"], stage["name"]), job_id))
        previous = job_id

    audit.chain_submitted(
        host=host,
        project=spec["stages"][0]["project"],
        job_ids=[member.job_id for member in submitted],
    )
    return submitted


__all__ = ["submit_chain"]
