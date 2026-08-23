"""Submitting a job: validate, queue, record. In that order, always.

The script is written to the cluster and submitted by path rather than piped
to ``sbatch`` on stdin. Two reasons, both learned rather than assumed: the
submitted script remains on disk as the record of exactly what ran, and a
job that is later requeued after preemption re-reads that file, so it must
outlive the submitting process.

Two things this module does that a caller cannot decline, because a rule the
user has to remember is not a rule:

* It **preflights**. Every submission asks the scheduler for a verdict on the
  real uploaded script before queueing it. There is no flag to skip.
* It **records**. The ledger write happens before the id is returned, so a
  crash cannot leave a running job that nothing on this machine knows about.
"""

from __future__ import annotations

import pathlib

from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.cluster import ClusterFacts
from hpc3.contracts.job import JobSpec
from hpc3.contracts.layout import qualified_name
from hpc3.contracts.ledger import LedgerEntry
from hpc3.core import audit, ledger, preflight, remote

_SUBMIT_PREFIX = "Submitted batch job "


def parse_job_id(output: str) -> str:
    """Read the job id out of ``sbatch`` output.

    Args:
        output: The command's standard output, normally
            ``Submitted batch job 12345678``.

    Returns:
        The job id.

    Raises:
        AppError: With
            :attr:`~platform_core.errors.Hpc3ErrorCode.REMOTE_COMMAND_FAILED` if no
            line announces a submission, or if the announced id is not
            numeric. ``sbatch`` can exit zero while printing a warning
            instead of a submission, and returning an unusable id would defer
            the failure to the first status query.
    """
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith(_SUBMIT_PREFIX):
            continue
        job_id = stripped[len(_SUBMIT_PREFIX) :].strip()
        if not job_id.isdigit():
            raise AppError(
                Hpc3ErrorCode.REMOTE_COMMAND_FAILED,
                f"sbatch announced a non-numeric job id {job_id!r}.",
            )
        return job_id
    raise AppError(
        Hpc3ErrorCode.REMOTE_COMMAND_FAILED,
        f"sbatch printed no submission line; got {output.strip()!r}.",
    )


def submit(
    spec: JobSpec,
    *,
    host: str,
    script_dir: str,
    log_dir: str,
    ledger_path: pathlib.Path,
    submitted_at: str,
    cluster: ClusterFacts,
) -> str:
    """Render, upload and submit a job, recording it locally first.

    The ledger write happens BEFORE this function returns and before the
    caller can print anything. A job outlives the process that submitted it,
    so an id that exists only in a return value is an id that a crash loses --
    while the job it names keeps running on a shared machine with nobody able
    to find it. Recording is therefore not optional and has no flag.

    Args:
        spec: A spec already validated by
            :func:`~hpc3.contracts.job.decode_job_spec`.
        host: SSH destination.
        script_dir: Absolute cluster directory to hold the batch script.
        log_dir: Absolute cluster directory for the job's stdout and stderr.
        ledger_path: Local append-only record of submitted jobs.
        submitted_at: ISO-8601 timestamp for the record, supplied by the
            caller so this function reads no clock and stays testable.
        cluster: The cluster whose measured limits preflight and the audit
            record are taken from.

    Returns:
        The submitted job's id.

    Raises:
        AppError: With ``ENV_PATH_MISSING`` if the environment is absent,
            ``PREFLIGHT_REJECTED`` if the scheduler would refuse the job,
            ``PREFLIGHT_UNPARSABLE`` if its verdict cannot be read, or
            ``REMOTE_COMMAND_FAILED`` if a command failed or ``sbatch``
            announced no usable job id.
    """
    # Preflight is not a separate step a caller may skip. It uploads the
    # script and asks the scheduler whether it would be admitted; submission
    # then queues that same uploaded file. Making it a prefix of submit rather
    # than a sibling command means there is no path to the cluster that
    # bypasses validation, and no second upload to drift from the first.
    preflight.preflight(spec, host=host, script_dir=script_dir, log_dir=log_dir, cluster=cluster)

    label = qualified_name(spec["project"], spec["name"])
    output = remote.run_remote(host, f"cd {script_dir} && sbatch {label}.sbatch")
    job_id = parse_job_id(output)

    ledger.append(
        ledger_path,
        LedgerEntry(
            job_id=job_id,
            project=spec["project"],
            name=label,
            host=host,
            partition=spec["partition"],
            submitted_at=submitted_at,
            log_dir=log_dir,
        ),
    )
    audit.job_submitted(spec, host=host, job_id=job_id, cluster=cluster)
    return job_id


__all__ = ["parse_job_id", "submit"]
