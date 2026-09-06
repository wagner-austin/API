"""Submitting a rendered image build, and recording it like every other job.

WHY THIS EXISTS. Every other job in this package reaches the cluster through
:func:`~hpc3.core.submit.submit`, which preflights and then writes the ledger
row before it returns. The image build did not. Its documented recipe was a
raw ``ssh <host> 'cd <dir> && sbatch build.sbatch'``, and twenty-one builds
ran that way -- real jobs holding eight cores each, invisible to
``hpc3-trace``, unfindable by ``hpc3-watch`` because nothing had their ids.
The reverse-direction check added on 2026-08-28 reported the twenty-second as
``unclaimed``, correctly, and that finding is what this module answers.

WHY IT IS NOT ``submit()``. A run is described by a document and rendered into
a script here. A build is the other way round: ``hpc3-image`` already rendered
the script, from an image spec, and the resources in it describe *building*
rather than the thing built -- CPU-only, ``free``, two hours. There is no
:class:`~hpc3.contracts.job.JobSpec` to make and making one would mean
inventing fields the build does not have. So this submits the script that
exists, and reads the two facts the ledger needs out of that same script
rather than accepting a caller's claim about them.

THE NAME MUST AGREE, and this is the one refusal here. The ledger's ``name``
is the qualified ``<project>.<name>``, which is what makes a shared ``squeue``
self-describing and what ``hpc3-watch`` groups budgets by. A build script
rendered with ``--job-name img.abl-sif-v22`` records a row whose project
reads as ``img`` -- a project no workspace declares -- so the row would be in
the ledger and still not answer "whose work was this". Rather than rewrite
the caller's name or silently record a different one than Slurm shows, the
two are required to match and the refusal prints the ``hpc3-image`` invocation
that would make them.
"""

from __future__ import annotations

import pathlib

from platform_core.errors import AppError, Hpc3ErrorCode
from typing_extensions import TypedDict

from hpc3.contracts.cluster import ClusterFacts
from hpc3.contracts.ledger import LedgerEntry
from hpc3.core import audit, ledger, preflight, remote
from hpc3.core.image_layout import SBATCH_NAME
from hpc3.core.submit import parse_job_id

_JOB_NAME_DIRECTIVE = "#SBATCH -J "
_PARTITION_DIRECTIVE = "#SBATCH -p "


class BuildDirectives(TypedDict):
    """The two facts the ledger needs, read from the script that will run.

    Attributes:
        job_name: Name Slurm will show, from ``#SBATCH -J``.
        partition: Partition it goes to, from ``#SBATCH -p``. Read rather
            than assumed from
            :data:`~hpc3.core.image_sbatch.BUILD_PARTITION`: that constant
            says what this package renders today, and the file on the cluster
            is what actually runs. A build directory staged from an older
            render would otherwise be recorded against a partition it is not
            using.
    """

    job_name: str
    partition: str


def _directive(script: str, prefix: str, field: str) -> str:
    """Read one ``#SBATCH`` directive out of a rendered build script.

    Args:
        script: The script's full text.
        prefix: Directive prefix to find, including its trailing space.
        field: Human name of the field, for the error message.

    Returns:
        The directive's value, stripped.

    Raises:
        AppError: With ``IMAGE_BUILD_SCRIPT_UNREADABLE`` if the directive is
            absent or carries nothing. The first match wins because Slurm
            takes the first too; a script with two would be ambiguous to
            both, and this reads it the way the scheduler does.
    """
    for line in script.splitlines():
        if not line.startswith(prefix):
            continue
        value = line[len(prefix) :].strip()
        if value == "":
            raise AppError(
                Hpc3ErrorCode.IMAGE_BUILD_SCRIPT_UNREADABLE,
                f"{SBATCH_NAME} declares an empty {field} ({prefix.strip()}).",
            )
        return value
    raise AppError(
        Hpc3ErrorCode.IMAGE_BUILD_SCRIPT_UNREADABLE,
        f"{SBATCH_NAME} declares no {field} ({prefix.strip()}). Either it is not a "
        "script this package rendered, or the render is older than the directive.",
    )


def parse_build_directives(script: str) -> BuildDirectives:
    """Read the job name and partition from a rendered build script.

    Args:
        script: Contents of ``build.sbatch``, as it exists on the cluster.

    Returns:
        The two directives.

    Raises:
        AppError: With ``IMAGE_BUILD_SCRIPT_UNREADABLE`` if either is absent
            or empty.
    """
    return BuildDirectives(
        job_name=_directive(script, _JOB_NAME_DIRECTIVE, "job name"),
        partition=_directive(script, _PARTITION_DIRECTIVE, "partition"),
    )


def check_name_agrees(*, declared: str, rendered: str) -> None:
    """Refuse a build whose script names a different job than the ledger will.

    Args:
        declared: The qualified name this submission will record.
        rendered: The name ``#SBATCH -J`` gives, which is what Slurm shows.

    Raises:
        AppError: With ``IMAGE_BUILD_NAME_MISMATCH`` when they differ. The
            ledger row would then name a job no ``squeue`` search finds,
            which is the precise defect the ledger exists to prevent, arrived
            at from the other side.
    """
    if declared == rendered:
        return
    raise AppError(
        Hpc3ErrorCode.IMAGE_BUILD_NAME_MISMATCH,
        f"{SBATCH_NAME} renders the job name {rendered!r} and this submission would "
        f"record {declared!r}. The ledger row would name a job the cluster never "
        f"shows under that name. Re-render with the qualified name -- "
        f"hpc3-image ... --job-name {declared} -- and stage it again.",
    )


def submit_build(
    *,
    host: str,
    image_dir: str,
    project: str,
    label: str,
    artifact: str,
    ledger_path: pathlib.Path,
    submitted_at: str,
    submitter: str,
    cluster: ClusterFacts,
) -> str:
    """Preflight, submit and record an already-rendered image build.

    The ordering is :func:`~hpc3.core.submit.submit`'s and for the same
    reasons: nothing reaches the queue unvalidated, and the ledger row is
    written before the id is returned, so a crash cannot leave a running
    build that nothing on this machine knows about.

    Args:
        host: SSH destination.
        image_dir: Absolute cluster directory holding ``build.sbatch``. Also
            where the build writes its logs and its image, so it is the
            ledger's ``log_dir`` too -- a build's logs are ``build-<id>.out``
            rather than ``<name>-<id>.out``, so the directory is the part of
            the convention that still holds.
        project: Workspace project this build belongs to.
        label: Qualified job name to record, which must match the script's.
        artifact: Absolute path of the ``.sif`` this build produces, so
            ``hpc3-trace`` can answer which job built a given image -- a
            question nothing could answer before.
        ledger_path: Local append-only record of submitted jobs.
        submitted_at: ISO-8601 timestamp, supplied by the caller so this
            function reads no clock.
        submitter: The submitting session's agent-board label, or ``""``
            when it declared none -- supplied by the caller so this
            function reads no environment.
        cluster: The cluster whose measured limits the verdict is checked
            against.

    Returns:
        The submitted job's id.

    Raises:
        AppError: With ``REMOTE_COMMAND_FAILED`` if the script cannot be read
            or ``sbatch`` announces no usable id, ``IMAGE_BUILD_SCRIPT_UNREADABLE``
            if it carries no job name or partition, ``IMAGE_BUILD_NAME_MISMATCH``
            if that name is not ``label``, ``PREFLIGHT_REJECTED`` if Slurm
            would refuse it, or ``PREFLIGHT_UNPARSABLE`` if its verdict cannot
            be read.
    """
    script_path = f"{image_dir}/{SBATCH_NAME}"
    directives = parse_build_directives(remote.run_remote(host, f"cat {script_path}"))
    check_name_agrees(declared=label, rendered=directives["job_name"])

    # The same non-skippable prefix `submit` has, against the same bytes that
    # will run. No env probe: a build has no environment yet -- producing one
    # is what it is for -- so the check that would run here is the one thing
    # that cannot be true until after it succeeds.
    probe = f'cd {image_dir} && sbatch --test-only {SBATCH_NAME} 2>&1; echo "rc=$?"'
    output = remote.run_remote(host, probe)
    if "rc=0" not in output:
        raise AppError(
            Hpc3ErrorCode.PREFLIGHT_REJECTED,
            f"Slurm would refuse the build {label!r}: {output.strip()}",
        )
    preflight.parse_test_only(output, cluster)

    job_id = parse_job_id(remote.run_remote(host, f"cd {image_dir} && sbatch {SBATCH_NAME}"))
    ledger.append(
        ledger_path,
        LedgerEntry(
            job_id=job_id,
            project=project,
            name=label,
            host=host,
            partition=directives["partition"],
            submitted_at=submitted_at,
            log_dir=image_dir,
            # A build is not a numerical run. Recording True because the
            # project's runs are deterministic would assert something about
            # this job that nothing established.
            deterministic=False,
            experiment={"kind": "image-build", "image_dir": image_dir},
            # Empty, matching a directory-environment run: the build produces
            # the image, so it cannot have run inside one. That is a positive
            # fact and differs from every real digest rather than matching any.
            image_digest="",
            submitter=submitter,
            artifact=artifact,
        ),
    )
    audit.image_build_submitted(
        host=host,
        job_id=job_id,
        project=project,
        label=label,
        partition=directives["partition"],
        artifact=artifact,
        cluster=cluster,
    )
    return job_id


__all__ = [
    "BuildDirectives",
    "check_name_agrees",
    "parse_build_directives",
    "submit_build",
]
