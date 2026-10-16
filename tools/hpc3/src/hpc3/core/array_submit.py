"""Submitting a whole sweep as one job array: validate, queue, record, once.

The member-by-member sweep loop paid three SSH round trips per member --
upload, preflight, sbatch -- at ~13 seconds each, which made submission the
bottleneck of a pipeline whose cluster scheduled everything instantly
(rusted ab48, 96 members, ~18 minutes of pure submission, 2026-09-01). The
array is the same members behind ONE upload, ONE preflight, ONE ``sbatch``.

Everything the single-job path refuses, this refuses, by the same machinery:

* The artifact race is checked for EVERY selected member against one account
  enumeration, before anything is uploaded.
* Preflight uploads the real array script and asks ``sbatch --test-only``
  with the real ``--array`` argument, so the bytes and the shape the
  scheduler admits are the bytes and shape the submission runs.
* The ledger is written per member -- task id ``<base>_<index>`` against the
  member's own qualified name and artifact -- BEFORE the id is returned, so
  a crash cannot leave forty-eight running tasks nobody can find.

Which tasks run is the ``indices`` argument, always explicit: a fresh sweep
passes every document position, a converging campaign passes the sparse gap,
and both run against a script whose member table never varies -- see
:mod:`hpc3.core.array_sbatch` for why that table is the record.
"""

from __future__ import annotations

import pathlib

from platform_core.cluster_layout import qualified_name
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.array import array_task_id, format_array_argument
from hpc3.contracts.cluster import ClusterFacts
from hpc3.contracts.ledger import LedgerEntry
from hpc3.contracts.preflight import Admission
from hpc3.contracts.sweep import SweepSpec, expand_sweep
from hpc3.core import audit, code_claim, drift, entrypoint, env_probe, ledger, remote
from hpc3.core.array_sbatch import render_array_sbatch
from hpc3.core.inflight import (
    check_artifact_is_free,
    check_no_unrecorded_live_job,
    claimed_artifacts,
)
from hpc3.core.preflight import check_env_path, dependency_hint, parse_test_only
from hpc3.core.squeue import account_command, parse_account_output
from hpc3.core.submit import SubmittedMember, parse_job_id


def selected_members(spec: SweepSpec, indices: tuple[int, ...]) -> list[tuple[int, str]]:
    """Resolve the indices being submitted into (index, qualified name) rows.

    Args:
        spec: The validated sweep.
        indices: Document positions to run, strictly increasing.

    Returns:
        One row per index, in index order.

    Raises:
        AppError: With ``ARRAY_ID_UNPARSABLE`` when an index names no member.
            The index list is the campaign's bookkeeping, and a position past
            the table means that bookkeeping and the document disagree --
            submitting anything from that state would run the wrong member.
    """
    members = expand_sweep(spec)
    rows: list[tuple[int, str]] = []
    for index in indices:
        if index < 0 or index >= len(members):
            raise AppError(
                Hpc3ErrorCode.ARRAY_ID_UNPARSABLE,
                f"array index {index} names no member: the sweep declares "
                f"{len(members)} member(s). The index bookkeeping and the "
                "document disagree, and a submission built from that state "
                "would run the wrong member.",
            )
        member = members[index]
        rows.append((index, qualified_name(member["project"], member["name"])))
    return rows


def array_preflight(
    spec: SweepSpec,
    indices: tuple[int, ...],
    *,
    host: str,
    script_dir: str,
    log_dir: str,
    cluster: ClusterFacts,
    charge_account: str,
    repo: pathlib.Path,
) -> Admission:
    """Validate the array against the live scheduler without running it.

    Uploads the real array script and dry-runs it with the real ``--array``
    argument. One environment probe, not one per member: the members share
    the template's environment by construction.

    Args:
        spec: The validated sweep.
        indices: Document positions the submission will run.
        host: SSH destination.
        script_dir: Absolute cluster directory to hold the script.
        log_dir: Absolute cluster directory the script names for output.
        cluster: The cluster whose measured limits the verdict is decoded
            against.
        charge_account: Slurm account to bill, or empty for none.
        repo: The project's repository on this machine, which the image's
            code is compared against (:mod:`hpc3.core.drift`).

    Returns:
        The scheduler's verdict and the selected members' code compared with
        the submitter's. ``sbatch --test-only`` on an array answers with the
        same single verdict line a plain job gets (measured, probe job
        55678542), so the single-job parser reads it unchanged.

    Raises:
        AppError: With ``ENV_PATH_MISSING`` / ``ENV_PACKAGE_MISMATCH`` /
            ``ENV_PROBE_UNREADABLE`` / ``ENV_INTERPRETER_BORROWED`` from the
            environment checks, ``REPO_COMMIT_NOT_HEAD`` /
            ``REPO_COMMIT_PROBE_UNREADABLE`` from the code claim,
            ``ENTRYPOINT_IMAGE_UNSTAMPED`` / ``ENTRYPOINT_NOT_INSTALLED`` /
            ``ENTRYPOINT_PROBE_UNREADABLE`` when the image cannot say which
            commit built it, the environment cannot start a selected
            member's ``python -m`` module, or cannot say,
            ``ENTRYPOINT_IMAGE_COMMIT_UNKNOWN`` /
            ``ENTRYPOINT_DRIFT_UNREADABLE`` when ``repo`` does not hold the
            image's commit or git cannot compare it, ``PREFLIGHT_REJECTED`` when Slurm
            refuses -- carrying Slurm's own
            reason plus the dependency hint when the base waits on a job that
            can no longer satisfy it -- or ``PREFLIGHT_UNPARSABLE`` when the
            verdict cannot be read.
    """
    base = spec["base"]
    check_env_path(host, base)
    env_probe.verify_environment(
        host, base["env_path"], base["pinned_packages"], image=base["image"]
    )
    # One claim for the whole array, like the environment: every member runs
    # from the template's checkout, and only `member` is added per row.
    code_claim.check_code_claim(host, base["experiment"])

    remote.make_directory(host, script_dir)
    remote.make_directory(host, log_dir)
    # One probe for every selected member's command: they share the
    # template's environment, but each member may start its own module.
    members = expand_sweep(spec)
    reading = entrypoint.check_entrypoints(
        host,
        [members[index]["command"] for index in indices],
        env_path=base["env_path"],
        image=base["image"],
        working_directory=script_dir,
    )
    image_drift = drift.measure_drift(repo, reading)
    script = render_array_sbatch(spec, log_dir=log_dir, charge_account=charge_account)
    label = qualified_name(base["project"], base["name"])
    remote.put_bytes(host, f"{script_dir}/{label}.sbatch", script.encode("utf-8"))

    expression = format_array_argument(indices, spec["throttle"])
    probe = (
        f"cd {script_dir} && sbatch --test-only --array={expression} {label}.sbatch 2>&1; "
        'echo "rc=$?"'
    )
    output = remote.run_remote(host, probe)
    if "rc=0" not in output:
        raise AppError(
            Hpc3ErrorCode.PREFLIGHT_REJECTED,
            f"Slurm would refuse array {label!r} ({expression}): {output.strip()}"
            + dependency_hint(base, output),
        )
    return Admission(verdict=parse_test_only(output, cluster), image_drift=image_drift)


def submit_array(
    spec: SweepSpec,
    indices: tuple[int, ...],
    *,
    host: str,
    script_dir: str,
    log_dir: str,
    ledger_path: pathlib.Path,
    submitted_at: str,
    submitter: str,
    cluster: ClusterFacts,
    charge_account: str,
    repo: pathlib.Path,
) -> list[SubmittedMember]:
    """Render, upload and submit the selected members as one array.

    Args:
        spec: A sweep already validated by
            :func:`~hpc3.contracts.sweep.decode_sweep_spec`.
        indices: Document positions to run, strictly increasing. A fresh
            sweep passes them all; a campaign passes the gap.
        host: SSH destination.
        script_dir: Absolute cluster directory to hold the script.
        log_dir: Absolute cluster directory for the tasks' output.
        ledger_path: Local append-only record, written per member before the
            ids are returned.
        submitted_at: ISO-8601 timestamp for the records, supplied by the
            caller so this function reads no clock.
        submitter: The submitting session's agent-board label, or ``""``
            when it declared none -- supplied by the caller so this
            function reads no environment. One value for every member: an
            array has one submitter.
        cluster: The cluster whose measured limits preflight and the audit
            record are taken from.
        charge_account: Slurm account to bill, or empty for none.
        repo: The project's repository on this machine, which the image's
            code is compared against.

    Returns:
        One record per selected member, in index order, each carrying its
        task id ``<base>_<index>`` and the drift of the modules it starts.

    Raises:
        AppError: With ``ARRAY_INDICES_EMPTY`` / ``ARRAY_ID_UNPARSABLE`` on
            bad indices, ``LIVE_JOB_UNRECORDED`` when a live job carries the
            array's name or a selected member's and the ledger has no row
            for it, ``ARTIFACT_ALREADY_IN_FLIGHT`` when a recorded live job
            is already writing a selected member's artifact, the preflight
            codes from :func:`array_preflight`, or ``REMOTE_COMMAND_FAILED``
            when a command failed or ``sbatch`` announced no usable id.
    """
    expression = format_array_argument(indices, spec["throttle"])
    rows = selected_members(spec, indices)

    # One account enumeration for every member, then the same two refusals
    # the single-job path applies: a live job this ledger has no row for
    # under the array's name or a selected member's, then each artifact a
    # recorded live job is writing. Before any upload: a submission that
    # would race a live job should cost nothing but this query.
    entries = ledger.read(ledger_path, cluster)
    account = parse_account_output(remote.run_remote(host, account_command()))
    base = spec["base"]
    label = qualified_name(base["project"], base["name"])
    check_no_unrecorded_live_job(
        entries,
        account,
        names=frozenset([label, *(name for _, name in rows)]),
        ledger_path=str(ledger_path),
    )
    claimed = claimed_artifacts(entries, account)
    members = expand_sweep(spec)
    for index, name in rows:
        check_artifact_is_free(members[index]["artifact"], claimed, name=name)

    admission = array_preflight(
        spec,
        indices,
        host=host,
        script_dir=script_dir,
        log_dir=log_dir,
        cluster=cluster,
        charge_account=charge_account,
        repo=repo,
    )

    output = remote.run_remote(
        host, f"cd {script_dir} && sbatch --array={expression} {label}.sbatch"
    )
    base_id = parse_job_id(output)

    image = base["image"]
    submitted: list[SubmittedMember] = []
    for index, name in rows:
        member = members[index]
        task_id = array_task_id(base_id, index)
        # Each row names what ITS member runs: one comparison serves the
        # array, but members may start different modules.
        member_drift = drift.drift_for_command(
            admission["image_drift"], member["command"], working_directory=script_dir
        )
        ledger.append(
            ledger_path,
            LedgerEntry(
                job_id=task_id,
                project=member["project"],
                name=name,
                host=host,
                partition=member["partition"],
                submitted_at=submitted_at,
                log_dir=log_dir,
                deterministic=member["deterministic"],
                experiment=member["experiment"],
                image_digest="" if image is None else image["sha256"],
                submitter=submitter,
                artifact=member["artifact"],
                image_drift=member_drift,
            ),
        )
        submitted.append(SubmittedMember(name, task_id, member_drift))

    audit.sweep_submitted(
        host=host,
        project=base["project"],
        base_name=base["name"],
        job_ids=[member.job_id for member in submitted],
        partition=base["partition"],
        cluster=cluster,
    )
    return submitted


__all__ = [
    "array_preflight",
    "selected_members",
    "submit_array",
]
