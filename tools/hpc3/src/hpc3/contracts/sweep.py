"""The sweep contract: many jobs from one template, bounded by the QOS.

A scale rung is six jobs -- two corpora crossed with three seeds -- that share
every resource setting and differ only in the payload command. Submitting them
one at a time is not the problem; submitting more of them than the QOS admits
is, because Slurm does not reject the excess. Those jobs sit ``PENDING``
against ``MaxTRESPU`` rather than against a busy cluster, which reads as
contention and is not, and the operator waits on a queue that will not move
until something else of theirs finishes.

So the ceiling is checked here, before submission, against the measured
per-partition limits. The alternative -- submit and see -- produces a half-
running sweep whose remainder is invisible without reading ``squeue``'s reason
column.
"""

from __future__ import annotations

from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

from hpc3.contracts.cluster import ClusterFacts, partition_facts
from hpc3.contracts.job import JobSpec, decode_job_spec, encode_job_spec


class SweepMember(TypedDict):
    """One variation on the sweep's template.

    Attributes:
        suffix: Appended to the template's name to make this job's name.
            Distinct across the sweep, because the name determines the log
            filenames and two jobs sharing one would interleave into the
            same file.
        command: Payload for this member, replacing the template's.
    """

    suffix: str
    command: str


class SweepSpec(TypedDict):
    """A template plus the variations to run from it.

    Attributes:
        base: The shared job settings. Already validated, so every member
            inherits a spec that satisfies all five submission rules.
        members: The variations. Never empty.
    """

    base: JobSpec
    members: list[SweepMember]


def expand_sweep(spec: SweepSpec) -> list[JobSpec]:
    """Build one job spec per member.

    Args:
        spec: The validated sweep.

    Returns:
        One spec per member, in declaration order, each carrying the
        template's resources with the member's name and command.
    """
    return [
        JobSpec(
            project=spec["base"]["project"],
            name=f"{spec['base']['name']}-{member['suffix']}",
            partition=spec["base"]["partition"],
            gpu=spec["base"]["gpu"],
            gpu_count=spec["base"]["gpu_count"],
            cpus=spec["base"]["cpus"],
            mem_gb=spec["base"]["mem_gb"],
            minutes=spec["base"]["minutes"],
            requeue=spec["base"]["requeue"],
            checkpoint_steps=spec["base"]["checkpoint_steps"],
            accept_billing=spec["base"]["accept_billing"],
            env_path=spec["base"]["env_path"],
            pinned_packages=spec["base"]["pinned_packages"],
            # The template's identity plus the member's own suffix: six arms
            # sharing one experiment record would be six rows the ledger
            # cannot tell apart, which is the failure this field exists for.
            experiment={**spec["base"]["experiment"], "member": member["suffix"]},
            command=member["command"],
        )
        for member in spec["members"]
    ]


def _require_nonempty_str(obj: dict[str, JSONValue], key: str) -> str:
    """Read a required string field that must not be empty.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not a string, or empty.
    """
    value = require_str(obj, key)
    if value == "":
        raise JSONTypeError(f"Field '{key}' must not be empty")
    return value


def decode_sweep_member(value: JSONValue) -> SweepMember:
    """Decode and validate a JSON value into one sweep member.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        Validated member.

    Raises:
        JSONTypeError: If the value is not an object, or a field is missing,
            mistyped, empty, or -- for the suffix -- carries a character that
            would leave the job name unusable as a filename.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"sweep member must be a JSON object, got {type(value).__name__}")
    suffix = _require_nonempty_str(value, "suffix")
    if "/" in suffix or "\\" in suffix:
        raise JSONTypeError(f"Field 'suffix' must not contain a path separator, got {suffix!r}")
    return SweepMember(suffix=suffix, command=_require_nonempty_str(value, "command"))


def _check_ceilings(cluster: ClusterFacts, base: JobSpec, count: int) -> None:
    """Reject a sweep larger than the partition's per-user QOS admits.

    Args:
        cluster: The cluster whose measured QOS ceilings apply.
        base: The template, carrying the partition and per-job GPU count.
        count: Number of members.

    Raises:
        AppError: With
            :attr:`~platform_core.errors.Hpc3ErrorCode.SWEEP_EXCEEDS_GPU_CEILING`
            if the members together ask for more GPUs than one user may hold,
            or
            :attr:`~platform_core.errors.Hpc3ErrorCode.SWEEP_EXCEEDS_JOB_CEILING`
            if there are more members than concurrently-runnable jobs. Slurm
            queues the excess rather than refusing it, so the operator would
            otherwise wait on a limit that looks like contention.
    """
    facts = partition_facts(cluster, base["partition"])

    gpus = count * base["gpu_count"]
    gpu_ceiling = facts["max_gpus_per_user"]
    if gpus > gpu_ceiling:
        raise AppError(
            Hpc3ErrorCode.SWEEP_EXCEEDS_GPU_CEILING,
            f"{count} members x {base['gpu_count']} GPU(s) = {gpus}, but "
            f"{base['partition']!r} on {cluster['slug']!r} allows one user "
            f"{gpu_ceiling} at once. "
            "The excess would pend against the QOS, not against the cluster.",
        )

    job_ceiling = facts["max_jobs_per_user"]
    if count > job_ceiling:
        raise AppError(
            Hpc3ErrorCode.SWEEP_EXCEEDS_JOB_CEILING,
            f"{count} members, but {base['partition']!r} on {cluster['slug']!r} "
            f"runs at most {job_ceiling} of one user's jobs at once.",
        )


def encode_sweep_spec(spec: SweepSpec) -> dict[str, JSONValue]:
    """Encode a sweep spec to a JSON object.

    Args:
        spec: Spec to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    members: list[JSONValue] = [
        {"suffix": member["suffix"], "command": member["command"]} for member in spec["members"]
    ]
    return {"base": encode_job_spec(spec["base"]), "members": members}


def decode_sweep_spec(value: JSONValue, cluster: ClusterFacts) -> SweepSpec:
    """Decode and validate a JSON value into a sweep spec.

    Args:
        value: Value produced by the JSON loader.
        cluster: The cluster whose measured limits apply.

    Returns:
        A sweep whose template satisfies every submission rule and whose size
        fits the partition's per-user ceilings.

    Raises:
        JSONTypeError: If the value is not an object, the member list is
            missing or empty, a member is invalid, or two members share a
            suffix -- which would point two jobs at one log file.
        AppError: If the template breaks a submission rule, or the sweep is
            larger than the QOS admits.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"sweep spec must be a JSON object, got {type(value).__name__}")

    base = decode_job_spec(value.get("base"), cluster)

    raw = require_list(value, "members")
    if raw == []:
        raise JSONTypeError("Field 'members' must not be empty")
    members = [decode_sweep_member(item) for item in raw]

    suffixes = [member["suffix"] for member in members]
    if len(set(suffixes)) != len(suffixes):
        raise JSONTypeError(f"Field 'members' must not repeat a suffix, got {suffixes}")

    _check_ceilings(cluster, base, len(members))
    return SweepSpec(base=base, members=members)


__all__ = [
    "SweepMember",
    "SweepSpec",
    "decode_sweep_member",
    "decode_sweep_spec",
    "encode_sweep_spec",
    "expand_sweep",
]
