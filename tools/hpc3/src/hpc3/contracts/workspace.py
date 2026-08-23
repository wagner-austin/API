"""The workspace contract: one document every command reads.

Before this existed, each command carried the cluster's whole address in its
flags -- ``--host``, ``--root``, ``--budget``, ``--ledger`` -- and each job
document restated ten resource fields its neighbours already had. Two costs
followed, and only the second is obvious:

* Repetition. Adding a second body of work meant copying ten fields, and
  changing an environment path meant editing every document that named it.
* **Divergence.** Nothing tied ``hpc3-triage --ledger`` to the ledger
  ``hpc3-submit`` had written. Point them at different paths and triage
  reports a clean board while jobs run unwatched, or reports every job as
  ``unaccounted`` while nothing is wrong. Both readings are wrong and neither
  looks wrong.

So the connection, the root, the ledger, the budget and the per-project
defaults are declared once, here, and every command derives them. There is no
flag to override any of it: an override is exactly how the two ledgers drift
apart again.

What is deliberately NOT configurable is the cluster's own limits -- which
partitions exist, what GPUs they carry, how many one user may hold. Those live
in :mod:`hpc3.contracts.cluster` as measured facts. A caller who could declare
``max_gpus_per_user: 999`` would not raise the ceiling; they would only disable
the check that predicts the pending job.
"""

from __future__ import annotations

import pathlib

from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    require_bool,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from hpc3.clusters import require_cluster
from hpc3.contracts.budget import Budget, decode_budget, encode_budget
from hpc3.contracts.cluster import ClusterFacts, require_gpu_type, require_partition
from hpc3.contracts.layout import require_project, require_root

DEFAULT_QUIET_SECONDS = 1800
"""How long a running job may write nothing before triage calls it silent.

Thirty minutes is long enough to cover model download, dataset tokenisation
and a slow first epoch on this cluster, and short enough that a wedged job is
found the same afternoon rather than the next morning.
"""


class ProjectConfig(TypedDict):
    """The resource settings every job in one project starts from.

    Every field is a default, and every one may be overridden by an individual
    run -- but the override goes through the same validation a full spec does,
    so overriding cannot reach a state authoring a spec by hand could not.

    Attributes:
        partition: Partition this project's work goes to.
        gpu: GPU model to pin. Never generic; see
            :mod:`hpc3.contracts.job` rule 1.
        gpu_count: GPUs per job.
        cpus: CPU cores per job.
        mem_gb: Host memory per job, in GiB.
        minutes: Wall-clock limit per job.
        requeue: Whether Slurm should resubmit after a preemption.
        checkpoint_steps: Training steps between checkpoints; 0 means none.
        accept_billing: Standing consent to spend service units. Declared per
            project rather than per job because whether a body of work is
            allowed to cost money is a property of the work, not of one run.
        env_path: Absolute path to this project's Python environment on the
            cluster.
    """

    partition: str
    gpu: str
    gpu_count: int
    cpus: int
    mem_gb: int
    minutes: int
    requeue: bool
    checkpoint_steps: int
    accept_billing: bool
    env_path: str


class Workspace(TypedDict):
    """Everything a command needs that is not specific to one run.

    Attributes:
        cluster: Slug of the measured cluster whose limits every rule is
            checked against. Selects a module from :mod:`hpc3.clusters`; it
            cannot supply limits of its own.
        host: SSH destination. One name, so every command reaches the same
            cluster.
        root: Absolute cluster directory under which every project's scripts
            and logs are derived.
        ledger: Local path to the append-only submission record, already
            resolved against the config file's own directory -- so a workspace
            can be checked in and used from anywhere without absolute paths.
        quiet_seconds: Staleness threshold triage applies to running jobs.
        budget: Self-imposed caps, shared by every project. One pool, because
            the machine is one machine.
        projects: Resource defaults, keyed by project name.
    """

    cluster: str
    host: str
    root: str
    ledger: str
    quiet_seconds: int
    budget: Budget
    projects: dict[str, ProjectConfig]


PROJECT_FIELDS = (
    "partition",
    "gpu",
    "gpu_count",
    "cpus",
    "mem_gb",
    "minutes",
    "requeue",
    "checkpoint_steps",
    "accept_billing",
    "env_path",
)
"""The fields a project declares, and exactly the fields a run may override.

Kept as one tuple so the two uses cannot drift: adding a field to
:class:`ProjectConfig` without adding it here would make it undeclarable, and
adding it here without adding it there would make it unreadable.
"""


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


def _require_positive(obj: dict[str, JSONValue], key: str) -> int:
    """Read a required integer field that must be at least one.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not an integer, or below one.
    """
    value = require_int(obj, key)
    if value < 1:
        raise JSONTypeError(f"Field '{key}' must be at least 1, got {value}")
    return value


def decode_project_config(value: JSONValue, cluster: ClusterFacts) -> ProjectConfig:
    """Decode and validate one project's resource defaults.

    The cross-field submission rules -- billing consent, preemption
    protection, partition-carries-GPU -- are NOT checked here. They are
    checked when a run resolves against these defaults, because a run may
    override any field involved in them, and rejecting a project whose
    defaults only become valid once combined would refuse a legitimate shape.

    Args:
        value: Value produced by the JSON loader.
        cluster: The cluster whose partitions and GPUs the defaults must name.

    Returns:
        Validated defaults.

    Raises:
        JSONTypeError: If the value is not an object, or a field is missing,
            mistyped, empty or non-positive.
        AppError: With ``PARTITION_UNKNOWN`` or ``GPU_TYPE_UNPINNED`` if the
            partition or GPU model is not one this cluster has.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"project config must be a JSON object, got {type(value).__name__}")

    checkpoint_steps = require_int(value, "checkpoint_steps")
    if checkpoint_steps < 0:
        raise JSONTypeError(
            f"Field 'checkpoint_steps' must not be negative, got {checkpoint_steps}"
        )

    return ProjectConfig(
        partition=require_partition(cluster, value, "partition"),
        gpu=require_gpu_type(cluster, value, "gpu"),
        gpu_count=_require_positive(value, "gpu_count"),
        cpus=_require_positive(value, "cpus"),
        mem_gb=_require_positive(value, "mem_gb"),
        minutes=_require_positive(value, "minutes"),
        requeue=require_bool(value, "requeue"),
        checkpoint_steps=checkpoint_steps,
        accept_billing=require_bool(value, "accept_billing"),
        env_path=_require_nonempty_str(value, "env_path"),
    )


def encode_project_config(config: ProjectConfig) -> dict[str, JSONValue]:
    """Encode one project's defaults to a JSON object.

    Args:
        config: Defaults to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "partition": config["partition"],
        "gpu": config["gpu"],
        "gpu_count": config["gpu_count"],
        "cpus": config["cpus"],
        "mem_gb": config["mem_gb"],
        "minutes": config["minutes"],
        "requeue": config["requeue"],
        "checkpoint_steps": config["checkpoint_steps"],
        "accept_billing": config["accept_billing"],
        "env_path": config["env_path"],
    }


def _decode_projects(
    value: dict[str, JSONValue], cluster: ClusterFacts
) -> dict[str, ProjectConfig]:
    """Decode the project table, validating every key as a project name.

    Args:
        value: The workspace object being decoded.
        cluster: The cluster the projects' defaults are checked against.

    Returns:
        Defaults keyed by validated project name.

    Raises:
        JSONTypeError: If ``projects`` is missing, not an object, empty, or a
            key is not a usable project name. An empty table describes a
            workspace that can submit nothing.
    """
    raw = value.get("projects")
    if not isinstance(raw, dict):
        raise JSONTypeError(f"Field 'projects' must be a JSON object, got {type(raw).__name__}")
    if raw == {}:
        raise JSONTypeError("Field 'projects' must declare at least one project")

    decoded: dict[str, ProjectConfig] = {}
    for name, config in raw.items():
        # Validated through the same function the job contract uses, so a name
        # that reaches `squeue` from here is a name that would be accepted
        # there -- a workspace cannot smuggle in one the layout rejects.
        checked = require_project({"project": name}, "project")
        decoded[checked] = decode_project_config(config, cluster)
    return decoded


def decode_workspace(value: JSONValue, *, config_dir: pathlib.Path) -> Workspace:
    """Decode and validate a workspace document.

    Args:
        value: Value produced by the JSON loader.
        config_dir: Directory the document was read from. Relative ledger
            paths resolve against it, so a workspace committed alongside its
            runs works from any working directory and on any machine.

    Returns:
        Validated workspace.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, the project table is empty, or a project name or
            resource default is invalid.
        ValueError: If the root is not an absolute POSIX path.
        AppError: With ``CLUSTER_UNKNOWN`` if no module has been measured for
            the named cluster, or ``PARTITION_UNKNOWN`` / ``GPU_TYPE_UNPINNED``
            if a project names hardware that cluster does not have.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"workspace must be a JSON object, got {type(value).__name__}")

    # Resolved first: every other field is validated against this machine's
    # measured limits, so reading them in any other order would check a
    # project's partition against the wrong cluster.
    cluster = require_cluster(_require_nonempty_str(value, "cluster"))

    quiet_seconds = value.get("quiet_seconds", DEFAULT_QUIET_SECONDS)
    if not isinstance(quiet_seconds, int) or isinstance(quiet_seconds, bool):
        raise JSONTypeError(
            f"Field 'quiet_seconds' must be an integer, got {type(quiet_seconds).__name__}"
        )
    if quiet_seconds < 1:
        # Zero reports every running job as silent, which is the same as
        # reporting nothing: a board of false findings is not read.
        raise JSONTypeError(f"Field 'quiet_seconds' must be at least 1, got {quiet_seconds}")

    return Workspace(
        cluster=cluster["slug"],
        host=_require_nonempty_str(value, "host"),
        root=require_root(_require_nonempty_str(value, "root")),
        ledger=str(config_dir / _require_nonempty_str(value, "ledger")),
        quiet_seconds=quiet_seconds,
        budget=decode_budget(value.get("budget")),
        projects=_decode_projects(value, cluster),
    )


def encode_workspace(workspace: Workspace) -> dict[str, JSONValue]:
    """Encode a workspace to a JSON object.

    The ledger is emitted as resolved, which is not what was read: the
    document may have carried a relative path. Round-tripping therefore
    produces an equivalent workspace, not identical bytes.

    Args:
        workspace: Workspace to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    projects: dict[str, JSONValue] = {
        name: encode_project_config(config) for name, config in workspace["projects"].items()
    }
    return {
        "cluster": workspace["cluster"],
        "host": workspace["host"],
        "root": workspace["root"],
        "ledger": workspace["ledger"],
        "quiet_seconds": workspace["quiet_seconds"],
        "budget": encode_budget(workspace["budget"]),
        "projects": projects,
    }


def workspace_cluster(workspace: Workspace) -> ClusterFacts:
    """Resolve the measured facts for the cluster a workspace selected.

    The workspace stores the slug rather than the facts, so every field of it
    is JSON-encodable and a round trip cannot smuggle in a modified ceiling.
    The lookup is a dictionary read; callers are free to do it per call.

    Args:
        workspace: The decoded workspace.

    Returns:
        That cluster's measured facts.

    Raises:
        AppError: With ``CLUSTER_UNKNOWN`` if the registry no longer has it,
            which can only happen if a cluster module was deleted between
            decoding and use.
    """
    return require_cluster(workspace["cluster"])


def require_project_config(workspace: Workspace, project: str) -> ProjectConfig:
    """Look up one project's defaults.

    Args:
        workspace: The decoded workspace.
        project: Project the run named.

    Returns:
        That project's defaults.

    Raises:
        AppError: With ``WORKSPACE_PROJECT_UNKNOWN`` if the workspace declares
            no such project. The message lists what it does declare, because
            the cause is nearly always a typo or a run document that belongs
            to a different workspace, and both are answered by seeing the
            list.
    """
    config = workspace["projects"].get(project)
    if config is None:
        known = sorted(workspace["projects"])
        raise AppError(
            Hpc3ErrorCode.WORKSPACE_PROJECT_UNKNOWN,
            f"Workspace declares no project {project!r}; it declares {known}. "
            "Add it to the workspace's 'projects' table before submitting to it.",
        )
    return config


__all__ = [
    "DEFAULT_QUIET_SECONDS",
    "PROJECT_FIELDS",
    "ProjectConfig",
    "Workspace",
    "decode_project_config",
    "decode_workspace",
    "encode_project_config",
    "encode_workspace",
    "require_project_config",
    "workspace_cluster",
]
