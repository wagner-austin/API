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
from platform_core.json_utils import JSONTypeError, JSONValue
from typing_extensions import TypedDict

from hpc3.clusters import require_cluster
from hpc3.contracts.cluster import ClusterFacts
from hpc3.contracts.fields import require_nonempty_str
from hpc3.contracts.layout import require_project, require_root
from hpc3.contracts.project import ProjectConfig, decode_project_config, encode_project_config

DEFAULT_QUIET_SECONDS = 1800
"""How long a running job may write nothing before triage calls it silent.

Thirty minutes is long enough to cover model download, dataset tokenisation
and a slow first epoch on this cluster, and short enough that a wedged job is
found the same afternoon rather than the next morning.
"""


class WorkspaceConnection(TypedDict):
    """Where the cluster is, without the registry of what runs on it.

    THE SPLIT EXISTS BECAUSE ONBOARDING HAPPENS BEFORE REGISTRATION. Every
    registered project declares an image
    (:func:`_require_project_image`), and the digest in that declaration is
    produced by a build, which is driven by a spec, which is written by
    ``hpc3-image-capture`` probing an environment over SSH. Capture therefore
    needs the HOST -- and nothing else from this document -- while the project
    it is onboarding is by definition not yet in ``projects``.

    Decoding the whole workspace to reach ``host`` made that impossible: the
    project table is validated eagerly, so ONE unimaged project anywhere in
    the file refused the read for every caller, including the command whose
    job is to produce the image that would fix it. That is the deadlock this
    type removes, and it removes it without an exemption anybody can declare:
    the registry's rule is untouched, and a caller that does not need the
    registry no longer decodes it.

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
    """

    cluster: str
    host: str
    root: str
    ledger: str
    quiet_seconds: int


class Workspace(WorkspaceConnection):
    """A connection plus the registry of what runs over it.

    Attributes:
        projects: Resource defaults and caps, keyed by project name. The
            caps live there rather than here; see
            :attr:`ProjectConfig.budget` for what forking this document
            three ways cost before they did.
    """

    projects: dict[str, ProjectConfig]


def _require_projects_table(value: dict[str, JSONValue]) -> dict[str, JSONValue]:
    """Read the project table without decoding what is in it.

    Args:
        value: The workspace object being decoded.

    Returns:
        The raw table.

    Raises:
        JSONTypeError: If ``projects`` is missing, not an object, or empty. An
            empty table describes a workspace that can submit nothing.
    """
    raw = value.get("projects")
    if not isinstance(raw, dict):
        raise JSONTypeError(f"Field 'projects' must be a JSON object, got {type(raw).__name__}")
    if raw == {}:
        raise JSONTypeError("Field 'projects' must declare at least one project")
    return raw


def _decode_projects(
    value: dict[str, JSONValue], cluster: ClusterFacts, config_dir: pathlib.Path
) -> dict[str, ProjectConfig]:
    """Decode the project table, validating every key as a project name.

    Args:
        value: The workspace object being decoded.
        cluster: The cluster the projects' defaults are checked against.
        config_dir: Directory the document was read from, which each
            project's ``repo`` resolves against.

    Returns:
        Defaults keyed by validated project name.

    Raises:
        JSONTypeError: If ``projects`` is missing, not an object, empty, or a
            key is not a usable project name. An empty table describes a
            workspace that can submit nothing.
    """
    raw = _require_projects_table(value)
    return {
        require_project({"project": name}, "project"): decode_project_config(
            config, cluster, config_dir=config_dir
        )
        for name, config in raw.items()
    }


def _require_workspace_object(value: JSONValue) -> dict[str, JSONValue]:
    """Narrow a loaded value to a workspace object.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The object.

    Raises:
        JSONTypeError: If the value is not an object.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"workspace must be a JSON object, got {type(value).__name__}")
    return value


def decode_workspace_connection(
    value: JSONValue, *, config_dir: pathlib.Path
) -> WorkspaceConnection:
    """Decode where the cluster is, WITHOUT decoding the project registry.

    For callers that need the host and nothing else. It does not read
    ``projects`` at all, so a document whose registry is mid-change -- or that
    declares a project this command is about to make valid -- can still be
    read for its connection. See :class:`WorkspaceConnection` for why that
    matters and why it is not an exemption.

    Args:
        value: Value produced by the JSON loader.
        config_dir: Directory the document was read from. Relative ledger
            paths resolve against it.

    Returns:
        The validated connection.

    Raises:
        JSONTypeError: If the value is not an object, or a connection field is
            missing, mistyped, empty or non-positive.
        ValueError: If the root is not an absolute POSIX path.
        AppError: With ``CLUSTER_UNKNOWN`` if no module has been measured for
            the named cluster.
    """
    obj = _require_workspace_object(value)

    # Resolved first: every other field is validated against this machine's
    # measured limits, so reading them in any other order would check a
    # project's partition against the wrong cluster.
    cluster = require_cluster(require_nonempty_str(obj, "cluster"))

    quiet_seconds = obj.get("quiet_seconds", DEFAULT_QUIET_SECONDS)
    if not isinstance(quiet_seconds, int) or isinstance(quiet_seconds, bool):
        raise JSONTypeError(
            f"Field 'quiet_seconds' must be an integer, got {type(quiet_seconds).__name__}"
        )
    if quiet_seconds < 1:
        # Zero reports every running job as silent, which is the same as
        # reporting nothing: a board of false findings is not read.
        raise JSONTypeError(f"Field 'quiet_seconds' must be at least 1, got {quiet_seconds}")

    return WorkspaceConnection(
        cluster=cluster["slug"],
        host=require_nonempty_str(obj, "host"),
        root=require_root(require_nonempty_str(obj, "root")),
        ledger=str(config_dir / require_nonempty_str(obj, "ledger")),
        quiet_seconds=quiet_seconds,
    )


def decode_workspace(value: JSONValue, *, config_dir: pathlib.Path) -> Workspace:
    """Decode and validate a workspace document, registry included.

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
            the named cluster, ``PARTITION_UNKNOWN`` / ``GPU_TYPE_UNPINNED``
            if a project names hardware that cluster does not have, or
            ``PROJECT_UNIMAGED`` if a project declares no image.
    """
    connection = decode_workspace_connection(value, config_dir=config_dir)
    return Workspace(
        cluster=connection["cluster"],
        host=connection["host"],
        root=connection["root"],
        ledger=connection["ledger"],
        quiet_seconds=connection["quiet_seconds"],
        projects=_decode_projects(
            _require_workspace_object(value), require_cluster(connection["cluster"]), config_dir
        ),
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
    "Workspace",
    "WorkspaceConnection",
    "decode_workspace",
    "decode_workspace_connection",
    "encode_workspace",
    "require_project_config",
    "workspace_cluster",
]
