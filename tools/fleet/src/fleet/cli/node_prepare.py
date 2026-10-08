"""Everything a node runner resolves for a claimed job before its lease.

Moved out of :mod:`fleet.cli.node_agent` when that module, which runs the
claim and the launch, reached the file ceiling as it took on the serving
loop's wiring (:mod:`fleet.cli.node_serve`, MCPs board task 8993c306): what
a job needs before a lease is one concern, checked in one place, and the
claim pass calls it.
"""

from __future__ import annotations

import pathlib

from platform_core.error_codes_fleet import FleetErrorCode
from typing_extensions import TypedDict

from fleet.cli import _config
from fleet.cli import run as run_cli
from fleet.cli.node_ready import Ready
from fleet.contracts.dispatch import DispatchJob
from fleet.contracts.node import NodeConfig
from fleet.contracts.project import ProjectConfig
from fleet.contracts.source import ProjectSource
from fleet.contracts.workspace import require_project
from fleet.core import archive_scope, capacity, export


def tags_refusal(job: DispatchJob, declared: tuple[str, ...]) -> str | None:
    """Whether the job requires a tag the registry does not declare for its project.

    WHAT A JOB NEEDS IS ITS PROJECT'S DECLARATION; the job's own tags only
    route it through the queue. Every claim names the projects its runner
    fits (:func:`fleet.core.capacity.fitting_projects`), each assessed
    against the declaration and the runner's tags, and :func:`prepare`
    assesses it again, so a job that names FEWER tags than its project
    still runs only on a runner that carries all of them, and runs. MCPs
    board task 939ec5c7: MCPs/scripts/ps-harness job 7c16305c, submitted on
    2026-10-03 with ``[windows]`` for a project declaring ``[windows,
    elevated]``, waited from 17:16Z on a node with room, because no runner
    would take it and this rule, then an equality, would have refused it.

    Args:
        job: The claimed job.
        declared: The project's ``required_tags`` in the registry.

    Returns:
        The ``PROJECT_TAGS_MISMATCH`` refusal when the job requires a tag
        the project does not declare, which asks for a node the registry
        never said the project needs; None when every tag it names is
        declared.
    """
    extra = sorted(set(job["required_tags"]) - set(declared))
    if not extra:
        return None
    return (
        f"{FleetErrorCode.PROJECT_TAGS_MISMATCH.value}: the job requires "
        f"[{', '.join(extra)}], which fleet.json does not declare for {job['project']} "
        f"(it declares [{', '.join(declared)}]); resubmit with the registry's tags"
    )


class Admitted(TypedDict):
    """What the claim decides for a job before its launch is handed off.

    Attributes:
        plan: The project's declaration.
        workers: Test workers the capacity check granted on this node, which
            the room gate charges to the node from the claim on
            (:mod:`fleet.cli.node_launch`).
    """

    plan: ProjectConfig
    workers: int


def admit(
    loaded: _config.LoadedWorkspace, job: DispatchJob, *, node: NodeConfig, ready: Ready
) -> Admitted | str:
    """Read a claimed job's registry line, its tags against it, and its grant.

    Only what reads the registry and the probe already taken, so a claim
    decides it in no time and hands the launch off (MCPs board task
    8993c306): the commit's fetch, the companions and the fleet-wide
    resources are the launch's, in :func:`prepare`.

    Args:
        loaded: The workspace and its resolved record paths.
        job: The claimed job.
        node: This node's declaration.
        ready: What it reported when probed this pass, and the tags its
            runner claimed with, the project's fit judged against both.

    Returns:
        The plan and the grant, or the ``PROJECT_TAGS_MISMATCH`` refusal
        (:func:`tags_refusal`) as its ``CODE: message`` line.

    Raises:
        AppError: ``WORKSPACE_PROJECT_UNKNOWN``, or the capacity codes
            :func:`fleet.core.capacity.plan_dispatch` raises; each a local
            refusal the caller reports to the queue verbatim.
    """
    plan = require_project(loaded.workspace, job["project"])
    mismatch = tags_refusal(job, plan["required_tags"])
    if mismatch is not None:
        return mismatch
    workers = capacity.plan_dispatch(node, ready["state"], plan, ready["tags"])
    return Admitted(plan=plan, workers=workers)


class Prepared(TypedDict):
    """Everything a claimed job needs before its lease is taken.

    Attributes:
        plan: The project's declaration.
        source: Its source, present by construction here.
        mirror: The mirror on the hub, holding the commit.
        companions: The bundles of the repositories staged beside the
            export, each at the commit its declared ref names now.
        scope: The pathspec the export's archive is built with, leaving out
            the data directories this repository declares that this project
            does not own (:func:`fleet.core.archive_scope.archive_pathspec`).
            Empty for a repository that declares none.
        workers: Test workers the capacity check granted on this node.
    """

    plan: ProjectConfig
    source: ProjectSource
    mirror: pathlib.Path
    companions: tuple[export.CompanionExport, ...]
    scope: tuple[str, ...]
    workers: int


def prepare(
    loaded: _config.LoadedWorkspace, job: DispatchJob, *, admitted: Admitted, sha: str
) -> Prepared:
    """Resolve, check and fetch everything an admitted job needs before its lease.

    In this order because each step is cheaper than the next and each
    refusal is more the submitter's than the last: the remote, the commit
    on the remote, the companions the project's check reads beside it, and
    the fleet-wide resources it holds. The companions are fetched and
    bundled HERE, with the commit, so a declared ref the remote does not
    serve refuses with no lease held and nothing copied to a node.

    Args:
        loaded: The workspace and its resolved record paths.
        job: The claimed job.
        admitted: Its plan and grant, from :func:`admit`.
        sha: The job's commit.

    The archive's scope is resolved here too, with the rest of what the
    dispatch needs and before the lease: it reads only the registry and the
    project's own path, so a repository whose data declaration contradicts
    its project list has already been refused by the workspace decoder and
    never reaches a node.

    Returns:
        What the dispatch needs.

    Raises:
        AppError: ``PROJECT_REMOTE_MISSING``, ``SHA_NOT_ON_REMOTE``,
            ``INSTALL_PATH_NOT_IN_COMMIT``, ``COMPANION_REF_NOT_ON_REMOTE``,
            ``EXPORT_FAILED``, or ``RESOURCE_HELD`` from
            :func:`fleet.cli.run.require_resources_free`; every one a local
            refusal the caller reports to the queue verbatim.
    """
    plan = admitted["plan"]
    source = export.require_source(job["project"], plan["source"])
    mirror = export.prepare_mirror(loaded.mirrors, remote=source["remote"], sha=sha)
    export.require_install_paths(mirror, sha, source["install"])
    companions = export.export_companions(loaded.mirrors, loaded.archives, source["companions"])
    run_cli.require_resources_free(loaded, plan)
    workers = admitted["workers"]
    return Prepared(
        plan=plan,
        source=source,
        mirror=mirror,
        companions=companions,
        scope=archive_scope.archive_pathspec(
            loaded.workspace["data_paths"], remote=source["remote"], project_path=source["path"]
        ),
        workers=workers,
    )


__all__ = ["Admitted", "Prepared", "admit", "prepare", "tags_refusal"]
