"""CLI: one tick of a node's runner on the queue's node lane.

Usage:
    fleet-node-agent --config fleet.json --node sedona
    fleet-node-agent --config fleet.json --node sedona --announce

ONE RUNNER PER ENABLED NODE, ALL OF THEM ON THE HUB (MCPs board task
fd5cabfa, A1 and A5). Until this command the queue had one runner, the hub's
``fleet-agent``, claiming the oldest job once per tick whatever it was, so
nine nodes behaved as one slow node and a revive waited behind a queue of
kills. Now each enabled node in ``fleet.json`` has a scheduled task of its own
(``scripts/register-node-agents.ps1``) running this command with ``--node``
set to it, claiming from the queue's NODE lane the jobs that name it or name
no node, and only those whose required tags the node carries
(:func:`fleet.contracts.tags.node_tags`). The hub's ``fleet-agent`` keeps the
HUB lane, so the two never hold each other's work. The runners live on the
hub rather than on the nodes because the hub holds the git credentials and
the ssh keys and the tailnet policy lets nothing else reach it.

THE NODE IS PROBED BEFORE ANYTHING IS CLAIMED. A tick asks its node what it
has free first, and a node that does not answer or has room for nothing
(:func:`fleet.core.capacity.room_for_any`) claims nothing, so the job stays
in the lane for a node that can run it; the first tick of these runners
(2026-09-21T10:00:02Z) had a sleeping node take the oldest job and refuse
it while a live one found the lane empty. It then asks the node's build
toolchain (:func:`fleet.core.toolchain.readiness_gap`, MCPs board task
bad56f65), and a node missing a tool or on the wrong Python claims nothing
either, naming the tool and the command that would install it there.

WHAT A CLAIMED JOB BECOMES. The job names a project and a commit. The runner
re-checks the job's tags against the registry's declaration for the project,
fetches the commit from the project's declared remote into a bare mirror
(:mod:`fleet.core.export`) BEFORE any lease is taken, so a sha the remote has
never seen is refused with nothing held, judges the project's fit on the
probe already taken, then takes the project's lease on this node, stages
``git archive`` of the commit through the same verified transport every
dispatch uses, sends the build script with the project's install steps and
the node's caches, launches it detached, and reports the run started. A
later tick collects it (:mod:`fleet.cli.node_collect`): reads the result,
reads the tail of the transcript, composes the verdict
(:mod:`fleet.core.verdict`), posts it to the job's task thread or the
submitter's feed, and closes the job on both sides; or stops the build, when
it runs past its lease or its queue job was cancelled under it.

THE IDENTITY IS DERIVED, NOT CONFIGURED. The label is ``fleet-node-<alias>``
and the session id is the version-5 UUID of that label, so every tick of one
node's runner is the same session on the board's ledger and ``held_by`` sees
its own claims across ticks; a fresh UUID per tick would make every tick a
stranger to the last. ``--announce`` posts the check-in that registers that
session on the ledger (MCPs mig 530), once, from the registration script.

Exits 0 whenever the agent itself worked, refused jobs and failed suites
included, for the reason :mod:`fleet.cli.agent` gives: the status is whether
THE AGENT worked, and a loop that stopped on a red build would stop on the
one condition it exists to keep reporting.
"""

from __future__ import annotations

import pathlib
import sys
import uuid
from collections.abc import Sequence

from board_watch import config as board_config
from platform_core import cli_args
from platform_core.error_codes_tooling import FleetErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import JSONObject
from platform_core.logging import LogFormat, LogLevel, get_logger, setup_logging
from platform_core.mcp_client import McpCredentials
from typing_extensions import TypedDict

from fleet.cli import _config
from fleet.cli import run as run_cli
from fleet.cli.node_collect import CLAIM_LEASE_SECONDS, collect_pass, require_sha
from fleet.contracts.dispatch import DispatchJob, encode_job_line
from fleet.contracts.ledger import LedgerEntry
from fleet.contracts.node import NodeConfig, NodeState
from fleet.contracts.project import ProjectConfig
from fleet.contracts.source import ProjectSource
from fleet.contracts.tags import node_tags
from fleet.contracts.workspace import require_node, require_project
from fleet.core import (
    _test_hooks,
    archive_scope,
    capacity,
    dispatch,
    export,
    probe,
    queue,
    records,
    run_lease,
    toolchain,
)

_log = get_logger(__name__)

NODE_FLAG = "--node"
ANNOUNCE_FLAG = "--announce"

_FLAGS = (_config.CONFIG_FLAG, NODE_FLAG)

#: The namespace the runner's session UUID is derived in.
IDENTITY_NAMESPACE = uuid.NAMESPACE_URL


def node_identity(alias: str) -> tuple[str, str]:
    """The label and session id a node's runner acts as.

    Args:
        alias: The node's workspace name.

    Returns:
        ``fleet-node-<alias>`` and the version-5 UUID of
        ``fleet-node-agent/<alias>`` in :data:`IDENTITY_NAMESPACE`, lowercase
        hyphenated as the board wants it.
    """
    return f"fleet-node-{alias}", str(uuid.uuid5(IDENTITY_NAMESPACE, f"fleet-node-agent/{alias}"))


def refuse(
    credentials: McpCredentials, job: DispatchJob, identity: JSONObject, *, detail: str
) -> DispatchJob:
    """Close a job as refused, with its named reason, and log it.

    Args:
        credentials: The queue's endpoint and headers.
        job: The claimed job.
        identity: This runner's identity arguments.
        detail: The ``CODE: message`` refusal for the queue.

    Returns:
        The job, for the caller to hand back.

    Raises:
        AppError: Only from the queue call itself.
    """
    queue.report_close(
        credentials,
        job_id=job["job_id"],
        status="refused",
        exit_code=None,
        detail=detail,
        identity=identity,
    )
    _log.info("refused %s: %s", job["job_id"], detail)
    return job


def tags_refusal(job: DispatchJob, declared: tuple[str, ...]) -> str | None:
    """Whether the job's tags disagree with the registry's for its project.

    Args:
        job: The claimed job.
        declared: The project's ``required_tags`` in the registry.

    Returns:
        The ``PROJECT_TAGS_MISMATCH`` refusal, or None when they agree as
        sets. Refused rather than run under either: the queue matched the
        job to this node by the job's tags, and a job that named fewer than
        the project needs would have landed on a node the project refuses.
    """
    if set(job["required_tags"]) == set(declared):
        return None
    return (
        f"{FleetErrorCode.PROJECT_TAGS_MISMATCH.value}: the job requires "
        f"[{', '.join(job['required_tags'])}] but fleet.json declares "
        f"[{', '.join(declared)}] for {job['project']}; resubmit with the registry's tags"
    )


def claim_pass(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    identity: JSONObject,
    *,
    alias: str,
    node: NodeConfig,
) -> DispatchJob | None:
    """Take one job for this node and launch it, or report why it could not.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments.
        alias: This node's workspace name.
        node: Its declaration.

    THE NODE IS ASKED BEFORE THE QUEUE IS. A runner that claimed first and
    probed second took the oldest job off the lane and refused it while a
    live node beside it found nothing (:func:`fleet.core.capacity.room_for_any`
    carries the measurement), so a node that does not answer, or has room
    for nothing, claims nothing this tick and the job stays for one that
    can run it. The same holds for its TOOLCHAIN, asked second
    (:func:`fleet.core.toolchain.attempt_toolchain`, over the ssh account
    whose SID the build's scheduled task is registered for): lavender
    claimed slime jobs d515d038 and 7235d4c4, staged a whole export and ran
    npm ci before dying at the Makefile's first python call on the Store
    alias stub (MCPs board task e62c8120), so a node missing a tool now logs
    the code, the tool and this node's install command and claims nothing.

    Returns:
        The job that was claimed, whatever became of it, or None when this
        node could take nothing or nothing in the lane matched it.

    Raises:
        AppError: Only from the queue calls themselves. A LOCAL refusal (an
            unknown project, tags that disagree, a project with no remote,
            a sha the remote lacks, too little capacity for the project, a
            held lease) is reported to the queue as ``refused`` with its
            code and message verbatim and does not propagate: transport,
            not recovery.
    """
    probed = probe.attempt_probe(node, live_runs=records.live_runs(loaded.ledger, node=alias))
    state = probed["state"]
    if state is None:
        _log.info("%s did not answer; claiming nothing: %s", alias, probed["reason"])
        return None
    full = capacity.room_for_any(node, state)
    if full is not None:
        _log.info("%s has room for nothing; claiming nothing: %s", alias, full)
        return None
    answered = toolchain.attempt_toolchain(node)
    if not isinstance(answered, tuple):
        _log.info(
            "%s did not answer the toolchain probe; claiming nothing: %s: %s",
            alias,
            answered["code"],
            answered["message"],
        )
        return None
    gap = toolchain.readiness_gap(alias, node, answered)
    if gap is not None:
        _log.info("%s cannot build; claiming nothing: %s: %s", alias, gap.code, gap.message)
        return None
    _log.info("%s toolchain ready: %s", alias, toolchain.ready_summary(answered))
    job = queue.claim_next(
        credentials,
        lane="node",
        tags=tuple(sorted(node_tags(node))),
        node=alias,
        lease_seconds=CLAIM_LEASE_SECONDS,
        identity=identity,
    )
    if job is None:
        return None
    _log.info("claimed %s", encode_job_line(job))
    sha = require_sha(job)
    try:
        prepared = prepare(loaded, job, node=node, state=state, sha=sha)
    except AppError as refusal:
        return refuse(credentials, job, identity, detail=f"{refusal.code}: {refusal.message}")
    if isinstance(prepared, str):
        return refuse(credentials, job, identity, detail=prepared)

    def build(run_id: str) -> dispatch.Payload:
        data = export.archive_commit(
            prepared["mirror"], sha, loaded.archives / f"{run_id}-{alias}.tgz", prepared["scope"]
        )
        return dispatch.Payload(data=data, description=f"git archive of {sha}")

    row = launch_claimed(loaded, job, alias=alias, node=node, prepared=prepared, build=build)
    if isinstance(row, str):
        return refuse(credentials, job, identity, detail=row)
    queue.report_start(
        credentials,
        job_id=job["job_id"],
        node=alias,
        run_id=row["run_id"],
        lease_seconds=CLAIM_LEASE_SECONDS,
        identity=identity,
    )
    _log.info("started %s on %s as %s at %s", job["job_id"], alias, row["run_id"], sha)
    return job


def launch_claimed(
    loaded: _config.LoadedWorkspace,
    job: DispatchJob,
    *,
    alias: str,
    node: NodeConfig,
    prepared: Prepared,
    build: dispatch.PayloadBuilder,
) -> LedgerEntry | str:
    """Take the project's lease on this node and launch the job under it.

    GUARDED LIKE ``prepare``, because a fault here is otherwise reported
    NOWHERE: there is no failed feed row on this path, so an AppError out of
    staging ended the tick and left the queue row in status claimed until its
    lease ran out, an hour later, with no reason recorded anywhere a waiting
    session could read. Measured 2026-09-24 (board task 1e57ebe5): two jobs
    sat exactly that way for 32 and 14 minutes.

    A FAILURE AFTER THE LEASE GIVES THE LEASE BACK before the refusal, or the
    resubmission the refusal invites bounces off it: MCPs board task
    e12affc5, where job 7143a25e was refused ``LEASE_HELD`` by the dead run
    of job 4d8b28a9 with 793 s of its lease left. A ``LEASE_HELD`` out of the
    lease itself gives nothing back, because that lease is another run's
    (:mod:`fleet.core.run_lease`).

    Args:
        loaded: The workspace and its resolved record paths.
        job: The claimed job.
        alias: This node's workspace name.
        node: Its declaration.
        prepared: What :func:`prepare` resolved for the job.
        build: Builds the job's export once the lease is held.

    Returns:
        The running ledger row, or the ``CODE: message`` refusal to report.
    """
    try:
        lease = run_lease.take(
            loaded.leases,
            loaded.feed,
            node_name=alias,
            project=job["project"],
            plan=prepared["plan"],
            workers=prepared["workers"],
            agent=job["submitted_by"],
            session_id=job["session_id"],
        )
    except AppError as refusal:
        return f"{refusal.code}: {refusal.message}"
    try:
        return dispatch.launch(
            loaded.ledger,
            loaded.feed,
            lease=lease,
            node=node,
            workers=prepared["workers"],
            build_payload=build,
            companions=prepared["companions"],
            recipe=dispatch.Recipe(
                path=prepared["source"]["path"], install=prepared["source"]["install"]
            ),
        )
    except AppError as refusal:
        detail = f"{refusal.code}: {refusal.message}"
        given_back = run_lease.abandon(loaded.leases, loaded.feed, lease=lease, detail=detail)
        _log.info("%s never launched; %s", lease["run_id"], given_back)
        return detail


class Prepared(TypedDict):
    """Everything a claimed job needs before its lease is taken.

    Attributes:
        plan: The project's declaration.
        source: Its source, present by construction here.
        mirror: The mirror on the hub, holding the commit.
        companions: The archives of the repositories staged beside the
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
    loaded: _config.LoadedWorkspace,
    job: DispatchJob,
    *,
    node: NodeConfig,
    state: NodeState,
    sha: str,
) -> Prepared | str:
    """Resolve, check and fetch everything a job needs before its lease.

    In this order because each step is cheaper than the next and each
    refusal is more the submitter's than the last: the registry line, the
    tags against it, the remote, the commit on the remote, the companions
    the project's check reads beside it, and only then the project's fit on
    this node, judged on the probe the claim pass already took, so no second
    ssh is paid. The companions are fetched and archived HERE, with the
    commit, so a declared ref the remote does not serve refuses with no
    lease held and nothing copied to a node.

    Args:
        loaded: The workspace and its resolved record paths.
        job: The claimed job.
        node: This node's declaration.
        state: What it reported when probed this tick.
        sha: The job's commit.

    The archive's scope is resolved here too, with the rest of what the
    dispatch needs and before the lease: it reads only the registry and the
    project's own path, so a repository whose data declaration contradicts
    its project list has already been refused by the workspace decoder and
    never reaches a node.

    Returns:
        What the dispatch needs, or the ``PROJECT_TAGS_MISMATCH`` refusal
        as its ``CODE: message`` line.

    Raises:
        AppError: ``WORKSPACE_PROJECT_UNKNOWN``, ``PROJECT_REMOTE_MISSING``,
            ``SHA_NOT_ON_REMOTE``, ``COMPANION_REF_NOT_ON_REMOTE``,
            ``EXPORT_FAILED``, ``RESOURCE_HELD`` from
            :func:`fleet.cli.run.require_resources_free`, or the capacity
            codes :func:`fleet.core.capacity.plan_dispatch` raises; every
            one a local refusal the caller reports to the queue verbatim.
    """
    plan = require_project(loaded.workspace, job["project"])
    mismatch = tags_refusal(job, plan["required_tags"])
    if mismatch is not None:
        return mismatch
    source = export.require_source(job["project"], plan["source"])
    mirror = export.prepare_mirror(
        loaded.mirrors, project=job["project"], remote=source["remote"], sha=sha
    )
    companions = export.export_companions(loaded.mirrors, loaded.archives, source["companions"])
    run_cli.require_resources_free(loaded, plan)
    workers = capacity.plan_dispatch(node, state, plan)
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run one tick for one node: collect what finished, claim at most one job.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 whenever the agent itself worked. See the module docstring.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or
            a required one is absent.
        AppError: When the queue or the board cannot be reached or answered
            a shape this runner cannot read, when the node is not declared,
            or when this machine's records and the fleet disagree about a
            run.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    announce = ANNOUNCE_FLAG in tokens
    remaining = [token for token in tokens if token != ANNOUNCE_FLAG]
    parsed = cli_args.parse_single_flags(remaining, _FLAGS)
    loaded = _config.load_workspace(parsed)
    alias = cli_args.require_flag(parsed, NODE_FLAG)
    node = require_node(loaded.workspace, alias)
    agent, session_id = node_identity(alias)
    identity = queue.identity_arguments(agent, session_id, str(loaded.directory))
    board = board_config.load_credentials()
    if announce:
        _log.info(
            "%s",
            queue.announce(
                board,
                machine=f"{sys.platform}:{_test_hooks.hostname()}",
                body=(
                    f"fleet-node-agent for {alias}: claims the queue's node lane for jobs "
                    f"naming {alias} or no node, carrying {', '.join(sorted(node_tags(node)))}"
                ),
                identity=identity,
            ),
        )
        return 0
    credentials = queue.load_credentials()
    collect_pass(loaded, credentials, board, identity, agent=agent, alias=alias)
    if claim_pass(loaded, credentials, identity, alias=alias, node=node) is None:
        _log.info("nothing in the node lane for %s", alias)
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level=LogLevel.INFO,
        format_mode=LogFormat.TEXT,
        service_name="fleet-node-agent",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


# Without this, `python -m fleet.cli.node_agent` imports the module, runs
# nothing and exits 0, which reads as a tick that found nothing to do.
if __name__ == "__main__":
    entrypoint()


__all__ = [
    "ANNOUNCE_FLAG",
    "IDENTITY_NAMESPACE",
    "NODE_FLAG",
    "Prepared",
    "claim_pass",
    "entrypoint",
    "main",
    "node_identity",
    "prepare",
    "refuse",
    "tags_refusal",
]
