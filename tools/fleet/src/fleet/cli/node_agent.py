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
it while a live one found the lane empty.

WHAT A CLAIMED JOB BECOMES. The job names a project and a commit. The runner
re-checks the job's tags against the registry's declaration for the project,
fetches the commit from the project's declared remote into a bare mirror
(:mod:`fleet.core.export`) BEFORE any lease is taken, so a sha the remote has
never seen is refused with nothing held, judges the project's fit on the
probe already taken, then takes the project's lease on this node, stages
``git archive`` of the commit through the same verified transport every
dispatch uses, sends the build script with the project's install steps and
the node's caches, launches it detached, and reports the run started. A
later tick collects it: reads the result, reads the tail of the transcript,
composes the verdict (:mod:`fleet.core.verdict`), posts it to the job's
task thread or the submitter's feed, and closes the job on both sides.

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
from platform_core.logging import get_logger, setup_logging
from platform_core.mcp_client import McpCredentials
from typing_extensions import TypedDict

from fleet.cli import _config
from fleet.cli import collect as collect_cli
from fleet.cli import run as run_cli
from fleet.contracts.dispatch import DispatchJob, encode_job_line
from fleet.contracts.ledger import LedgerEntry
from fleet.contracts.node import NodeConfig, NodeState
from fleet.contracts.project import ProjectConfig
from fleet.contracts.source import ProjectSource
from fleet.contracts.tags import node_tags
from fleet.contracts.workspace import require_node, require_project
from fleet.core import (
    _test_hooks,
    capacity,
    collect,
    dialect,
    dispatch,
    export,
    names,
    probe,
    queue,
    records,
    remote,
    verdict,
)

_log = get_logger(__name__)

NODE_FLAG = "--node"
ANNOUNCE_FLAG = "--announce"

_FLAGS = (_config.CONFIG_FLAG, NODE_FLAG)

#: How long a claim survives without a report: the same hour the hub runner
#: takes, covering the fetch, the staging and the wait until the next tick's
#: collect renews it (:func:`collect_pass` renews every running job it holds,
#: so the lease is never sized for the slowest suite in advance).
CLAIM_LEASE_SECONDS = 3600

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


def collect_one_job(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    board: McpCredentials,
    job: DispatchJob,
    identity: JSONObject,
) -> str:
    """Close one running job out, on both sides, if its node has finished.

    Lifted from ``fleet.cli.agent`` when the node lane took over the make
    targets; the verdict post is what this lane adds to it.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        board: The board's endpoint and headers, for the verdict.
        job: The queue job this runner holds.
        identity: This runner's identity arguments.

    Returns:
        One line saying what happened, for the log.

    Raises:
        AppError: With a node or workspace code when the node cannot be
            reached or its declaration has gone, ``LEASE_NOT_HELD`` when the
            build was still writing after its lease lapsed, or
            ``QUEUE_ANSWER_MALFORMED`` when a node-lane job carries no sha,
            which the queue's pin makes impossible. Not caught: those mean
            this machine's own records and the fleet disagree.
    """
    row: LedgerEntry | None = None
    for candidate in collect_cli.live_rows(loaded, run_id=job["run_id"]):
        row = candidate
    if row is None:
        return f"{encode_job_line(job)}: no live run on this machine, leaving it"
    node = require_node(loaded.workspace, row["node"])
    result = collect.poll_result(node, run_id=row["run_id"])
    if result is None:
        queue.report_progress(
            credentials,
            job_id=job["job_id"],
            note=f"still running on {row['node']} as {row['run_id']}",
            lease_seconds=CLAIM_LEASE_SECONDS,
            identity=identity,
        )
        return f"{encode_job_line(job)}: still running, lease renewed"

    plan = require_project(loaded.workspace, row["project"])
    if collect.outlived_its_lease(row, plan, finished_unix=result["finished_unix"]):
        raise collect_cli.lapsed_lease_refusal(row, plan, finished_unix=result["finished_unix"])

    sha = require_sha(job)
    exit_code = result["exit_code"]
    detail = collect.describe(node, run_id=row["run_id"], exit_code=exit_code)
    target = f"{node['stage_root']}/{row['run_id']}"
    spoken = dialect.for_platform(node["platform"])
    tail = remote.run_script(
        node["host"],
        spoken.script_path(target, names.LOG_TAIL_STEM),
        spoken.log_tail_script(target, verdict.LOG_TAIL_LINES),
        platform=node["platform"],
    )
    judged = verdict.judge(
        job_id=job["job_id"],
        project=row["project"],
        sha=sha,
        node=row["node"],
        exit_code=exit_code,
        tail=tail,
        log_path=names.log_path(target),
        run_id=row["run_id"],
    )
    line = verdict.render_verdict(judged)
    queue.post_verdict(
        board,
        task_id=job["task_id"],
        submitted_by=job["submitted_by"],
        line=line,
        identity=identity,
    )
    dispatch.finish(
        loaded.leases,
        loaded.ledger,
        loaded.feed,
        row=row,
        outcome=collect.outcome_for(exit_code),
        exit_code=exit_code,
        detail=detail,
    )
    queue.report_close(
        credentials,
        job_id=job["job_id"],
        status="passed" if exit_code == 0 else "failed",
        exit_code=exit_code,
        detail=line,
        identity=identity,
    )
    return f"{encode_job_line(job)}: {line}"


def collect_pass(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    board: McpCredentials,
    identity: JSONObject,
    *,
    agent: str,
) -> None:
    """Close out every finished job this runner is holding; renew the rest.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        board: The board's endpoint and headers.
        identity: This runner's identity arguments.
        agent: This runner's label.

    Raises:
        AppError: As :func:`collect_one_job` describes.
    """
    for job in queue.held_by(credentials, agent=agent):
        if job["status"] != "running":
            continue
        _log.info("%s", collect_one_job(loaded, credentials, board, job, identity))


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
    can run it.

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
            prepared["mirror"], sha, loaded.archives / f"{run_id}-{alias}.tgz"
        )
        return dispatch.Payload(data=data, description=f"git archive of {sha}")

    row = dispatch.start(
        loaded.leases,
        loaded.ledger,
        loaded.feed,
        node_name=alias,
        node=node,
        project=job["project"],
        plan=prepared["plan"],
        workers=prepared["workers"],
        agent=job["submitted_by"],
        session_id=job["session_id"],
        build_payload=build,
        recipe=dispatch.Recipe(
            path=prepared["source"]["path"], install=prepared["source"]["install"]
        ),
    )
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


class Prepared(TypedDict):
    """Everything a claimed job needs before its lease is taken.

    Attributes:
        plan: The project's declaration.
        source: Its source, present by construction here.
        mirror: The mirror on the hub, holding the commit.
        workers: Test workers the capacity check granted on this node.
    """

    plan: ProjectConfig
    source: ProjectSource
    mirror: pathlib.Path
    workers: int


def require_sha(job: DispatchJob) -> str:
    """The commit a node-lane job names.

    Args:
        job: The claimed job.

    Returns:
        Its sha.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when the job carries none,
            which the queue's pin (MCPs mig 532) makes impossible for a
            node-lane row; a null here means the contract moved.
    """
    sha = job["sha"]
    if sha is None:
        raise AppError(
            code=FleetErrorCode.QUEUE_ANSWER_MALFORMED,
            message=f"node-lane job {job['job_id']} carries no sha; the queue's pin forbids it",
        )
    return sha


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
    tags against it, the remote, the commit on the remote, and only then
    the project's fit on this node, judged on the probe the claim pass
    already took, so no second ssh is paid.

    Args:
        loaded: The workspace and its resolved record paths.
        job: The claimed job.
        node: This node's declaration.
        state: What it reported when probed this tick.
        sha: The job's commit.

    Returns:
        What the dispatch needs, or the ``PROJECT_TAGS_MISMATCH`` refusal
        as its ``CODE: message`` line.

    Raises:
        AppError: ``WORKSPACE_PROJECT_UNKNOWN``, ``PROJECT_REMOTE_MISSING``,
            ``SHA_NOT_ON_REMOTE``, ``EXPORT_FAILED``, ``RESOURCE_HELD`` from
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
    run_cli.require_resources_free(loaded, plan)
    workers = capacity.plan_dispatch(node, state, plan)
    return Prepared(plan=plan, source=source, mirror=mirror, workers=workers)


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
    collect_pass(loaded, credentials, board, identity, agent=agent)
    if claim_pass(loaded, credentials, identity, alias=alias, node=node) is None:
        _log.info("nothing in the node lane for %s", alias)
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level="INFO",
        format_mode="text",
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
    "CLAIM_LEASE_SECONDS",
    "IDENTITY_NAMESPACE",
    "NODE_FLAG",
    "Prepared",
    "claim_pass",
    "collect_one_job",
    "collect_pass",
    "entrypoint",
    "main",
    "node_identity",
    "prepare",
    "refuse",
    "require_sha",
    "tags_refusal",
]
