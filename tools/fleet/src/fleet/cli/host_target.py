"""CLI: run a project's host target on its node, at its remote's default branch.

Usage:
    python -m fleet.cli.host_target --config registry [--records-dir <hub records>] \\
        --project osrs-lab/manual-desktop --agent <board label> --session <uuid>

MCPs board task 57791f39: osrs-lab's manual-desktop declares ``update`` on
loki, which keeps every OSRS desktop there on RuneLite's newest release, and
the hub runs this command for it every hour. One run fetches the remote's
default branch into the hub's mirror, takes the project's lease on the node,
stages that commit there, runs the target as the node's account under its
deadline (:mod:`fleet.core.host_target`), prints every line make printed
with the commit, records the end on the feed, gives the lease back, retires
the staged tree and exits with make's status.

A FAILURE AFTER THE LEASE GIVES IT BACK, then propagates. This command runs
unattended, so a lease left standing would refuse the next hour's run
``LEASE_HELD`` for a run that is over; :func:`fleet.core.run_lease.abandon`
records the failure on the feed and gives the lease back, and the retire
owed since the lease removes whatever was staged at the next collect pass,
as :func:`fleet.cli.node_launch.launch_claimed` does for a check.

WHY ``--agent`` AND ``--session`` ARE REQUIRED, as for ``fleet-run``: the
lease names who holds the project on the node, and a default would be one
identity shared by every caller.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from platform_core import cli_args
from platform_core.cli_boundary import run_cli
from platform_core.errors import AppError
from platform_core.logging import LogFormat, LogLevel, get_logger, setup_logging
from platform_core.report_line import emit_error_line, emit_line

from fleet.cli import _config
from fleet.contracts.feed import FeedKind
from fleet.contracts.host_target import HostTarget
from fleet.contracts.lease import Lease
from fleet.contracts.node import NodeConfig
from fleet.contracts.workspace import require_project
from fleet.core import (
    _test_hooks,
    commit_stage,
    export,
    host_target,
    leases,
    retire_owed,
    run_lease,
)

_log = get_logger(__name__)

PROJECT_FLAG = "--project"
AGENT_FLAG = "--agent"
SESSION_FLAG = "--session"

#: ``--records-dir`` names the hub's records, so this run's lease, feed
#: lines and owed retire are the ones its node runners and collect read.
_FLAGS = (_config.CONFIG_FLAG, _config.RECORDS_FLAG, PROJECT_FLAG, AGENT_FLAG, SESSION_FLAG)


def _staged_run(
    loaded: _config.LoadedWorkspace,
    lease: Lease,
    *,
    node: NodeConfig,
    plan_path: str,
    branch: export.DefaultBranch,
    remote_url: str,
    declared: HostTarget,
) -> host_target.HostTargetRun:
    """Stage the head under the held lease, announce it, and run the target.

    Args:
        loaded: The workspace and its resolved record paths.
        lease: The lease this run holds.
        node: The node the host target runs on.
        plan_path: The project's directory in its repository.
        branch: The default branch fetched into the hub's mirror.
        remote_url: The project's declared remote.
        declared: The project's host target.

    Returns:
        The run on the node.

    Raises:
        AppError: As :func:`fleet.core.commit_stage.stage_commit` and
            :func:`fleet.core.host_target.run_on_node` describe.
    """
    run_id = lease["run_id"]
    key = export.remote_mirror_key(remote_url)
    staged = commit_stage.stage_commit(
        node["host"],
        stage_root=node["stage_root"],
        run_id=run_id,
        commit=commit_stage.CommitExport(
            mirror=branch["mirror"],
            key=key,
            sha=branch["head"],
            scope=(),
            bundle=loaded.archives / f"{run_id}.bundle",
            base_branch=None,
        ),
    )
    run_lease.emit(
        loaded.feed,
        run_id=run_id,
        node=lease["node"],
        project=lease["project"],
        kind=FeedKind.STAGED,
        detail=staged["detail"],
        now_unix=_test_hooks.now(),
    )
    return host_target.run_on_node(
        node, run_id=run_id, target=staged["target"], path=plan_path, host_target=declared
    )


def run(
    loaded: _config.LoadedWorkspace,
    *,
    project: str,
    agent: str,
    session_id: str,
    emit: Callable[[str], None],
) -> int:
    """Run one project's host target at its default branch's head.

    Args:
        loaded: The workspace and its resolved record paths.
        project: The project key.
        agent: Board label of the caller, which the lease names.
        session_id: That caller's session UUID.
        emit: Where each of make's lines and the summary line go.

    Returns:
        make's exit status, or ``timeout``'s when the deadline stopped it.

    Raises:
        AppError: ``WORKSPACE_PROJECT_UNKNOWN``, ``HOST_TARGET_UNDECLARED``,
            ``WORKSPACE_NODE_UNKNOWN`` or ``NODE_DISABLED`` before anything
            is fetched; ``PROJECT_REMOTE_MISSING`` or a fetch's code before
            the lease; ``LEASE_HELD`` from the lease; a staging or transport
            code once it is held, after the lease is given back; and
            ``DISPATCH_FAILED`` from a retire the node answered and failed.
    """
    plan = require_project(loaded.workspace, project)
    declared = host_target.require_host_target(project, plan)
    node_name = declared["node"]
    node = host_target.require_host_node(loaded.workspace, declared)
    source = export.require_source(project, plan["source"])
    branch = export.prepare_default_branch(loaded.mirrors, remote=source["remote"])
    lease = run_lease.take(
        loaded.leases,
        loaded.feed,
        loaded.retires,
        node_name=node_name,
        project=project,
        plan=plan,
        window_seconds=run_lease.host_target_window(declared),
        workers=declared["workers"],
        agent=agent,
        session_id=session_id,
        node_local=loaded.workspace["node_local_resources"],
    )
    run_id = lease["run_id"]
    try:
        ran = _staged_run(
            loaded,
            lease,
            node=node,
            plan_path=source["path"],
            branch=branch,
            remote_url=source["remote"],
            declared=declared,
        )
    except AppError as refusal:
        given_back = run_lease.abandon(
            loaded.leases, loaded.feed, lease=lease, detail=f"{refusal.code}: {refusal.message}"
        )
        _log.info("%s ended before make did; %s", run_id, given_back)
        raise
    head = branch["head"]
    for line in ran["lines"]:
        emit(f"{head} {line}")
    ended = host_target.describe(ran, declared)
    run_lease.emit(
        loaded.feed,
        run_id=run_id,
        node=node_name,
        project=project,
        kind=FeedKind.FAILED if ran["status"] else FeedKind.PASSED,
        detail=ended,
        now_unix=_test_hooks.now(),
    )
    leases.release(loaded.leases, run_id=run_id, now_unix=_test_hooks.now())
    retired = retire_owed.retire_or_owe(loaded.retires, node, alias=node_name, run_id=run_id)
    emit(
        f"host target {declared['target']} of {project} at {head} on {node_name}, run {run_id}: "
        f"{ended}; retire {retired}"
    )
    return ran["status"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the named project's host target and exit with make's status.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        make's exit status.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or a
            required one is absent.
        AppError: As :func:`run` describes.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    return run(
        _config.load_workspace(parsed),
        project=cli_args.require_flag(parsed, PROJECT_FLAG),
        agent=cli_args.require_flag(parsed, AGENT_FLAG),
        session_id=cli_args.require_flag(parsed, SESSION_FLAG),
        emit=emit_line,
    )


def entrypoint() -> None:
    """Module entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level=LogLevel.INFO,
        format_mode=LogFormat.TEXT,
        service_name="fleet-host-target",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(run_cli(main, emit=emit_line, emit_error=emit_error_line))


__all__ = ["entrypoint", "main", "run"]


# Without this, `python -m fleet.cli.host_target` imports the module, runs
# nothing and exits 0 -- which reads as an update that ran and changed nothing.
if __name__ == "__main__":
    entrypoint()
