"""CLI: dispatch a project's build to a node.

Usage:
    fleet-run --config runs/fleet.json --project services/Model-Trainer \\
        --agent opus-fleet-0904 --session <uuid>
    fleet-run --config runs/fleet.json --project services/Model-Trainer \\
        --agent opus-fleet-0904 --session <uuid> --node lavender

IT RETURNS AS SOON AS THE SUITE IS RUNNING, and does not wait for it. The
build outlives this command because it is launched through the node's task
scheduler rather than as a child of the ssh call -- see
:mod:`fleet.core.dispatch` for the job-object reason that is not optional.

So the result arrives on the feed, not on this command's standard output:

    fleet-watch --config runs/fleet.json --run <the printed run id>

WHY ``--agent`` AND ``--session`` ARE REQUIRED. The incident this package
exists for was two sessions colliding with no way for either to know the other
was there, and a refusal that could only say "another dispatch holds this" is
half a diagnostic. These are the board's own identity fields, so a ledger row
and a board post can be matched by whoever reads both. Required rather than
defaulted, because a default would be one label shared by every session, which
is the same as having none.

A REFUSAL RAISES AND IS NOT RECORDED, deliberately. A dispatch turned away
never got a run id, so there is nothing for a feed subscriber to have been
filtering on -- the reason goes to whoever ran the command, in the exception's
own message, which names every node it asked and why each declined.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.logging import get_logger, setup_logging

from fleet.cli import _config
from fleet.contracts.node import NodeConfig, NodeState
from fleet.contracts.workspace import require_node, require_project
from fleet.core import capacity, dispatch, probe, records

_log = get_logger(__name__)

PROJECT_FLAG = "--project"
NODE_FLAG = "--node"
AGENT_FLAG = "--agent"
SESSION_FLAG = "--session"
ROOT_FLAG = "--repo-root"

_FLAGS = (
    _config.CONFIG_FLAG,
    PROJECT_FLAG,
    NODE_FLAG,
    AGENT_FLAG,
    SESSION_FLAG,
    ROOT_FLAG,
)


def choose(
    loaded: _config.LoadedWorkspace, *, project: str, named: str | None
) -> tuple[str, NodeConfig, int]:
    """Decide which node runs this, and how many workers it gets.

    Args:
        loaded: The workspace and its resolved record paths.
        project: Repo-relative project path.
        named: The node the caller insisted on, or None to let the fleet
            choose.

    Returns:
        The node's workspace name, its declaration, and the worker count.

    Raises:
        AppError: With a workspace code if a name is undeclared, a capacity
            code if the chosen node refuses, or ``NODE_MEMORY_EXHAUSTED`` if
            no node can take the work.
    """
    plan = require_project(loaded.workspace, project)
    if named is not None:
        node = require_node(loaded.workspace, named)
        state = _probe(loaded, name=named, node=node)
        return named, node, capacity.plan_dispatch(node, state, plan)

    candidates: list[tuple[str, NodeConfig, NodeState]] = []
    for name, declared in sorted(loaded.workspace["nodes"].items()):
        candidates.append((name, declared, _probe(loaded, name=name, node=declared)))
    chosen, workers = capacity.first_fit(tuple(candidates), plan)
    return chosen, loaded.workspace["nodes"][chosen], workers


def _probe(loaded: _config.LoadedWorkspace, *, name: str, node: NodeConfig) -> NodeState:
    """Ask one node what it has free, counting our own live runs on it.

    Args:
        loaded: The workspace and its resolved record paths.
        name: The node's workspace name.
        node: Its declaration.

    Returns:
        Its live state.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` if it does not answer.
    """
    return probe.probe_node(node, live_runs=records.live_runs(loaded.ledger, node=name))


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one project's build and print the run id.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 once the suite is running on a node.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or a
            required one is absent.
        AppError: When no node will take the work, when another dispatch
            holds the project, or when staging or launching fails.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    loaded = _config.load_workspace(parsed)
    project = cli_args.require_flag(parsed, PROJECT_FLAG)
    agent = cli_args.require_flag(parsed, AGENT_FLAG)
    session_id = cli_args.require_flag(parsed, SESSION_FLAG)
    project_root = pathlib.Path(cli_args.require_flag(parsed, ROOT_FLAG)).resolve()

    node_name, node, workers = choose(loaded, project=project, named=parsed.get(NODE_FLAG))
    row = dispatch.start(
        loaded.leases,
        loaded.ledger,
        loaded.feed,
        node_name=node_name,
        node=node,
        project=project,
        plan=require_project(loaded.workspace, project),
        workers=workers,
        agent=agent,
        session_id=session_id,
        project_root=project_root,
        archive_dir=loaded.directory,
    )

    _log.info(
        "dispatched %s to %s as %s with %d worker(s)",
        project,
        node_name,
        row["run_id"],
        workers,
    )
    _log.info("follow it: fleet-watch --config <this> --run %s", row["run_id"])
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="fleet-run",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = ["choose", "entrypoint", "main"]


# Without this, `python -m fleet.cli.run` imports the module, runs nothing and
# exits 0 -- which reads as a successful dispatch that never happened.
if __name__ == "__main__":
    entrypoint()
