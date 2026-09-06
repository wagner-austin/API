"""CLI: dispatch a project's build to a node.

Usage:
    fleet-run --config fleet.json --project services/Model-Trainer \\
        --agent opus-fleet-0904 --session <uuid>
    fleet-run --config fleet.json --project services/Model-Trainer \\
        --agent opus-fleet-0904 --session <uuid> --node lavender

IT RETURNS AS SOON AS THE SUITE IS RUNNING, and does not wait for it. The
build outlives this command because it is launched through the node's task
scheduler rather than as a child of the ssh call -- see
:mod:`fleet.core.dispatch` for the job-object reason that is not optional.

So the result arrives on the feed, not on this command's standard output:

    fleet-watch --config fleet.json --run <the printed run id>

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
from platform_core.errors import AppError, FleetErrorCode
from platform_core.logging import get_logger, setup_logging

from fleet.cli import _config
from fleet.contracts.lease import describe_contention
from fleet.contracts.node import NodeConfig, NodeState
from fleet.contracts.project import ProjectConfig
from fleet.contracts.workspace import require_node, require_project
from fleet.core import _test_hooks, capacity, dispatch, leases, probe, records

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
        AppError: With a workspace code if a name is undeclared,
            ``NODE_DISABLED`` if the named node is one the workspace says is
            off, a capacity code if the chosen node refuses, or -- when no
            node can take the work -- whichever of ``NODE_UNREACHABLE`` /
            ``NODE_DISABLED`` / ``NODE_MEMORY_EXHAUSTED``
            :func:`capacity.first_fit` classifies the fleet-wide refusal as.
    """
    plan = require_project(loaded.workspace, project)
    require_resources_free(loaded, plan)
    if named is not None:
        node = require_node(loaded.workspace, named)
        if not node["enabled"]:
            # Its own code, not NODE_UNREACHABLE. "Was never asked" is not
            # "did not answer", and silently rerouting to another machine
            # would answer a question nobody put.
            raise AppError(
                FleetErrorCode.NODE_DISABLED,
                f"{named} is declared disabled in this workspace, so nothing was asked of "
                "it. Re-enable it in fleet.json once it is expected to answer, or dispatch "
                "without --node to let the fleet choose among the machines that are.",
            )
        state = _probe(loaded, name=named, node=node)
        return named, node, capacity.plan_dispatch(node, state, plan)

    candidates: list[tuple[str, NodeConfig, NodeState]] = []
    unassessed: list[capacity.Unassessed] = []
    for name, declared in sorted(loaded.workspace["nodes"].items()):
        if not declared["enabled"]:
            # NOT PROBED AT ALL, and ``asked=False`` is how the refusal knows
            # to say so. A disabled node is one nobody expects to answer, and
            # asking anyway costs a ten-second ssh timeout per dispatch --
            # which is what this workspace did for loki, every time, for the
            # whole of 2026-09-05.
            unassessed.append(
                capacity.Unassessed(
                    name=name, reason="declared disabled in this workspace", asked=False
                )
            )
            continue
        # The VALUE form, deliberately. A node that does not answer is one
        # fewer candidate, not the end of the search -- see
        # ``capacity.first_fit``'s ``unassessed`` for the dispatch that was
        # refused because loki was asleep while lavender had room.
        outcome = probe.attempt_probe(
            declared, live_runs=records.live_runs(loaded.ledger, node=name)
        )
        answered: NodeState | None = outcome["state"]
        if answered is None:
            unassessed.append(capacity.Unassessed(name=name, reason=outcome["reason"], asked=True))
            continue
        candidates.append((name, declared, answered))
    chosen, workers = capacity.first_fit(tuple(candidates), plan, unassessed=tuple(unassessed))
    return chosen, loaded.workspace["nodes"][chosen], workers


def require_resources_free(loaded: _config.LoadedWorkspace, plan: ProjectConfig) -> None:
    """Refuse before probing anything if a fleet-wide resource is held.

    CHECKED BEFORE THE NODES, not among them. A resource there is one of in
    the fleet makes EVERY node refuse for the same reason, so probing three
    and collecting three identical refusals costs three round trips and
    produces a message shaped like a capacity problem -- which sends the
    reader looking for a bigger machine that would not have helped.

    This is not the enforcement. :func:`fleet.core.leases.acquire` is, and it
    re-checks through the same function, so a resource taken between here and
    there is still refused.

    Args:
        loaded: The workspace and its resolved record paths.
        plan: The project being dispatched.

    Raises:
        AppError: With ``RESOURCE_HELD``, naming the resource, its holder,
            and that no other node is an alternative.
    """
    contention = leases.contended_by(
        loaded.leases,
        wanted=plan["exclusive_resources"],
        now_unix=_test_hooks.now(),
    )
    if contention is None:
        return
    blocking, names = contention
    detail = describe_contention(blocking, names=names, now_unix=_test_hooks.now())
    raise AppError(FleetErrorCode.RESOURCE_HELD, f"cannot dispatch: {detail}")


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
        archive_dir=loaded.archives,
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


__all__ = ["choose", "entrypoint", "main", "require_resources_free"]


# Without this, `python -m fleet.cli.run` imports the module, runs nothing and
# exits 0 -- which reads as a successful dispatch that never happened.
if __name__ == "__main__":
    entrypoint()
