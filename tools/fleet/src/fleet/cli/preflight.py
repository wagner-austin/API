"""CLI: would this project run right now, and on which node.

Usage:
    fleet-preflight --config fleet.json --project services/Model-Trainer
    fleet-preflight --config fleet.json --project services/Model-Trainer \\
        --node lavender

Two questions, one command. Without ``--node`` it asks the fleet and names the
node that affords the most workers; with one it asks that node and reports its
refusal if it has one. The second form exists because "some work needs a
particular machine" is a real reason, and a tool that only ever chose for you
could not express it.

NOTHING IS DISPATCHED AND NO LEASE IS TAKEN. A preflight that acquired a lease
would make asking a question cost the thing being asked about, and two people
checking would deadlock. The consequence is that a preflight's answer can be
stale by the time a dispatch runs, which is why :mod:`fleet.core.capacity` is
consulted again inside the dispatch rather than trusting this.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.logging import get_logger, setup_logging

from fleet.cli import _config, run
from fleet.contracts.node import NodeConfig, NodeState
from fleet.contracts.workspace import require_node, require_project
from fleet.core import capacity, probe, records

_log = get_logger(__name__)

PROJECT_FLAG = "--project"
NODE_FLAG = "--node"

_FLAGS = (_config.CONFIG_FLAG, PROJECT_FLAG, NODE_FLAG)


def preflight_named_node(loaded: _config.LoadedWorkspace, *, project: str, node: str) -> int:
    """Ask one named node whether it would take this project.

    Args:
        loaded: The workspace and its resolved record paths.
        project: Repo-relative project path.
        node: The node's workspace name.

    Returns:
        The worker count that node would grant.

    Raises:
        AppError: With ``WORKSPACE_NODE_UNKNOWN`` or
            ``WORKSPACE_PROJECT_UNKNOWN`` if either name is undeclared, with
            ``NODE_UNREACHABLE`` if the node does not answer, or with a
            capacity code if it answers and refuses.
    """
    declared = require_node(loaded.workspace, node)
    wanted = require_project(loaded.workspace, project)
    state = probe.probe_node(declared, live_runs=records.live_runs(loaded.ledger, node=node))
    return capacity.plan_dispatch(declared, state, wanted)


def preflight_fleet(loaded: _config.LoadedWorkspace, *, project: str) -> tuple[str, int]:
    """Ask every node and choose the one that affords the most workers.

    A node that does not answer is left out of the running rather than
    failing the command: the question is where this can run, and one machine
    being off does not make the others unusable. Its absence is reported by
    ``fleet-nodes``, which is the command whose job that is.

    Args:
        loaded: The workspace and its resolved record paths.
        project: Repo-relative project path.

    Returns:
        The chosen node's name and its worker count.

    Raises:
        AppError: With ``WORKSPACE_PROJECT_UNKNOWN`` if the project is
            undeclared, or ``NODE_MEMORY_EXHAUSTED`` when no node can take
            it, carrying every node's own refusal.
    """
    wanted = require_project(loaded.workspace, project)
    candidates: list[tuple[str, NodeConfig, NodeState]] = []
    for name, declared in sorted(loaded.workspace["nodes"].items()):
        live = records.live_runs(loaded.ledger, node=name)
        candidates.append((name, declared, probe.probe_node(declared, live_runs=live)))
    return capacity.first_fit(tuple(candidates), wanted)


def main(argv: Sequence[str] | None = None) -> int:
    """Report whether this project would run, and where.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 when a node would take the work.

    Raises:
        ValueError: When a flag is unknown, repeated, or missing its value,
            or ``--config`` or ``--project`` is absent.
        AppError: When no node can take the work, or a named one refuses.
            Raised rather than returned as a non-zero status, because a
            refusal carries a reason and an exit code carries none.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    loaded = _config.load_workspace(parsed)
    project = cli_args.require_flag(parsed, PROJECT_FLAG)

    # A fleet-wide resource is checked first, through the same function the
    # dispatch enforces with. Without it this command would answer "yes,
    # lavender, 12 workers" about a project that fleet-run will refuse for a
    # reason no node can fix -- which is worse than not asking, because the
    # reader now has a specific wrong answer.
    run.require_resources_free(loaded, require_project(loaded.workspace, project))

    named = parsed.get(NODE_FLAG)
    if named is None:
        node, workers = preflight_fleet(loaded, project=project)
    else:
        node, workers = named, preflight_named_node(loaded, project=project, node=named)

    _log.info("%s would run on %s with %d worker(s)", project, node, workers)
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="fleet-preflight",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = ["entrypoint", "main", "preflight_fleet", "preflight_named_node"]


# Without this, `python -m fleet.cli.preflight` imports the module, runs
# nothing and exits 0 -- which reads as "it would run" and is the worst
# possible false answer for this particular command.
if __name__ == "__main__":
    entrypoint()
