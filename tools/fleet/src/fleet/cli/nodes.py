"""CLI: what every node has free right now.

Usage:
    fleet-nodes --config fleet.json
    fleet-nodes --config fleet.json --registry ../MCPs/fleet-mcp/fleet-nodes.json

The ``sinfo`` of this package, and the command to run before wondering why a
dispatch was refused. It probes every declared node and prints one line each,
so the answer to "where can this go" is one call rather than three ssh
sessions.

A NODE THAT CANNOT BE REACHED IS A LINE, NOT A CRASH. Three of the fleet's
seven aliases did not answer when this package was written, and a command that
refused to describe four reachable machines because a fifth was off would be
useless exactly when the fleet is degraded. The unreachable node's own error
message is printed on its line, so the reason is visible without a second
command -- and the exit status is non-zero, so a script cannot read a partial
fleet as a whole one.

A DISABLED NODE IS NOT PROBED, and says so. Nobody expects it to answer, and
asking costs a ten-second ssh timeout per node per run.

``--registry`` RECONCILES THIS WORKSPACE AGAINST THE FLEET'S IDENTITY REGISTRY
(``fleet-mcp/fleet-nodes.json``, in the MCPs repo) and exits non-zero if they
disagree about which machines are expected to answer. The fleet is written down
in two repositories -- see :mod:`fleet.core.registry` for why merging them
would be wrong -- and on 2026-09-05 one marked loki off for a trip while the
other kept dispatching to it all day.

THE PATH IS PASSED, NEVER SEARCHED FOR. Omit the flag and no reconciliation is
claimed; a command that hunted for the other repo would report agreement on
any machine where it simply failed to find it.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.errors import AppError
from platform_core.logging import get_logger, setup_logging

from fleet.cli import _config
from fleet.contracts.node import NodeConfig, describe_node
from fleet.core import probe, records, registry

_log = get_logger(__name__)

REGISTRY_FLAG = "--registry"

_FLAGS = (_config.CONFIG_FLAG, REGISTRY_FLAG)


def describe_fleet(loaded: _config.LoadedWorkspace) -> tuple[list[str], int]:
    """Probe every node and render one line each.

    Args:
        loaded: The workspace and its resolved record paths.

    Returns:
        The lines in workspace order, and how many nodes failed to answer.
        The count is returned rather than raised so the caller can print
        every line before deciding the exit status -- a fleet report that
        stopped at the first dead node would hide the live ones.
    """
    lines: list[str] = []
    unreachable = 0
    for name, node in sorted(loaded.workspace["nodes"].items()):
        if not node["enabled"]:
            # Not probed, and not counted against the exit status. A machine
            # nobody expects to answer has not failed to.
            lines.append(f"{name}: DISABLED -- declared off in this workspace, not probed")
            continue
        live = records.live_runs(loaded.ledger, node=name)
        verdict = _probe_line(name, node, live)
        lines.append(verdict[0])
        unreachable += verdict[1]
    return lines, unreachable


def _probe_line(name: str, node: NodeConfig, live: int) -> tuple[str, int]:
    """Probe one node and render it, or render why it could not be probed.

    THE ONE PLACE THIS PACKAGE CATCHES, and the exception is re-raised as
    text on a line rather than swallowed: the status still goes non-zero and
    the node's own message is what gets printed. Catching here is what makes
    a degraded fleet describable at all; catching anywhere else would be
    softening a failure, which is why it is confined to this function.

    Args:
        name: The node's workspace name.
        node: Its declaration.
        live: Fleet dispatches currently live on it.

    Returns:
        The line to print, and 1 when the node did not answer.
    """
    try:
        state = probe.probe_node(node, live_runs=live)
    except AppError as unreachable:
        return f"{name}: UNREACHABLE -- {unreachable.message}", 1
    return f"{name}: {describe_node(node, state)}", 0


def main(argv: Sequence[str] | None = None) -> int:
    """Describe every node in the workspace.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        0 when every enabled node answered and, if ``--registry`` was given,
        the two registries agree. 1 when any enabled node did not answer or
        the registries have drifted.

    Raises:
        ValueError: When a flag is unknown, repeated, or missing its value.
        JSONTypeError: If the workspace document is invalid.
        AppError: ``NODE_REGISTRY_UNREADABLE`` when ``--registry`` names
            something this cannot read. Raised rather than reported as a
            line: a reconciliation that could not run has established
            nothing, and must not be mistaken for one that found agreement.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    loaded = _config.load_workspace(parsed)

    lines, unreachable = describe_fleet(loaded)
    for line in lines:
        _log.info("%s", line)

    drift = _reconcile(loaded, parsed.get(REGISTRY_FLAG))
    for line in drift:
        _log.info("REGISTRY DRIFT %s", line)

    if unreachable:
        _log.info("%d node(s) did not answer", unreachable)
    if drift:
        _log.info("%d registry disagreement(s); the fleet is written down twice", len(drift))
    if unreachable or drift:
        return 1
    return 0


def _reconcile(loaded: _config.LoadedWorkspace, registry_path: str | None) -> tuple[str, ...]:
    """Reconcile against the identity registry, if one was named.

    Args:
        loaded: The workspace and its resolved record paths.
        registry_path: What ``--registry`` was given, or None.

    Returns:
        One line per disagreement, empty when none were found OR when no
        registry was named. Those two are deliberately the same value: the
        caller's exit status is driven by drift FOUND, and a reconciliation
        nobody asked for cannot find any.

    Raises:
        AppError: ``NODE_REGISTRY_UNREADABLE`` from the reader.
    """
    if registry_path is None:
        return ()
    return registry.reconcile(loaded.workspace, registry_path=registry_path)


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="fleet-nodes",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = ["REGISTRY_FLAG", "describe_fleet", "entrypoint", "main"]


# Without this, `python -m fleet.cli.nodes` imports the module, runs nothing
# and exits 0 -- a form that looks like a fleet with no nodes.
if __name__ == "__main__":
    entrypoint()
