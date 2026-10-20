"""CLI: what every node has free right now.

Usage:
    fleet-nodes --config registry

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

EACH LINE SAYS WHETHER THE NODE'S RUNNERS CAN CLAIM, AND WHY NOT (MCPs board
task 7e467416). On 2026-09-29 thirty checks sat queued with one running and
the reasons were only in each runner's daily log on the hub: sedona held 2.9 GB
free against its 4.0 GB reservation, lavender 2.2 GB against 6.0 GB. So a
reachable node's line ends with every runner's own pre-claim verdict,
:func:`fleet.core.capacity.room_for_any` on the same probe (the owner's
reservation against free memory, the cores and memory its live runs hold, disk),
for its node lane and, where it declares one, its elevated lane; and an
unreachable node inside another machine's WSL adds what that host reports.

A DISABLED NODE IS NOT PROBED, and says so. Nobody expects it to answer, and
asking costs a ten-second ssh timeout per node per run.

Whether a machine is expected to answer is said once, by the ``enabled`` of
its own ``registry/machines/<name>.json``, the fleet's one roster (MCPs board
task e447de72). Until then MCPs' ``fleet-mcp/fleet-nodes.json`` said it a
second time, and this command's ``--registry`` and ``--probe never`` compared
the two only when a session ran them by hand; both went with that copy.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.cli_boundary import run_cli
from platform_core.errors import AppError
from platform_core.logging import LogFormat, LogLevel, get_logger, setup_logging
from platform_core.report_line import emit_error_line, emit_line

from fleet.cli import _config
from fleet.contracts.node import NodeConfig, NodeState, describe_node
from fleet.contracts.project import ProjectConfig
from fleet.contracts.tags import runner_tags
from fleet.core import capacity, host_report, probe, records

_log = get_logger(__name__)

#: The writer this command's capacity probes are named for on each node
#: (:func:`fleet.core.names.capacity_probe_stem`).
PROBE_WRITER = "fleet-nodes"

_FLAGS = (_config.CONFIG_FLAG,)


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
            explained = name in loaded.workspace["not_dispatchable"]
            why = "; its machine file gives the reason it is held" if explained else ""
            lines.append(f"{name}: DISABLED -- declared off in this workspace, not probed{why}")
            continue
        verdict = _probe_line(loaded, name, node)
        lines.append(verdict[0])
        unreachable += verdict[1]
    return lines, unreachable


def lane_verdicts(node: NodeConfig, state: NodeState, projects: tuple[ProjectConfig, ...]) -> str:
    """Whether each of a node's runners could claim on this probe, and why not.

    Args:
        node: The node's declaration.
        state: What it reported just now.
        projects: Every registered project, as the runners size a claim.

    Returns:
        One clause per runner, the node lane and (where the node declares
        one) the elevated lane, each ``<lane> can claim`` or ``<lane> claims
        nothing: CODE: reason`` in the runner's own words.
    """
    lanes = [("node lane", False)] + ([("elevated lane", True)] if node["elevated"] else [])
    clauses: list[str] = []
    for label, elevated in lanes:
        full = capacity.room_for_any(node, state, projects, runner_tags(node, elevated=elevated))
        clauses.append(f"{label} can claim" if full is None else f"{label} claims nothing: {full}")
    return "; ".join(clauses)


def _probe_line(loaded: _config.LoadedWorkspace, name: str, node: NodeConfig) -> tuple[str, int]:
    """Probe one node and render it, or render why it could not be probed.

    THE ONE PLACE THIS PACKAGE CATCHES, and the exception is re-raised as
    text on a line rather than swallowed: the status still goes non-zero and
    the node's own message is what gets printed. Catching here is what makes
    a degraded fleet describable at all; catching anywhere else would be
    softening a failure, which is why it is confined to this function.

    Args:
        loaded: The workspace and its resolved record paths.
        name: The node's workspace name.
        node: Its declaration.

    Returns:
        The line to print, and 1 when the node did not answer.
    """
    live = records.live_load(loaded.ledger, node=name, projects=loaded.workspace["projects"])
    try:
        state = probe.probe_node(node, live=live, writer=PROBE_WRITER)
    except AppError as unreachable:
        line = f"{name}: UNREACHABLE -- {unreachable.message}"
        if node["wsl_host"] is not None:
            seen = host_report.describe_wsl_host(
                loaded.workspace, loaded.ledger, node["wsl_host"], writer=PROBE_WRITER
            )
            line = f"{line}; {seen}"
        return line, 1
    projects = tuple(loaded.workspace["projects"].values())
    return f"{name}: {describe_node(node, state)}; {lane_verdicts(node, state, projects)}", 0


def main(argv: Sequence[str] | None = None) -> int:
    """Describe every node in the workspace.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        0 when every probed node answered, 1 when any did not.

    Raises:
        ValueError: When a flag is unknown, repeated or missing its value.
        JSONTypeError: If the workspace document is invalid.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    loaded = _config.load_workspace(parsed)

    lines, unreachable = describe_fleet(loaded)
    for line in lines:
        _log.info("%s", line)
    if unreachable:
        _log.info("%d node(s) did not answer", unreachable)
        return 1
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level=LogLevel.INFO,
        format_mode=LogFormat.TEXT,
        service_name="fleet-nodes",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(run_cli(main, emit=emit_line, emit_error=emit_error_line))


__all__ = [
    "describe_fleet",
    "entrypoint",
    "main",
]


# Without this, `python -m fleet.cli.nodes` imports the module, runs nothing
# and exits 0 -- a form that looks like a fleet with no nodes.
if __name__ == "__main__":
    entrypoint()
