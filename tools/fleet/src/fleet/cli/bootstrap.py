"""CLI: is a node able to run a build, and make it so.

Usage:
    fleet-bootstrap --config fleet.json
    fleet-bootstrap --config fleet.json --node lavender
    fleet-bootstrap --config fleet.json --node lavender --install

WHY THIS EXISTS. Measured 2026-09-04: of three reachable nodes, ONE could have
run a ``make check``. ``make`` was on loki alone, poetry was absent from
lavender, and the repository was on no node at all. A dispatcher that assumed
a machine on the tailnet was ready would stage a whole monorepo, launch, and
fail on the first line of the recipe -- having spent the transfer to learn
something one probe answers instantly.

REPORTING IS THE DEFAULT AND INSTALLING IS NOT. These are other people's
machines. A tool that installed software on one because a build wanted it
would be doing the thing this package exists to stop -- acting on somebody
else's computer without their knowing -- so ``--install`` has to be typed, and
it only ever installs tools this package has a command for. tar carries none,
since it ships with the platform. Python carried none until 2026-09-23 on the
grounds that which interpreter a machine should have is a decision; the
operator made it that day (python.org 3.11.9 at user scope,
:data:`fleet.contracts.toolchain.PINNED_PYTHON`), after lavender's runner
met only the Store alias and every slime check it took died at its first
python call.

THE EXIT STATUS IS THE ANSWER. Zero when every node asked is ready, one when
any is not, so this is usable as a gate in front of a dispatch rather than
something a person has to read.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.logging import LogFormat, LogLevel, get_logger, setup_logging

from fleet.cli import _config
from fleet.contracts.node import NodeConfig
from fleet.contracts.toolchain import describe_gap, is_ready
from fleet.contracts.workspace import require_node
from fleet.core import toolchain

_log = get_logger(__name__)

NODE_FLAG = "--node"
INSTALL_FLAG = "--install"

_FLAGS = (_config.CONFIG_FLAG, NODE_FLAG)


def bootstrap_node(name: str, node: NodeConfig, *, install: bool) -> tuple[str, bool]:
    """Probe one node, optionally installing what it lacks, and describe it.

    Args:
        name: The node's workspace name.
        node: Its declaration.
        install: Whether to install the absent tools this package can supply.

    Returns:
        The line to print, and whether the node ended up ready.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` if it cannot be reached, or
            ``DISPATCH_FAILED`` if the probe or an install command fails.
            Not softened: a half-installed node is worse than an untouched
            one, because it looks ready.
    """
    reports = toolchain.probe_toolchain(node)
    if not install:
        return describe_gap(name, reports), is_ready(reports)

    installed = toolchain.install_missing(node, reports)
    if not installed:
        return describe_gap(name, reports), is_ready(reports)

    after = toolchain.probe_toolchain(node)
    return (
        f"{describe_gap(name, after)} (installed {', '.join(installed)})",
        is_ready(after),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Report whether the fleet's nodes can run a build.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 when every node asked is ready, 1 when any is not.

    Raises:
        ValueError: When a flag is unknown, repeated, or missing its value.
        AppError: With ``WORKSPACE_NODE_UNKNOWN`` if a named node is not
            declared, or a transport code if one cannot be reached.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    install = INSTALL_FLAG in tokens
    parsed = cli_args.parse_single_flags(
        [token for token in tokens if token != INSTALL_FLAG], _FLAGS
    )
    loaded = _config.load_workspace(parsed)

    named = parsed.get(NODE_FLAG)
    if named is None:
        wanted = sorted(loaded.workspace["nodes"].items())
    else:
        wanted = [(named, require_node(loaded.workspace, named))]

    unready = 0
    for name, node in wanted:
        line, ready = bootstrap_node(name, node, install=install)
        _log.info("%s", line)
        unready += 0 if ready else 1

    if unready:
        _log.info("%d node(s) cannot run a build yet", unready)
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
        service_name="fleet-bootstrap",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = ["bootstrap_node", "entrypoint", "main"]


# Without this, `python -m fleet.cli.bootstrap` imports the module, runs
# nothing and exits 0 -- which reads as "every node is ready" and is the worst
# possible false answer for a gate.
if __name__ == "__main__":
    entrypoint()
