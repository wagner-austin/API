"""CLI: run a fleet agent from the rolled commit (board task 465689f5).

Usage:
    python -m fleet.cli.rolled --repo-root C:/Users/Test/PROJECTS/API \\
        --agent fleet-agent -- --agent fleet-runner-austinpc --session <uuid> ...
    python -m fleet.cli.rolled --repo-root C:/Users/Test/PROJECTS/API \\
        --agent fleet-node-agent -- --node sedona

What each scheduled tick runs instead of the checkout's ``fleet-agent``:
:mod:`fleet.core.rolled` extracts the commit ``make fleet-roll`` recorded,
and the named agent runs from that extraction, its four source trees first
on ``PYTHONPATH``, the extracted registry as its ``--config`` and the
checkout's ``tools/fleet`` as its ``--records-dir``, so the ledger, feed and
leases it keeps are the ones every earlier tick kept. Everything after
``--`` is the agent's own command line, passed through unchanged except
that it may carry neither flag: a tick that named another registry would
run pinned code against an unpinned configuration, and one that named
other records would run the fleet on state no other tick sees.

THIS MODULE IS THE ONE PIECE THAT RUNS FROM THE CHECKOUT, because something
has to read the ref before anything pinned exists. It resolves, extracts
and launches, and nothing else; the agent it launches, every module that
agent imports and the registry it reads come from the roll.

The exit status is the agent's, 2 when no rolled tree could be extracted,
and the child's own timeout status when it outlives
:data:`AGENT_WALL_SECONDS`.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence
from typing import Final

from platform_core import cli_args
from platform_core.logging import LogFormat, LogLevel, get_logger, setup_logging

from fleet.cli import _config
from fleet.core import _test_hooks, rolled

_log = get_logger(__name__)

REPO_ROOT_FLAG: Final = "--repo-root"
AGENT_FLAG: Final = "--agent"
_FLAGS: Final = (REPO_ROOT_FLAG, AGENT_FLAG)

#: Where the launcher's own flags end and the agent's begin.
SEPARATOR: Final = "--"

#: The agents a tick may run, by console-script name, and the module each is.
AGENT_MODULES: Final[dict[str, str]] = {
    "fleet-agent": "fleet.cli.agent",
    "fleet-node-agent": "fleet.cli.node_agent",
}

#: The agent's deadline: one minute inside the 40-minute ExecutionTimeLimit
#: both scheduled tasks carry, so the launcher reports a timeout in the
#: tick's log before Task Scheduler ends the tick without a word.
AGENT_WALL_SECONDS: Final[int] = 39 * 60

#: The exit status when no rolled tree could be extracted.
REFUSED_EXIT: Final[int] = 2


def split_command_line(tokens: Sequence[str]) -> tuple[dict[str, str], list[str]]:
    """Separate the launcher's flags from the agent's command line.

    Args:
        tokens: Everything after the program name.

    Returns:
        The launcher's parsed flags, and the agent's tokens.

    Raises:
        ValueError: When ``--`` is absent, or the agent's tokens carry
            ``--config``.
    """
    if SEPARATOR not in tokens:
        raise ValueError(
            f"FLEET_ROLL_USAGE: {SEPARATOR} must separate {list(_FLAGS)} from the agent's own "
            "command line"
        )
    split = tokens.index(SEPARATOR)
    passthrough = list(tokens[split + 1 :])
    for flag in (_config.CONFIG_FLAG, _config.RECORDS_FLAG):
        if flag in passthrough:
            raise ValueError(
                f"FLEET_ROLL_USAGE: the agent's command line may not carry {flag}; the "
                "registry is the rolled commit's own and the records are the checkout's"
            )
    return cli_args.parse_single_flags(tokens[:split], _FLAGS), passthrough


def require_module(agent: str) -> str:
    """Name the module a console-script name runs.

    Args:
        agent: ``fleet-agent`` or ``fleet-node-agent``.

    Returns:
        Its module.

    Raises:
        ValueError: For any other name.
    """
    module = AGENT_MODULES.get(agent)
    if module is None:
        raise ValueError(f"FLEET_ROLL_USAGE: {AGENT_FLAG} is one of {sorted(AGENT_MODULES)}")
    return module


def main(argv: Sequence[str]) -> int:
    """Extract the rolled tree and run one agent from it.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        The agent's exit status, or :data:`REFUSED_EXIT` when the rolled
        tree could not be extracted and nothing ran.

    Raises:
        ValueError: For a malformed command line.
    """
    parsed, passthrough = split_command_line(argv)
    module = require_module(cli_args.require_flag(parsed, AGENT_FLAG))
    repo_root = pathlib.Path(cli_args.require_flag(parsed, REPO_ROOT_FLAG))
    tree = rolled.extract_rolled_tree(repo_root)
    if isinstance(tree, str):
        _log.error("%s", tree)
        return REFUSED_EXIT
    _log.info("fleet-roll: %s runs from %s at %s", module, rolled.ROLLED_REF, tree["commit"])
    records = repo_root / pathlib.PurePosixPath(rolled.RECORDS_DIR)
    result = _test_hooks.run(
        (
            sys.executable,
            "-m",
            module,
            _config.CONFIG_FLAG,
            tree["config"],
            _config.RECORDS_FLAG,
            str(records),
            *passthrough,
        ),
        timeout_seconds=AGENT_WALL_SECONDS,
        set_env=(("PYTHONPATH", tree["python_path"]),),
    )
    rolled.discard_rolled_tree(tree)
    if result["stdout"]:
        _log.info("%s", result["stdout"].rstrip())
    if result["stderr"]:
        _log.info("%s", result["stderr"].rstrip())
    return result["returncode"]


def entrypoint() -> None:
    """Module entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level=LogLevel.INFO,
        format_mode=LogFormat.TEXT,
        service_name="fleet-rolled",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main(sys.argv[1:]))


__all__ = [
    "AGENT_FLAG",
    "AGENT_MODULES",
    "AGENT_WALL_SECONDS",
    "REFUSED_EXIT",
    "REPO_ROOT_FLAG",
    "SEPARATOR",
    "entrypoint",
    "main",
    "require_module",
    "split_command_line",
]


# Without this, `python -m fleet.cli.rolled` imports the module, runs nothing
# and exits 0, and a tick that ran no agent would read as a clean one.
if __name__ == "__main__":
    entrypoint()
