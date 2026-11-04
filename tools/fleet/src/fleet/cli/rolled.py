"""CLI: run a fleet agent from the rolled commit (board task 465689f5).

Usage:
    python -m fleet.cli.rolled --repo-root /home/corvis/PROJECTS/API \\
        --agent fleet-agent -- --agent fleet-hub-diphtheria --session <uuid> ...
    python -m fleet.cli.rolled --repo-root /home/corvis/PROJECTS/API \\
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
has to read the ref before anything pinned exists. It resolves, finds or
extracts the commit's one tree (removing what earlier ticks left behind in
the same turn, :mod:`fleet.core.scratch`) and launches, and nothing else;
the agent it launches, every module that agent imports and the registry it
reads come from the roll. The tree outlives the agent, because the next
tick of the same roll runs from it (MCPs board task fb505198).

The exit status is the agent's, 2 when no rolled tree could be extracted,
and the child's own timeout status when it outlives
:data:`AGENT_WALL_SECONDS`.

A NODE RUNNER'S REFUSED LAUNCH IS STILL A TICK (MCPs board task b841c214).
The node agent records each tick through ``dispatch_tick``, which keeps the
runner's row and a 30-day history (MCPs mig 903); a launch refused before
the agent started recorded nothing. From 2026-11-04T09:39Z GitHub refused
the hub's key, every node runner's tick ended ``FLEET_ROLL_FETCH_FAILED``,
and for 29 minutes the history said nothing about any of them while the
reason sat only in each runner's log on the hub. So when the refused launch
is a node runner's serving tick, the launcher records that tick itself,
through the same writer, under the runner's own identity: no load, since no
probe ran, no tags or fits, not claiming, and the refusal as its verdict. An
announce run (``--announce``) is a check-in, not a tick, and records none.
"""

from __future__ import annotations

import io
import pathlib
import sys
from collections.abc import Sequence
from typing import Final, TextIO

from platform_core import cli_args
from platform_core.cli_boundary import run_cli
from platform_core.logging import LogFormat, LogLevel, get_logger, setup_logging
from platform_core.report_line import emit_error_line, emit_line

from fleet.cli import _config
from fleet.contracts.runner_tick import RunnerTick
from fleet.core import _test_hooks, names, queue, rolled, tick_report

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
    "fleet-research": "fleet.cli.research",
}

#: The agent's deadline: one minute inside the 40-minute ExecutionTimeLimit
#: both scheduled tasks carry, so the launcher reports a timeout in the
#: tick's log before Task Scheduler ends the tick without a word.
AGENT_WALL_SECONDS: Final[int] = 39 * 60

#: The exit status when no rolled tree could be extracted.
REFUSED_EXIT: Final[int] = 2

#: The console-script name of the agent whose ticks the queue records.
NODE_AGENT: Final = "fleet-node-agent"

#: The node agent's flag naming the node it serves, which the tick passes it
#: (:func:`fleet.cli.tick.plan_tick`).
NODE_FLAG: Final = "--node"

#: The verdict a refused launch records, before the refusal itself.
REFUSED_VERDICT: Final = "ran no runner code; claiming nothing: "

#: How the launcher's own log stream encodes, whatever code page started it.
#: It relays the agent's streams, and on 2026-09-29 (MCPs board task
#: 88b8fe61) a traceback holding U+FFFD met a cp1252 stdout and logging
#: printed ``--- Logging error ---`` in place of it. UTF-8 is what the tick
#: decodes this process's output as; backslashreplace keeps a character no
#: encoding could write as its escape rather than failing the line.
LOG_ENCODING: Final = "utf-8"
LOG_ERRORS: Final = "backslashreplace"


def encode_log_stream(stream: TextIO) -> None:
    """Make the stream the launcher logs to write any character it is given.

    Args:
        stream: The launcher's standard output, which
            :func:`platform_core.logging.setup_logging` logs to.

    Raises:
        TypeError: ``FLEET_LOG_STREAM_UNCONFIGURABLE`` when the stream is not
            a text wrapper over bytes, so its encoding cannot be set.
    """
    if not isinstance(stream, io.TextIOWrapper):
        raise TypeError(
            f"FLEET_LOG_STREAM_UNCONFIGURABLE: the launcher's output is a "
            f"{type(stream).__name__}, whose encoding cannot be set to {LOG_ENCODING}"
        )
    stream.reconfigure(encoding=LOG_ENCODING, errors=LOG_ERRORS)


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


def served_node(agent: str, passthrough: Sequence[str]) -> str | None:
    """The node a launch would have served a tick for, if it is a tick.

    Args:
        agent: The launcher's ``--agent``.
        passthrough: The agent's own command line.

    Returns:
        The node the node agent's ``--node`` names, or None for another
        agent and for an announce run.

    Raises:
        ValueError: When a node agent's command line carries a flag other
            than ``--node`` and ``--announce``, repeats one, or names no node,
            as the agent itself would refuse it.
    """
    if agent != NODE_AGENT or queue.ANNOUNCE_FLAG in passthrough:
        return None
    return cli_args.require_flag(cli_args.parse_single_flags(passthrough, (NODE_FLAG,)), NODE_FLAG)


def record_refused_tick(node: str, refusal: str, records: pathlib.Path) -> str:
    """Record a node runner's tick that no rolled agent could serve.

    Args:
        node: The node the runner serves.
        refusal: Why no rolled tree could be extracted.
        records: The checkout's records directory, the identity's cwd.

    Returns:
        When the queue stored the tick.

    Raises:
        AppError: When the queue's credentials are missing, or it cannot be
            reached or refuses the tick; the launch's refusal was logged
            first, so the tick's log still says why nothing ran.
    """
    agent, session_id = names.node_identity(node)
    tick = RunnerTick(
        node=node,
        tags=(),
        fits=(),
        load=None,
        claiming=False,
        verdict=f"{REFUSED_VERDICT}{refusal}",
    )
    return tick_report.record_tick(
        queue.load_credentials(),
        tick,
        identity=queue.identity_arguments(agent, session_id, str(records)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Extract the rolled tree and run one agent from it.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        The agent's exit status, or :data:`REFUSED_EXIT` when the rolled
        tree could not be extracted and nothing ran; a node runner's refused
        tick is recorded on the queue first (:func:`record_refused_tick`).

    Raises:
        ValueError: For a malformed command line.
        AppError: From :func:`record_refused_tick`.
    """
    parsed, passthrough = split_command_line(list(argv) if argv is not None else sys.argv[1:])
    agent = cli_args.require_flag(parsed, AGENT_FLAG)
    module = require_module(agent)
    node = served_node(agent, passthrough)
    repo_root = pathlib.Path(cli_args.require_flag(parsed, REPO_ROOT_FLAG))
    records = repo_root / pathlib.PurePosixPath(rolled.RECORDS_DIR)
    tree = rolled.extract_rolled_tree(repo_root)
    if isinstance(tree, str):
        _log.error("%s", tree)
        if node is not None:
            stored = record_refused_tick(node, tree, records)
            _log.info("fleet-roll: recorded %s's refused tick at %s", node, stored)
        return REFUSED_EXIT
    _log.info("fleet-roll: %s runs from %s at %s", module, rolled.ROLLED_REF, tree["commit"])
    if tree["reaped"]:
        _log.info("fleet-roll: removed what was left behind: %s", ", ".join(tree["reaped"]))
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
    configure_logging()
    raise SystemExit(run_cli(main, emit=emit_line, emit_error=emit_error_line))


def configure_logging() -> None:
    """Log to standard output as UTF-8, the launcher's one log stream.

    Raises:
        TypeError: ``FLEET_LOG_STREAM_UNCONFIGURABLE`` from
            :func:`encode_log_stream`.
    """
    encode_log_stream(sys.stdout)
    setup_logging(
        level=LogLevel.INFO,
        format_mode=LogFormat.TEXT,
        service_name="fleet-rolled",
        instance_id=None,
        extra_fields=None,
    )


__all__ = [
    "AGENT_FLAG",
    "AGENT_MODULES",
    "AGENT_WALL_SECONDS",
    "LOG_ENCODING",
    "LOG_ERRORS",
    "NODE_AGENT",
    "NODE_FLAG",
    "REFUSED_EXIT",
    "REFUSED_VERDICT",
    "REPO_ROOT_FLAG",
    "SEPARATOR",
    "configure_logging",
    "encode_log_stream",
    "entrypoint",
    "main",
    "record_refused_tick",
    "require_module",
    "served_node",
    "split_command_line",
]


# Without this, `python -m fleet.cli.rolled` imports the module, runs nothing
# and exits 0, and a tick that ran no agent would read as a clean one.
if __name__ == "__main__":
    entrypoint()
