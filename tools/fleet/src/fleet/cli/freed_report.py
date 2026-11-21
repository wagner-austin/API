"""CLI: post what retires freed on each node since an instant on a board task.

Usage:
    python -m fleet.cli.freed_report --config registry \\
        --since 1795300000 --task 1526ece1-497d-4e67-8b48-4aba8f789c40

Run on the hub whose runners retire the fleet's runs, with the credentials
its pump hands every bridge (``FLEET_MCP_API_KEY`` and ``CORVIS_TENANT_ID``).
It totals the freed record beside the retire record
(:func:`fleet.core.retire_owed.freed_path`) per node from ``--since`` on
(:mod:`fleet.core.freed_report`), prints the line, and posts it as a note on
``--task`` under the dispatch identity of the hub runner on the machine it
runs on (:func:`fleet.cli.tick.runner_identity`), the writer that recorded
those lines: so the bytes on the board are the program's own count, never a
session's restatement of it (MCPs board task 1526ece1).
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence
from typing import Final

from platform_core import cli_args
from platform_core.cli_boundary import run_cli
from platform_core.logging import LogFormat, LogLevel, get_logger, setup_logging
from platform_core.mcp_client import call_mcp_tool
from platform_core.report_line import emit_error_line, emit_line

from fleet.cli import _config
from fleet.cli.tick import runner_identity
from fleet.contracts.dispatch import DispatchLane
from fleet.core import _test_hooks, freed_report, queue, records, retire_owed

_log = get_logger(__name__)

SINCE_FLAG: Final = "--since"
TASK_FLAG: Final = "--task"

_FLAGS: Final = (_config.CONFIG_FLAG, _config.RECORDS_FLAG, SINCE_FLAG, TASK_FLAG)


def main(argv: Sequence[str] | None = None) -> int:
    """Total the freed record per node and post it on a task.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 once the note is posted.

    Raises:
        ValueError: When a flag is unknown, repeated or missing its value,
            or ``--since`` is not whole seconds since the epoch.
        AppError: ``QUEUE_CREDENTIALS_MISSING`` from
            :func:`fleet.core.queue.load_credentials`, and any refusal of the
            board's.
        JSONTypeError: When a line of the freed record does not decode.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    loaded = _config.load_workspace(parsed)
    since_text = cli_args.require_flag(parsed, SINCE_FLAG)
    if not since_text.isdigit():
        raise ValueError(f"{SINCE_FLAG} must be whole seconds since the epoch, not {since_text!r}")
    task = cli_args.require_flag(parsed, TASK_FLAG)
    since = int(since_text)
    source = retire_owed.freed_path(loaded.retires)
    totals = freed_report.node_totals(
        records.read_freed(source), since_unix=since, nodes=tuple(loaded.workspace["nodes"])
    )
    line = freed_report.report_line(totals, since_unix=since, source=str(source))
    credentials = queue.load_credentials()
    label, session = runner_identity(DispatchLane.HUB, _test_hooks.hostname())
    identity = queue.identity_arguments(label, session, str(pathlib.Path.cwd()))
    call_mcp_tool(
        _test_hooks.http_post,
        credentials,
        "task_post",
        {**identity, "taskId": task, "kind": "note", "body": line},
    )
    _log.info("%s posted on %s as %s", line, task, label)
    return 0


def entrypoint() -> None:
    """Module entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level=LogLevel.INFO,
        format_mode=LogFormat.TEXT,
        service_name="fleet-freed-report",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(run_cli(main, emit=emit_line, emit_error=emit_error_line))


__all__ = ["SINCE_FLAG", "TASK_FLAG", "entrypoint", "main"]


# Without this, `python -m fleet.cli.freed_report` imports the module, posts
# nothing and exits 0, which reads as a report the task never received.
if __name__ == "__main__":
    entrypoint()
