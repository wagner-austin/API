"""One research queue tick under the existing rolled fleet scheduler."""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from board_watch import config as board_config
from platform_core import cli_args
from platform_core.cli_boundary import run_cli
from platform_core.json_utils import load_json_str
from platform_core.report_line import emit_error_line, emit_line

from fleet.cli import _config, research_queue
from fleet.contracts.dispatch import encode_job_line
from fleet.contracts.research_worker import decode_worker
from fleet.core import _test_hooks, queue

_FLAGS = (_config.CONFIG_FLAG, _config.RECORDS_FLAG, "--agent", "--session", "--repo-root")


def main(argv: Sequence[str] | None = None) -> int:
    """Serve one request, using the same queue identity and resource records as fleet.

    Args:
        argv: Explicit arguments or process arguments.

    Returns:
        Zero after an idle tick or a recorded observation.

    Raises:
        ValueError: Invalid controller configuration.
        AppError: Shared queue, credential or resource failure.
    """
    tokens = list(sys.argv[1:] if argv is None else argv)
    announce = queue.ANNOUNCE_FLAG in tokens
    if announce:
        tokens.remove(queue.ANNOUNCE_FLAG)
    flags = cli_args.parse_single_flags(tokens, _FLAGS)
    workspace = _config.load_workspace(flags)
    root = pathlib.Path(cli_args.require_flag(flags, "--repo-root"))
    identity = queue.identity_arguments(
        cli_args.require_flag(flags, "--agent"),
        cli_args.require_flag(flags, "--session"),
        str(root),
    )
    if announce:
        emit_line(
            queue.announce(
                board_config.load_credentials(),
                machine="linux:" + _test_hooks.hostname(),
                body="Research controller: bounded native polling through shared fleet leases.",
                identity=identity,
            )
        )
        return 0
    worker = decode_worker(
        load_json_str(_test_hooks.read_text(root / "tools/fleet/runs/research.json"))
    )
    result = research_queue.tick(queue.load_credentials(), identity, worker, workspace.leases)
    emit_line(
        "research tick: idle or controller resource held"
        if result is None
        else encode_job_line(result)
    )
    return 0


def entrypoint() -> None:
    """Use the fleet's existing named error boundary."""
    raise SystemExit(run_cli(main, emit=emit_line, emit_error=emit_error_line))


__all__ = ["entrypoint", "main"]

if __name__ == "__main__":
    entrypoint()
