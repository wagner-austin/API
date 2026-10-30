"""Run one evaluator-pinned proposal program in isolation and record its admission."""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.cli_boundary import run_cli
from platform_core.json_utils import dump_json_str
from platform_core.report_line import emit_error_line, emit_line

from fleet.contracts.research import require_digest
from fleet.core import research_control, research_proposal

_FLAGS = ("--root", "--digest")


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve a private proposer specification by digest and run it once.

    Args:
        argv: Explicit ``--root`` and ``--digest`` or process arguments.

    Returns:
        Zero once the admission controller recorded the attempt, admitted or refused.

    Raises:
        ValueError: Invalid digest, specification drift, isolation refusal or
            controller failure.
        OSError: Missing private configuration or unavailable process.
    """
    flags = cli_args.parse_single_flags(list(sys.argv[1:] if argv is None else argv), _FLAGS)
    digest = require_digest(cli_args.require_flag(flags, "--digest"))
    root = pathlib.Path(cli_args.require_flag(flags, "--root"))
    document = research_control.read_pinned(root, digest, "proposal.json")
    directory = root / digest
    result = research_proposal.propose(document, directory, directory / "admission.jsonl")
    emit_line(dump_json_str(result))
    return 0


def entrypoint() -> None:
    """Run the shared CLI boundary with metadata-only output."""
    raise SystemExit(run_cli(main, emit=emit_line, emit_error=emit_error_line))


__all__ = ["entrypoint", "main"]

if __name__ == "__main__":
    entrypoint()
