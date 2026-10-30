"""Run one evaluator-pinned proposal program in isolation and record its admission."""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence
from uuid import UUID

from board_watch import config as board_config
from platform_core import cli_args
from platform_core.board import post_to_task, register_service_session, service_identity
from platform_core.cli_boundary import run_cli
from platform_core.json_utils import dump_json_str
from platform_core.report_line import emit_error_line, emit_line

from fleet.contracts.research import require_digest
from fleet.core import _test_hooks, research_control, research_proposal

_FLAGS = ("--root", "--digest", "--report-task")


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve a private proposer specification by digest and run it once.

    Args:
        argv: Explicit root and digest, optionally a task to receive the measured
            admission outcome, or process arguments.

    Returns:
        Zero once the admission controller recorded the attempt, admitted or refused.

    Raises:
        ValueError: Invalid digest, specification drift, isolation refusal or
            controller failure.
        OSError: Missing private configuration or unavailable process.
    """
    flags = cli_args.parse_single_flags(list(sys.argv[1:] if argv is None else argv), _FLAGS)
    target = flags.get("--report-task")
    task = None if target is None else str(UUID(target))
    digest = require_digest(cli_args.require_flag(flags, "--digest"))
    root = pathlib.Path(cli_args.require_flag(flags, "--root"))
    document = research_control.read_pinned(root, digest, "proposal.json")
    directory = root / digest
    result = research_proposal.propose(document, directory, directory / "admission.jsonl")
    if task is not None:
        identity = service_identity(
            agent="fleet-research-proposer", service="corvis:fleet:research-proposer",
            cwd="service://fleet-research-proposer",
        )
        credentials = board_config.load_credentials()
        register_service_session(
            _test_hooks.http_post, credentials, identity, harness="fleet-research-proposer",
            purpose="records isolated proposal handoffs measured by the admission controller",
        )
        post_to_task(
            _test_hooks.http_post, credentials, identity, task_id=task, kind="note",
            body="RESEARCH PROPOSAL LOAD-REPORT\n" + dump_json_str({
                "spec_sha256": digest, "host": document["host"], "image": document["image"],
                "result": result,
            }),
        )
    emit_line(dump_json_str(result))
    return 0


def entrypoint() -> None:
    """Run the shared CLI boundary with metadata-only output."""
    raise SystemExit(run_cli(main, emit=emit_line, emit_error=emit_error_line))


__all__ = ["entrypoint", "main"]

if __name__ == "__main__":
    entrypoint()
