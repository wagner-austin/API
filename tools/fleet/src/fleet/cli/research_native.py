"""One bounded native controller tick on the machine holding the cluster credentials."""

from __future__ import annotations

import pathlib
import sys
import uuid
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.cli_boundary import run_cli
from platform_core.env_assignments import parse_env_assignments
from platform_core.json_utils import dump_json_str
from platform_core.report_line import emit_error_line, emit_line

from fleet.contracts.research import ResearchBackend, decode_research_request
from fleet.contracts.research_worker import decode_observation, encode_observation
from fleet.core import _test_hooks, file_turn, research_control, research_hpc, research_wrds

_FLAGS = ("--root", "--backend", "--digest", "--attempt", "--tenant", "--task", "--env-file")


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve a private frozen reference and run its maintained native controller.

    Args:
        argv: Explicit request metadata or process arguments.

    Returns:
        Zero when the controller returned a verified observation.

    Raises:
        ValueError: Invalid metadata, input drift or controller failure.
        OSError: Missing private configuration or unavailable process.
    """
    flags = cli_args.parse_single_flags(list(sys.argv[1:] if argv is None else argv), _FLAGS)
    request = decode_research_request(
        {
            "backend": cli_args.require_flag(flags, "--backend"),
            "specSha256": cli_args.require_flag(flags, "--digest"),
            "attempt": cli_args.require_flag(flags, "--attempt"),
        }
    )
    task = str(uuid.UUID(cli_args.require_flag(flags, "--task")))
    root = pathlib.Path(cli_args.require_flag(flags, "--root"))
    tenant = cli_args.require_flag(flags, "--tenant")
    document = research_control.read_specification(root, request, tenant)
    env_file = pathlib.Path(cli_args.require_flag(flags, "--env-file"))
    environment = parse_env_assignments(_test_hooks.read_text(env_file), source=str(env_file))
    directory = root / request["spec_sha256"]
    with file_turn.held(
        directory / "poll.lock",
        holder=f"research-{request['backend'].value}",
        bounds=file_turn.TurnBounds(
            what="native research poll",
            waited_for="prior controller poll",
            stale_seconds=1200,
            poll_seconds=1,
        ),
    ):
        poll = (
            research_hpc.poll if request["backend"] is ResearchBackend.HPC3 else research_wrds.poll
        )
        result = poll(document, directory / "controller.jsonl", task, environment)
        emit_line(dump_json_str(encode_observation(decode_observation(result))))
    return 0


def entrypoint() -> None:
    """Run the shared CLI boundary with metadata-only output."""
    raise SystemExit(run_cli(main, emit=emit_line, emit_error=emit_error_line))


__all__ = ["entrypoint", "main"]

if __name__ == "__main__":
    entrypoint()
