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
from fleet.core import (
    _test_hooks,
    file_turn,
    research_control,
    research_handover,
    research_hpc,
    research_wrds,
    staging,
)

_FLAGS = (
    "--root",
    "--backend",
    "--digest",
    "--attempt",
    "--tenant",
    "--task",
    "--env-file",
    "--handover",
)


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
        directory / "poll.turn",
        holder=f"research-{request['backend'].value}",
        bounds=file_turn.TurnBounds(
            what="native research poll",
            waited_for="prior controller poll",
        ),
    ):
        marker = directory / "handover.sha256"
        handover = flags.get("--handover")
        if handover is not None and _test_hooks.file_exists(marker):
            raise ValueError("RESEARCH_HANDOVER_EXISTS: this request already has a handover")
        if handover is not None or _test_hooks.file_exists(marker):
            digest = handover if handover is not None else _test_hooks.read_text(marker).strip()
            result = research_handover.poll(root, request, document, digest, task, environment)
            if handover is not None:
                research_handover.record(marker, digest)
        else:
            poll = (
                research_hpc.poll
                if request["backend"] is ResearchBackend.HPC3
                else research_wrds.poll
            )
            result = poll(document, directory / "controller.jsonl", task, environment)
            result["handover_sha256"] = None
        result["controller_python"] = sys.executable
        result["controller_python_sha256"] = staging.digest(
            _test_hooks.read_bytes(pathlib.Path(sys.executable))
        )
        emit_line(dump_json_str(encode_observation(decode_observation(result))))
    return 0


def entrypoint() -> None:
    """Run the shared CLI boundary with metadata-only output."""
    raise SystemExit(run_cli(main, emit=emit_line, emit_error=emit_error_line))


__all__ = ["entrypoint", "main"]

if __name__ == "__main__":
    entrypoint()
