"""Deliver a bounded private bundle once through the configured SSH route."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from platform_core import cli_args
from platform_core.cli_boundary import run_cli
from platform_core.json_utils import load_json_str
from platform_core.stage_manifest import decode_stage_manifest

from hpc3.cli import _config, _test_hooks
from hpc3.core import _test_hooks as core_hooks
from hpc3.core.private_delivery import PRIVATE_DELIVERY_MAX_BYTES, deliver_private_bundle


def main(argv: Sequence[str] | None = None) -> int:
    """Verify public metadata and deliver stdin bytes to a new private directory.

    Args:
        argv: Config and manifest flags; secret bytes must only arrive on stdin.

    Returns:
        Zero only after verified exclusive delivery.

    Raises:
        ValueError: Invalid arguments, manifest or payload.
        AppError: Unverified delivery; do not retry a consumed grant.
        OSError: Local input or transport failure.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, (_config.CONFIG_FLAG, "--manifest"))
    host = _config.load_workspace_connection(parsed)["host"]
    path = Path(cli_args.require_flag(parsed, "--manifest"))
    manifest = decode_stage_manifest(load_json_str(core_hooks.read_bytes(path).decode("utf-8")))
    payload = _test_hooks.read_stdin(PRIVATE_DELIVERY_MAX_BYTES + 1)
    deliver_private_bundle(host, manifest, payload)
    _test_hooks.emit("private delivery verified")
    return 0


def entrypoint() -> None:
    """Run the command with the shared CLI refusal boundary."""
    raise SystemExit(run_cli(main, emit=_test_hooks.emit, emit_error=_test_hooks.emit_error))


__all__ = ["entrypoint", "main"]


if __name__ == "__main__":
    entrypoint()
