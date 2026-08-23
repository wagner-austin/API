"""CLI: place manifest-described files on the cluster, verified on both sides.

Usage:
    hpc3-stage --config hpc3.json --manifest runs/stage.json --source-dir runs/corpora
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core.json_utils import load_json_str

from hpc3.cli import _argv, _config, _test_hooks
from hpc3.contracts.stage import decode_stage_manifest
from hpc3.core import _test_hooks as core_hooks
from hpc3.core.stage import stage_manifest

_FLAGS = (_config.CONFIG_FLAG, "--manifest", "--source-dir")


def main(argv: Sequence[str] | None = None) -> int:
    """Stage every file a manifest describes.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        Exit code 0 when every file was placed and verified on the cluster.

    Raises:
        ValueError: If a required flag is missing or an argument is unknown.
        JSONTypeError: If the manifest is not a valid stage manifest.
        AppError: If a file is missing, a digest does not match on either
            side, or a remote command fails. Nothing is caught: a partial
            stage is not a success and must not be reported as one.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = _argv.parse_single_flags(tokens, _FLAGS)
    host = _config.load_workspace(parsed)["host"]
    manifest_path = pathlib.Path(_argv.require_flag(parsed, "--manifest"))
    source_dir = pathlib.Path(_argv.require_flag(parsed, "--source-dir"))

    raw = core_hooks.read_bytes(manifest_path).decode("utf-8")
    manifest = decode_stage_manifest(load_json_str(raw))

    placed = stage_manifest(host, source_dir, manifest)
    for remote_path in placed:
        _test_hooks.emit(f"staged {remote_path}")
    _test_hooks.emit(f"verified {len(placed)} file(s) on {host}:{manifest['destination']}")
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(main(None))


__all__ = ["entrypoint", "main"]
