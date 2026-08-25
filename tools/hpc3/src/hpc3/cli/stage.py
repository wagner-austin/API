"""CLI: place manifest-described files on the cluster, verified on both sides.

Usage:
    hpc3-stage --config hpc3.json --manifest runs/stage.json \\
        --source-dir runs/corpora --expect-from runs/file_ids.txt

Three checks, and they answer different questions. The local digest proves the
emitter produced the file the manifest names. The cluster-side digest proves
that file arrived intact. ``--expect-from`` proves the manifest names the bytes
the published work actually used -- which the first two cannot, because a
manifest emitted alongside its files always agrees with them.

``--expect-from`` is required, not optional. The failure it catches leaves no
trace: a corpus regenerated from the wrong source state stages clean, verifies
clean, trains to completion, and is comparable to nothing. A check that only
runs when someone remembers to ask for it is not protection against a failure
nobody notices.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.json_utils import load_json_str

from hpc3.cli import _config, _fatal, _test_hooks
from hpc3.contracts.provenance import format_provenance
from hpc3.contracts.stage import decode_stage_manifest
from hpc3.core import _test_hooks as core_hooks
from hpc3.core.expected import check_expected, read_expected_digests
from hpc3.core.stage import stage_manifest

_FLAGS = (_config.CONFIG_FLAG, "--manifest", "--source-dir", "--expect-from")


def main(argv: Sequence[str] | None = None) -> int:
    """Stage every file a manifest describes.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        Exit code 0 when every file was placed and verified on the cluster.

    Raises:
        ValueError: If a required flag is missing or an argument is unknown.
        JSONTypeError: If the manifest is not a valid stage manifest, or its
            provenance record is missing or empty.
        AppError: With ``STAGED_DIGEST_UNEXPECTED`` if a digest is absent from
            the published record, ``MANIFEST_FILE_MISSING`` if a file is
            absent, ``DIGEST_MISMATCH`` if bytes differ on either side, or
            ``REMOTE_COMMAND_FAILED`` if a remote command fails. Nothing is
            caught: a partial stage is not a success and must not be reported
            as one.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    host = _config.load_workspace(parsed)["host"]
    manifest_path = pathlib.Path(cli_args.require_flag(parsed, "--manifest"))
    source_dir = pathlib.Path(cli_args.require_flag(parsed, "--source-dir"))
    expect_path = pathlib.Path(cli_args.require_flag(parsed, "--expect-from"))

    raw = core_hooks.read_bytes(manifest_path).decode("utf-8")
    manifest = decode_stage_manifest(load_json_str(raw))

    # Identity before transport. Verifying that the wrong bytes arrived intact
    # costs a network round trip per file and answers the wrong question.
    check_expected(manifest, read_expected_digests(expect_path), source=expect_path)
    _test_hooks.emit(f"digests vouched for by {expect_path}")
    _test_hooks.emit(f"provenance {format_provenance(manifest['provenance'])}")

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
    raise SystemExit(_fatal.run(main))


__all__ = ["entrypoint", "main"]
