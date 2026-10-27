"""Inventory WRDS metadata: python -m hpc3.cli.wrds_catalog.

Requires --host, --remote-dir and --output. No data rows are downloaded.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.cli_boundary import run_cli

from hpc3.cli import _test_hooks
from hpc3.core.wrds_catalog import inventory


def main(argv: Sequence[str] | None = None) -> int:
    """Run an inventory and report counts without printing dataset contents.

    Args:
        argv: Command arguments or process arguments.

    Returns:
        Zero after a complete inventory.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    flags = cli_args.parse_single_flags(tokens, ("--host", "--remote-dir", "--output"))
    host = cli_args.require_flag(flags, "--host")
    remote = cli_args.require_flag(flags, "--remote-dir")
    output = pathlib.Path(cli_args.require_flag(flags, "--output"))
    entries = inventory(host, remote, output)
    files = [entry for entry in entries if entry["kind"] == "file"]
    readable = [entry for entry in files if entry["readable"]]
    denied = sum(entry["kind"] == "denied_directory" for entry in entries)
    _test_hooks.emit(f"WRDS files: {len(files)} visible, {len(readable)} readable")
    _test_hooks.emit(f"Readable logical bytes: {sum(entry['bytes'] for entry in readable)}")
    _test_hooks.emit(f"Inaccessible directories: {denied}; SQL catalog not inventoried")
    _test_hooks.emit(f"Metadata catalog: {output}; scheduler receipt: {output}.receipt.json")
    return 0


def entrypoint() -> None:
    """Run the command through the common CLI error boundary."""
    raise SystemExit(run_cli(main, emit=_test_hooks.emit, emit_error=_test_hooks.emit_error))


__all__ = ["entrypoint", "main"]


if __name__ == "__main__":
    entrypoint()
