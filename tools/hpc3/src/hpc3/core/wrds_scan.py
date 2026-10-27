"""Standalone metadata probe copied to a WRDS compute node.

Only directory entries, stat metadata and effective read permissions are
examined. No dataset is opened. Directory aliases are recorded and pruned,
so archived mounts and symlink cycles cannot multiply a traversal forever.
The wire format is a JSON array per line: kind, path, bytes, readable, target.
The remote archive includes the shared JSON serializer and needs no installation.
"""

from __future__ import annotations

import argparse
import errno
import os
import pathlib
import stat
from collections.abc import Generator, Sequence

from platform_core.json_utils import dump_json_str


class ScanArguments(argparse.Namespace):
    """Typed arguments at the standalone process boundary."""

    root: str
    output: str


def permission_error(error: OSError, denied: list[str]) -> None:
    """Record denied directories; propagate every other traversal failure.

    Args:
        error: Error reported by os.walk.
        denied: Explicit omissions to include in the inventory.

    Raises:
        OSError: For failures other than denied access.
    """
    if error.errno != errno.EACCES or error.filename is None:
        raise error
    denied.append(error.filename)


def scan(root: pathlib.Path) -> Generator[tuple[str, str, int, bool, str], None, None]:
    """Yield filesystem metadata, including inaccessible paths and aliases.

    Args:
        root: Directory to inventory, including all its historical trees.

    Yields:
        Wire records. Sizes are logical file sizes, not physical disk usage.

    Raises:
        NotADirectoryError: If root is not an accessible directory.
        OSError: On unexpected filesystem failures rather than a partial success.
    """
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    visited: set[str] = set()
    denied: list[str] = []

    def onerror(error: OSError) -> None:
        permission_error(error, denied)

    for directory, children, files in os.walk(root, followlinks=True, onerror=onerror):
        resolved = os.path.realpath(directory)
        if resolved in visited:
            children.clear()
            yield ("alias", directory, 0, False, resolved)
            continue
        visited.add(resolved)
        yield ("directory", directory, 0, True, resolved)
        # Prefer current trees to .new/.old aliases when both reach the same directory.
        children.sort()
        for name in sorted(files):
            path = pathlib.Path(directory) / name
            target = os.path.realpath(path)
            if path.is_symlink() and not path.exists():
                # exists() cannot distinguish a missing target from denied traversal.
                yield ("unresolved_link", str(path), 0, False, target)
                continue
            metadata = path.stat()
            if stat.S_ISREG(metadata.st_mode):
                yield ("file", str(path), metadata.st_size, os.access(path, os.R_OK), target)
            else:
                yield ("special", str(path), 0, False, target)
    for path in denied:
        yield ("denied_directory", path, 0, False, os.path.realpath(path))


def main(argv: Sequence[str] | None = None) -> None:
    """Write an exclusive metadata artifact with an explicit completion marker.

    Args:
        argv: Arguments with required root and output file.

    Raises:
        OSError: If scanning or writing fails. No completion marker is written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = ScanArguments()
    parser.parse_args(argv, namespace=args)
    root = pathlib.Path(args.root)
    output = pathlib.Path(args.output)
    with output.open("x", encoding="utf-8") as handle:
        for record in scan(root):
            handle.write(dump_json_str(record) + "\n")
        handle.write(dump_json_str(("complete", str(root), 0, False, "")) + "\n")


__all__ = ["ScanArguments", "main", "permission_error", "scan"]


if __name__ == "__main__":
    main()
