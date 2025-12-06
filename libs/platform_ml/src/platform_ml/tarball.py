from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Final


class TarballError(Exception):
    """Base error for tarball operations."""


def create_tarball(src_dir: Path, dest_file: Path, *, root_name: str) -> Path:
    """Create a gzipped tarball of `src_dir` with a single root directory `root_name`.

    The archive layout will be:
      <root_name>/ ... contents of src_dir ...
    """
    if not src_dir.exists() or not src_dir.is_dir():
        raise TarballError("source directory does not exist or is not a directory")
    parent = dest_file.parent
    parent.mkdir(parents=True, exist_ok=True)
    root: Final[str] = root_name.strip()
    if root == "":
        raise TarballError("root_name must be non-empty")
    with tarfile.open(dest_file.as_posix(), mode="w:gz") as tf:
        for p in src_dir.rglob("*"):
            arcname = Path(root) / p.relative_to(src_dir)
            tf.add(p.as_posix(), arcname=arcname.as_posix(), recursive=False)
    return dest_file


def extract_tarball(src_file: Path, dest_dir: Path, *, expected_root: str) -> Path:
    """Extract a gzipped tarball into `dest_dir` and return the root directory path.

    Validates that all members reside under `expected_root` and that a single top-level
    directory with that exact name exists in the archive.
    """
    if not src_file.exists() or not src_file.is_file():
        raise TarballError("source tarball does not exist or is not a file")
    dest_dir.mkdir(parents=True, exist_ok=True)
    root: Final[str] = expected_root.strip()
    if root == "":
        raise TarballError("expected_root must be non-empty")
    with tarfile.open(src_file.as_posix(), mode="r:gz") as tf:
        names = tf.getnames()
        # Validate all paths start with the expected root
        for n in names:
            if not n or n.startswith("/"):
                raise TarballError("archive contains absolute or empty path entries")
            if not (n == root or n.startswith(root + "/")):
                raise TarballError("archive contains entries outside expected_root")
        tf.extractall(path=dest_dir.as_posix())
    out_root = dest_dir / root
    if not out_root.exists() or not out_root.is_dir():
        raise TarballError("extraction did not produce the expected root directory")
    return out_root


__all__ = ["TarballError", "create_tarball", "extract_tarball"]
