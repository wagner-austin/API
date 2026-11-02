"""The seam that gives one launch attempt a scratch directory of its own.

A node runner bundles a claimed job's companions before its lease, and a
job is claimed again after an attempt that ended mid-launch. When the
bundles went in a directory named for the job alone, the next attempt met
the earlier one's files there and its removal raised ``ENOTEMPTY``, ending
pendragon's serving loop (fleet job e9009971, MCPs board task 2e3e69cf).
So each attempt is given a directory nothing else is given.
"""

from __future__ import annotations

import pathlib
import tempfile
from typing import Protocol


class FreshDirectoryProtocol(Protocol):
    """Creates a new, empty directory no other caller is given."""

    def __call__(self, parent: pathlib.Path, prefix: str) -> pathlib.Path:
        """Make the directory.

        Args:
            parent: The directory to create it in, made with its parents
                when absent.
            prefix: The start of its name.

        Returns:
            Its path.
        """


def _default_fresh_directory(parent: pathlib.Path, prefix: str) -> pathlib.Path:
    """Create a real directory as ``mkdtemp`` does, which no two calls share.

    Args:
        parent: The directory to create it in.
        prefix: The start of its name.

    Returns:
        Its path.
    """
    parent.mkdir(parents=True, exist_ok=True)
    return pathlib.Path(tempfile.mkdtemp(prefix=prefix, dir=parent))


fresh_directory: FreshDirectoryProtocol = _default_fresh_directory

__all__ = ["FreshDirectoryProtocol", "fresh_directory"]
