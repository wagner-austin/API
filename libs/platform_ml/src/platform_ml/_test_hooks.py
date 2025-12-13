"""Internal test hooks for platform_ml.

These hooks allow testing internal boundaries like tarball operations
without mocking. Tests inject fakes, production uses real implementations.

For external dependencies (wandb), see testing.py which provides public
Protocol-based hooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .tarball import TarballError
from .tarball import create_tarball as _real_create_tarball


class _CreateTarballProtocol(Protocol):
    """Protocol for create_tarball function signature."""

    def __call__(self, src_dir: Path, dest_file: Path, *, root_name: str) -> Path: ...


def _default_create_tarball(src_dir: Path, dest_file: Path, *, root_name: str) -> Path:
    """Production implementation - calls real create_tarball."""
    return _real_create_tarball(src_dir, dest_file, root_name=root_name)


# Module-level hooks following platform_core pattern
create_tarball: _CreateTarballProtocol = _default_create_tarball


__all__ = [
    "TarballError",
    "create_tarball",
]
