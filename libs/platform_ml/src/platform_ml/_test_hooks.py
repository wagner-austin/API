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


class CreateTarballCallable(Protocol):
    """Protocol for create_tarball function signature."""

    def __call__(self, src_dir: Path, dest_file: Path, *, root_name: str) -> Path:
        """Create a tarball from src_dir to dest_file with given root_name."""
        ...


def _production_create_tarball(src_dir: Path, dest_file: Path, *, root_name: str) -> Path:
    """Production implementation - calls real create_tarball."""
    return _real_create_tarball(src_dir, dest_file, root_name=root_name)


class _Hooks:
    """Mutable container for internal test hooks."""

    create_tarball: CreateTarballCallable


# Global hooks instance
hooks = _Hooks()


def set_production_hooks() -> None:
    """Set all hooks to production implementations."""
    hooks.create_tarball = _production_create_tarball


def reset_hooks() -> None:
    """Reset hooks to production implementations."""
    set_production_hooks()


# Initialize with production hooks
set_production_hooks()


__all__ = [
    "TarballError",
    "hooks",
    "reset_hooks",
    "set_production_hooks",
]
