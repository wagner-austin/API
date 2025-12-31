"""Test hooks for guard script.

Production code uses real implementations; tests override these module-level
symbols to inject fakes without conditionals in core logic.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class IsDirProtocol(Protocol):
    """Protocol for checking if a path is a directory."""

    def __call__(self, path: Path) -> bool:
        """Check if path is a directory.

        Args:
            path: Path to check.

        Returns:
            True if path is a directory, False otherwise.
        """
        ...


def _real_is_dir(path: Path) -> bool:
    """Real implementation using Path.is_dir().

    Args:
        path: Path to check.

    Returns:
        True if path is a directory, False otherwise.
    """
    return path.is_dir()


is_dir: IsDirProtocol = _real_is_dir


__all__ = [
    "IsDirProtocol",
    "is_dir",
]
