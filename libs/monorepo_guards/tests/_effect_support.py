"""Building a throwaway package for the effect-seam and state-change rules.

Both rules read a whole package at once (its ``src``, ``scripts`` and
``tests``), so every test writes a small tree under ``tmp_path`` and runs the
real rule, or the real module function, over the files the guard would
collect for it.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from monorepo_guards.config import GuardConfig
from monorepo_guards.util import iter_py_files


def write(root: Path, relative: str, source: str) -> Path:
    """Write one dedented source file into the package.

    Args:
        root: The package root.
        relative: Its path under the root, e.g. ``src/pkg/_test_hooks.py``.
        source: Its contents, dedented before writing.

    Returns:
        The file's path.
    """
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def config_for(root: Path) -> GuardConfig:
    """Build the configuration a guard run over the package would use.

    Args:
        root: The package root.

    Returns:
        A configuration collecting ``src``, ``scripts`` and ``tests``.
    """
    return GuardConfig(
        root=root,
        monorepo_root=root,
        directories=("src", "scripts", "tests"),
        exclude_parts=(),
        forbid_pyi=True,
        allow_print_in_tests=False,
        dataclass_ban_segments=(),
    )


def files_of(root: Path) -> list[Path]:
    """Collect the files a guard run over the package would check.

    Args:
        root: The package root.

    Returns:
        Its ``src``, ``scripts`` and ``tests`` Python files.
    """
    return sorted(iter_py_files(config_for(root)))


__all__ = ["config_for", "files_of", "write"]
