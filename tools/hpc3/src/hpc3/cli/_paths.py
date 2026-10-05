"""Where the committed registry and the research index live.

Shared by ``hpc3-research-index`` and ``hpc3-register``, the two commands that
rewrite the index's generated table, so they cannot regenerate it from two
different directories. Both paths derive from one hook,
:data:`hpc3.cli._test_hooks.monorepo_root`, which a test points at a copy of
the tree.
"""

from __future__ import annotations

import pathlib

from hpc3.cli import _test_hooks


def runs_directory() -> pathlib.Path:
    """Locate the workspace documents.

    Returns:
        The ``runs`` directory of this package.
    """
    return _test_hooks.monorepo_root() / "tools" / "hpc3" / "runs"


def index_path() -> pathlib.Path:
    """Locate the research index.

    Returns:
        ``docs/RESEARCH.md`` at the monorepo root. It names work in other
        repositories, so it lives above the tool that submits some of it.
    """
    return _test_hooks.monorepo_root() / "docs" / "RESEARCH.md"


__all__ = ["index_path", "runs_directory"]
