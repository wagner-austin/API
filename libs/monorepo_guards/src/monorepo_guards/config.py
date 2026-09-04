from __future__ import annotations

from pathlib import Path
from typing import NamedTuple


class GuardConfig(NamedTuple):
    """What one guard run checks, and the tree it may look across to do it.

    Attributes:
        root: The package being checked. Every file the run collects lives
            under here.
        monorepo_root: The whole repository. A rule needs this when the thing
            it checks a package against is DECLARED in another package --
            ``LiteralSetRule`` reads its accepted set from the module that
            owns it, which for a set shared through a library is never in the
            package being checked.
        directories: Directory names under ``root`` to collect files from.
        exclude_parts: Path segments that disqualify a file.
        forbid_pyi: Whether a ``.pyi`` file is itself a violation.
        allow_print_in_tests: Whether tests may call ``print``.
        dataclass_ban_segments: Path segments under which dataclasses are
            refused.
    """

    root: Path
    monorepo_root: Path
    directories: tuple[str, ...]
    exclude_parts: tuple[str, ...]
    forbid_pyi: bool
    allow_print_in_tests: bool
    dataclass_ban_segments: tuple[tuple[str, ...], ...]


__all__ = ["GuardConfig"]
