"""The exact text every package's ``scripts/guard.py`` must contain.

One string, used two ways: it is what the ``guard-shim-not-canonical`` rule
compares each package's shim against, and it is what generated those shims
in the first place. A rule that merely described the shape it wanted would
be a second definition to keep in step with the first.

Why a whole file must be identical rather than merely similar: the previous
version was forty-one hand-maintained copies of the same sixty lines, and
they drifted into twenty-one distinct variants -- differing in how they
found the monorepo root, whether their plumbing routed through a
``_test_hooks`` module, and what their tests were willing to accept as a
pass. Byte equality is the only comparison with no room to argue.
"""

from __future__ import annotations

CANONICAL_GUARD_SHIM = '''"""Run this package's guard checks.

A bootstrap, not an implementation. Every guard rule -- and the argument
handling around it -- lives in ``libs/monorepo_guards``. This file exists
only because that package is a dependency of four of the forty-one packages
here and so cannot simply be imported by the other thirty-seven. It puts
``monorepo_guards`` on the path and hands over.

Invoked as ``python -m scripts.guard`` from the package directory, which is
the single form every Makefile uses. Running this file BY PATH instead puts
``scripts/`` on ``sys.path[0]`` rather than the package root, which is a
different program: it can only find an INSTALLED top-level ``scripts``.

Byte-identical in all forty-one packages, enforced by the
``guard-shim-not-canonical`` rule. Generated from
``monorepo_guards.guard_shim_template``; edit that, not this.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class _RunGuard(Protocol):
    def __call__(self, argv: Sequence[str] | None, *, project_root: Path) -> int: ...


def main(argv: Sequence[str] | None = None) -> int:
    """Run every guard rule against this package.

    Args:
        argv: Arguments excluding the program name, or None to read the
            process arguments.

    Returns:
        0 when no violations were found, 2 when there were.
    """
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[3] / "libs" / "monorepo_guards" / "src"))
    module = __import__("monorepo_guards.shim", fromlist=["run_guard"])
    run_guard: _RunGuard = module.run_guard
    return run_guard(argv, project_root=here.parents[1])


if __name__ == "__main__":
    raise SystemExit(main())
'''
"""Verbatim contents of every ``scripts/guard.py``, newline-terminated."""


CANONICAL_GUARD_SHIM_TEST = '''"""Tests for this package's guard shim.

The shim is a bootstrap with nothing package-specific in it, so this test is
the same everywhere too, and the ``guard-shim-not-canonical`` rule keeps it
that way. The argument handling it delegates to is covered once, in
``monorepo_guards/tests/test_shim.py``.

What is left to check here is the thing that IS per-package: that this
package's shim reaches the shared implementation at all, and that the guard
actually passes on this package. The predecessors of this file accepted
either outcome -- ``assert rc in (0, 2)`` and three other shapes that cannot
fail -- across seventy assertions in nineteen packages.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.guard import main


def test_this_package_passes_its_own_guard() -> None:
    """No --root, so this runs every rule over this package's real files.

    Deliberately not an empty temp directory. A shim pointed at an empty tree
    returns 0 without any rule having anything to look at, which is
    indistinguishable from a shim that works -- and it leaves the rules'
    file-skipping branches unexecuted, which is how this was first noticed.
    """
    assert main([]) == 0


def test_a_tree_with_violations_fails(tmp_path: Path) -> None:
    """The other half of the pair. Without it, a shim that silently checked
    an empty tree would look exactly like a shim that works."""
    bad = tmp_path / "src" / "bad.py"
    bad.parent.mkdir(parents=True)
    banned = "An" + "y"
    bad.write_text(f"from typing import {banned}\\nx: {banned} = 1\\n", encoding="utf-8")
    assert main(["--root", str(tmp_path)]) == 2


def test_running_as_a_module_exits_with_the_guard_code(tmp_path: Path) -> None:
    """`python -m scripts.guard` is the only invocation any Makefile uses, so
    it is the one the suite has to exercise."""
    original = list(sys.argv)
    sys.argv[:] = ["prog", "--root", str(tmp_path)]
    sys.modules.pop("scripts.guard", None)
    try:
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("scripts.guard", run_name="__main__")
        assert excinfo.value.code == 0
    finally:
        sys.argv[:] = original


__all__ = [
    "test_a_tree_with_violations_fails",
    "test_running_as_a_module_exits_with_the_guard_code",
    "test_this_package_passes_its_own_guard",
]
'''
"""Verbatim contents of every ``tests/test_guard_shim.py``."""


__all__ = ["CANONICAL_GUARD_SHIM", "CANONICAL_GUARD_SHIM_TEST"]
