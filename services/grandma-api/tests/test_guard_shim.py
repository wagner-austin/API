"""Tests for this package's guard shim.

The shim is a bootstrap with nothing package-specific in it, so this test is
the same everywhere too, and the ``guard-shim-not-canonical`` rule keeps it
that way. The argument handling it delegates to is covered once, in
``monorepo_guards/tests/test_shim.py``.

What is left to check here is the thing that IS per-package: that this
package's shim reaches the shared implementation at all, by refusing a tree
that breaks a rule and passing an empty one. The predecessors of this file
accepted either outcome -- ``assert rc in (0, 2)`` and three other shapes
that cannot fail -- across seventy assertions in nineteen packages.

Whether the guard passes on THIS package's real files is not asked here.
``make lint`` asks it, running ``python -m scripts.guard`` with no arguments
before the tests start, and a case that asked it again ran the whole guard a
second time inside the suite, under the coverage tracer: 176 s on one worker
of clients/TankpitBot's fleet check on 2026-10-05 (MCPs board task 46934cd6).
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.guard import main


def test_a_tree_with_violations_fails(tmp_path: Path) -> None:
    """The other half of the pair. Without it, a shim that silently checked
    an empty tree would look exactly like a shim that works."""
    bad = tmp_path / "src" / "bad.py"
    bad.parent.mkdir(parents=True)
    banned = "An" + "y"
    bad.write_text(f"from typing import {banned}\nx: {banned} = 1\n", encoding="utf-8")
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
]
