"""Invariants about the package's shape, asserted rather than agreed.

WHY THESE ARE TESTS AND NOT GUARD RULES. ``scripts/guard.py`` is a
byte-identical shim in all forty-one packages, enforced by
``guard-shim-not-canonical``; every rule it runs lives in
``libs/monorepo_guards`` and applies to the whole monorepo. The invariants
below are about THIS package's layering, so promoting them to a monorepo
rule would impose one bridge's design on forty other packages. They belong
here, where they are checked by the same ``make check`` and cost nobody
else anything.
"""

from __future__ import annotations

import pathlib
from typing import Final

from hpc_wake import _test_hooks, pending, settling

SRC: Final = pathlib.Path(__file__).resolve().parent.parent / "src" / "hpc_wake"


def source_of(name: str) -> str:
    """Read one module's source text.

    Args:
        name: Module file name, including the extension.

    Returns:
        The file's text.
    """
    return (SRC / name).read_text(encoding="utf-8")


def test_the_pending_store_adds_no_io_seam_of_its_own() -> None:
    """It must use hpc3's file seam, not grow a parallel one.

    A second seam would be a second thing tests must remember to rebind,
    and the first one anybody forgets writes to the developer's real disk
    during a test run.
    """
    text = source_of("pending.py")
    assert "hpc3.core import _test_hooks" in text
    # Every I/O call must be QUALIFIED BY THE SEAM. A blanket ban on the
    # verbs was the first predicate here and it flagged
    # ``hpc3_hooks.write_text(...)`` -- the very call it exists to require.
    # The rule is not "never say write_text", it is "never say it to
    # anything but the hooks module", so that is what it now checks.
    for verb in ("write_text(", "read_bytes(", "file_exists("):
        for line in text.splitlines():
            if verb in line and not line.lstrip().startswith(("#", "*", '"')):
                assert f"hpc3_hooks.{verb}" in line, line
    assert "open(" not in text


def test_the_settling_policy_touches_nothing_outside_itself() -> None:
    """Purity is the reason it can be exercised over simulated time.

    The moment this module reads a clock or a file, its tests need a fake
    and the spacings they can express are limited to what a fake can be
    talked into producing.
    """
    text = source_of("settling.py")
    for forbidden in ("_test_hooks", "datetime", "pathlib", "open(", "post_to_task"):
        assert forbidden not in text, forbidden


def test_every_hook_is_restored_by_reset() -> None:
    """A hook that reset forgets leaks a fake into the next test.

    Compares the module's declared hook names against what ``reset_hooks``
    actually rebinds, so adding a seam without adding it to reset fails
    here rather than as an unrelated test's mysterious flake.
    """
    # Read from ``__all__``, which is the module's own declared contract,
    # rather than from ``vars()``. The first version of this walked ``vars``
    # for anything callable and not underscored, which swept up every
    # IMPORTED name -- ``urllib_mcp_post``, ``datetime`` -- and demanded
    # reset rebind things it does not own. Hooks are the lowercase names;
    # the Protocols beside them are CamelCase.
    declared = {name for name in _test_hooks.__all__ if name.islower() and name != "reset_hooks"}
    rebound = set(_test_hooks.reset_hooks.__code__.co_names)
    assert declared <= rebound, declared - rebound
    assert "now_epoch" in declared, "the settling clock must be a declared, resettable hook"


def test_the_announcement_partition_has_exactly_one_definition() -> None:
    """Settling and announcing must agree on what a group is.

    Two copies of "project, and submitter-or-empty" would be two
    definitions of one partition, free to drift the first time either
    changes -- and the symptom would be a group that settled together
    posting as two.
    """
    assert "group_key" in source_of("settling.py") or "group_key" in source_of("cycle.py")
    assert "def group_key" not in source_of("settling.py")
    assert "def group_key" not in source_of("cycle.py")
    assert "def group_key" in source_of("announce.py")


def test_the_public_surface_of_each_new_module_is_declared() -> None:
    """``__all__`` is the package's stated contract with its siblings."""
    assert "PendingClosure" in pending.__all__
    assert "partition_ripe" in settling.__all__
