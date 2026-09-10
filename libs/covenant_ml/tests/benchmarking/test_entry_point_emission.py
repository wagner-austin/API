"""Every benchmark entry point emits its record, checked mechanically.

WHY THIS IS A TEST AND NOT A GUARD RULE. Board task ``6d5536cc`` left the
placement to whoever took it. ``monorepo_guards``' ``run_record_rules`` is
scoped to the PACKAGE and correctly so -- capturing and recording legitimately
live in different modules, and a file-scoped version would fire on every
correct arrangement. The predicate below is different in kind: it is about
THIS package's ``scripts/`` layout and its own encoder naming, so a general
rule would have to carry a covenant_ml-specific special case to express it.
That is the wrong direction for a shared guard, and it is the right shape for
a test that sits beside what it checks.

WHAT IT CHECKS, and why source text rather than imports. Until 2026-09-09 five
of six entry points wrote a manifest and no record. The fix was structural --
:func:`~covenant_ml.benchmarking.harness.write_manifest_and_record` writes both
in one call -- but structure alone does not stop the next entry point from
writing a file itself. So: a benchmark script must reach the harness, and must
NOT write any file directly. The second half is what makes the first
un-defeatable; without it a script could call the harness and then overwrite
the manifest on its own terms.
"""

from __future__ import annotations

from pathlib import Path

#: Every benchmark entry point, found rather than listed.
#:
#: Globbed so a new family is covered the day it is written, instead of the day
#: someone remembers to add it here -- which is the failure mode this whole
#: task is about.
ENTRY_POINTS: tuple[Path, ...] = tuple(
    sorted((Path(__file__).parents[2] / "scripts").glob("benchmark_cleargbm_*.py"))
)


def test_the_entry_points_were_actually_found() -> None:
    """A glob that matches nothing would make every test below vacuous."""
    assert len(ENTRY_POINTS) == 6, [path.name for path in ENTRY_POINTS]


def test_every_entry_point_reaches_the_shared_write_path() -> None:
    """No family may write a manifest without the record beside it."""
    for path in ENTRY_POINTS:
        source = path.read_text(encoding="utf-8")
        assert "write_manifest_and_record" in source, path.name


def test_no_entry_point_writes_a_file_itself() -> None:
    """The harness owns writing, so a script that opens a file has left it.

    This is the half that makes the check un-defeatable: calling the harness
    and then writing the manifest again on its own terms would satisfy the
    test above while reproducing the defect.
    """
    for path in ENTRY_POINTS:
        source = path.read_text(encoding="utf-8")
        assert "write_text(" not in source, path.name
        assert "open(" not in source, path.name


def test_every_entry_point_declares_rather_than_builds_its_arguments() -> None:
    """A hand-rolled parser is how the shared flags drifted apart before.

    All six carried their own ``add_argument`` calls, which is how one family
    ended up without ``--learning-rate`` and nobody noticed.
    """
    for path in ENTRY_POINTS:
        source = path.read_text(encoding="utf-8")
        assert "add_argument" not in source, path.name
        assert "shared_parser" in source, path.name


def test_no_entry_point_imports_covenant_ml_at_module_scope() -> None:
    """The pin must precede numpy, and importing this package loads it.

    Asserted on the source rather than by importing, because importing to
    check would itself be the violation. The marker is position: any
    ``covenant_ml`` import must sit indented inside a function, after the pin.
    """
    for path in ENTRY_POINTS:
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line.startswith("from covenant_ml") or line.startswith("import covenant_ml"):
                raise AssertionError(f"{path.name}: module-scope covenant_ml import: {line}")
