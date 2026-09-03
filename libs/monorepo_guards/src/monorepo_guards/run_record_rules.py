"""Guard rule: a package that fingerprints a run must also emit a RunRecord.

A ``RunFingerprint`` says what a number was produced under. A ``RunRecord``
is the envelope that lets that number be read beside another experiment's:
``compare_run_records`` and ``agree_across_runs`` take ``RunRecord`` and
nothing else. Capturing the first without emitting the second produces a
record that is complete, correct, and unreadable to every consumer in the
workspace.

The real instance: ``Model-Trainer``'s training path captured a full
fingerprint into its training manifest and emitted no ``RunRecord`` until
2026-09-03, so a fine-tuned adapter was less comparable than the benchmarks
measuring the card it trained on. It was found by reading, not by any check.
This rule is the mechanical version of that reading.

WHAT THIS RULE WOULD NOT HAVE CAUGHT, stated because the temptation is to
claim more. ``covenant_ml`` was believed to have the same defect and did not:
``benchmarking/provenance.py`` has emitted a ``RunRecord`` alongside its
manifest since the fingerprint landed. Two documents asserted otherwise and a
session rewrote an existing module before reading it. That failure is prose
drifting from code, which no import-graph rule can see; it is the argument for
this rule being mechanical rather than for it being wider.

SCOPE IS THE PACKAGE, NOT THE FILE, and deliberately. Capturing and recording
legitimately live in different modules: ``covenant_ml`` captures in
``benchmarking/provenance.py`` and records in the same file, while
``Model-Trainer`` captures in ``core/run_fingerprint.py`` and records in
``core/services/training/run_records.py``. A file-scoped rule would force
those together or fire on every correct arrangement.

WHAT IT DOES NOT CLAIM. Importing ``run_record`` is not proof that a record is
written on the path that matters, and this rule cannot show that. It closes
the gap where a package captures provenance and has no way to emit it at all,
which is the shape both real failures took.

Violations:
- run-record-missing: a package captures a RunFingerprint and never builds a
  RunRecord
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

from monorepo_guards import Violation
from monorepo_guards.util import parse_source

#: The symbol whose presence means a package is recording provenance.
_FINGERPRINT_SYMBOL: Final[str] = "RunFingerprint"

#: Symbols that mean a package emits the workspace's record. Any one is
#: enough: a package may build a record, encode one it was handed, or name
#: the sidecar path for a record built elsewhere in the same package.
_RECORD_SYMBOLS: Final[frozenset[str]] = frozenset(
    {"RunRecord", "run_record", "encode_run_record", "run_record_sidecar"}
)

#: Packages exempt because they DEFINE the vocabulary rather than use it.
#: ``platform_core`` owns both types, and the guard package itself names them
#: only in this rule's own text and tests.
_DEFINING_PACKAGES: Final[frozenset[str]] = frozenset({"platform_core", "monorepo_guards"})


def _package_of(path: Path) -> str:
    """Name the package a file belongs to.

    Args:
        path: The file.

    Returns:
        The directory name two levels above ``src``/``tests`` where the
        monorepo's layout puts the package, or the file's own parent when the
        path is shorter than that.
    """
    parts = path.as_posix().split("/")
    for marker in ("src", "tests", "scripts"):
        if marker in parts:
            index = parts.index(marker)
            if index > 0:
                return parts[index - 1]
    return path.parent.name


def _imported_symbols(tree: ast.Module) -> set[str]:
    """Collect the names a module IMPORTS.

    Imports only, deliberately. An earlier version also collected every
    ``Name`` and ``Attribute`` node, which meant a local variable happening to
    be called ``run_record`` satisfied the rule -- a check that a package can
    pass by accident is worse than no check, because it reads as verified.

    Args:
        tree: Parsed module.

    Returns:
        The imported names, including the final component of a dotted module
        import so ``import platform_core.run_record`` counts.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
            if node.module is not None:
                names.add(node.module.rsplit(".", maxsplit=1)[-1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.rsplit(".", maxsplit=1)[-1])
    return names


class RunRecordRule:
    """Guard rule pairing fingerprint capture with record emission."""

    name = "run-record"

    def run(self, files: list[Path]) -> list[Violation]:
        """Check every package that fingerprints a run.

        Args:
            files: Python source files to check.

        Returns:
            One violation per offending package, at the first file that
            captured a fingerprint, so the message points somewhere real.
        """
        captures: dict[str, Path] = {}
        records: set[str] = set()
        for path in files:
            package = _package_of(path)
            if package in _DEFINING_PACKAGES:
                continue
            names = _imported_symbols(parse_source(path))
            if _FINGERPRINT_SYMBOL in names and package not in captures:
                captures[package] = path
            if names & _RECORD_SYMBOLS:
                records.add(package)
        return [
            Violation(
                file=path,
                line_no=1,
                kind="run-record-missing",
                line=(
                    f"package '{package}' captures a {_FINGERPRINT_SYMBOL} but never "
                    f"builds a RunRecord; a fingerprint inside a private shape cannot "
                    f"be read by compare_run_records"
                ),
            )
            for package, path in sorted(captures.items())
            if package not in records
        ]


__all__ = ["RunRecordRule"]
