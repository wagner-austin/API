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
claim more -- and this passage said the opposite until 2026-09-09, which is
itself the lesson. It read: "``covenant_ml`` was believed to have the same
defect and did not: ``benchmarking/provenance.py`` has emitted a ``RunRecord``
alongside its manifest since the fingerprint landed." TRUE OF THE MODULE AND
FALSE OF THE PACKAGE. A power audit (board ``6d5536cc``) then measured the six
``benchmark_cleargbm_*`` entry points and found that exactly ONE called it;
the other five wrote a manifest and stopped. The module existed, was correct,
was imported by the package -- and five of the six paths that produce published
numbers never reached it.

THIS RULE PASSED covenant_ml THROUGHOUT, correctly and by design: the package
imports ``run_record``, which is all a package-scoped import check can see. The
audit's own finding names the gap better than a wider rule would close it --
"importing ``run_record`` is not proof that a record is written on the path
that matters". What closed it was structural rather than another check:
``benchmarking/harness.py`` now writes the manifest and the record in one call,
so an entry point cannot emit the first without the second.

The residual lesson for THIS docstring is narrower and worth keeping: a
sentence that clears a package on the strength of one module will read as a
clearance of every path in it.

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

from pathlib import Path
from typing import Final

from monorepo_guards import Violation
from monorepo_guards.util import imported_module_names, imported_names, package_of, parse_source

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
            package = package_of(path)
            if package in _DEFINING_PACKAGES:
                continue
            tree = parse_source(path)
            names = imported_names(tree) | imported_module_names(tree)
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
