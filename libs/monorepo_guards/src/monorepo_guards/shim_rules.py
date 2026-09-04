"""Rule refusing code that announces it is kept for an old caller.

The standing rule for this workspace is "no back-compat shims, no thin
wrappers, no fallbacks, no legacy code, no type alias, no re-exports". Almost
every clause of it is machine-checked somewhere: ``typing_rules`` refuses
``X: TypeAlias = ...``, ``passthrough_rules`` refuses the same shape written
without the annotation, ``exceptions_rules`` refuses a swallowed error. The
compatibility clause was checked only inside ``clients/TankpitBot``, by a
module in its ``scripts/`` directory that nothing but its own test ever
called.

So it protected one package out of forty-four, and the packages it did not
protect had accumulated exactly the shape it bans -- five re-export modules in
``handwriting-ai``, one in ``grandma-api``, one in ``platform_calendar``, one
in ``github-stats-api``, each carrying a comment saying what it was for. A
rule one package owns is a rule the other forty-three do not have; that is the
same finding as ``entrypoint_rules``, reached independently.

WHY THIS RULE IS VOCABULARY ONLY, AND DOES NOT ALSO CHECK ALIASES.

The lifted rule checked three things: compatibility prose, ``X = X``, and
``NEW = OLD`` where ``OLD`` is imported and ``NEW`` is exported. The second
and third are :class:`~monorepo_guards.passthrough_rules.PassthroughRule`,
which already runs on every package and was measured at 27 findings and no
false positives across 4896 files.

Run over this monorepo, the alias half reported four findings and every one
was correct code: ``JOB_RADIUS = RING_SLOT_RADIUS``, ``stdlib_logging =
logging``, and the two ``WORKSPACE_VAR = CUBLASLT_WORKSPACE_ENV_VAR`` bindings
whose comment explains that a second spelling is exactly what would let the
measured condition and the applied one drift apart. All four are constants or
a module, which ``PassthroughRule`` deliberately ignores -- its docstring
already says that "a constant bound to another constant is a duplicated value:
a different problem with a different fix", and its type-spelling predicate is
what makes that stick. Shipping a second, blunter copy of that check would be
the fork this module exists to remove.

WHAT WAS DROPPED FROM THE VOCABULARY, AND WHAT IT COST TO KEEP IT.

The lifted rule's charter says its patterns are "chosen because each is
unambiguous from the syntax alone. A rule that needs a human to adjudicate
would need an allowlist, and an allowlist is the thing this project refuses."
Measured against the whole monorepo rather than one client, three of its six
patterns fail that test outright:

``\\blegacy\\b`` matched twenty times in ``src/`` and named someone else's old
thing every time -- NVIDIA's ``cublasSgemm`` "legacy entry point", which
``legacy_gemm_probe`` exists to measure, and openpyxl's refusal to read
"legacy .xls". Banning the word would force renaming a probe whose entire
subject is that path.

``deprecat\\w+`` matched ten times and was a third-party deprecation WARNING
every time -- SWIG's, fasttext's, ddtrace's ``patch_all``.

A bare ``for compatibility`` went the same way on the same evidence. Its three
matches were one file, ``chunked_csv_reader``, converting a Polars frame into
the list-of-lists its callers take -- a boundary conversion, which this
workspace asks for rather than bans. Dropping it costs nothing: every real
shim in the tree said "backward compatibility" in full, and the narrower
``kept for api/signature compatibility`` still stands.

All three name a thing, or an act, rather than an intent, and none is
decidable without a human. What is left states an intent to preserve an old
interface -- the thing actually banned, and which nothing legitimate in this
monorepo does.

Tests are out of scope. A test asserting that a field has "no back-compat
default" is evidence the rule holds, and firing on it would make the rule
punish its own enforcement.
"""

from __future__ import annotations

import re
from pathlib import Path

from monorepo_guards import Violation
from monorepo_guards.util import read_source

SCANNED_ROOTS = ("src", "scripts")
"""Where shipped code lives. Tests are excluded -- see the module docstring."""

TESTS_ROOT = "tests"

_GAP = r"[-\s#*]+"
"""What may sit between the words of a marker.

A newline, because a wrapped docstring is still a marker. This is not a
nicety: ``covenant-radar-api``'s history entry reads "flat best_* fields for
backward" / "compatibility with the JSONL history format", and a rule matching
one line at a time reported it clean. A guard that ``ruff format`` can defeat
by rewrapping a comment is not a guard.
"""

COMPATIBILITY_PATTERNS: tuple[str, ...] = (
    rf"back{_GAP}?compat\w*",
    rf"backwards?{_GAP}compatib\w*",
    rf"kept{_GAP}for{_GAP}(?:api|signature){_GAP}compatibility",
)
"""Prose announcing that something is kept for an old caller.

Matched case-insensitively, across line breaks. Every one states an INTENT to
preserve an interface, which is what the standing rule bans; see the module
docstring for the three patterns that named a thing instead and were dropped,
with counts.
"""

_COMPATIBILITY = re.compile("|".join(COMPATIBILITY_PATTERNS), re.IGNORECASE)


def compatibility_markers(source: str) -> list[tuple[int, str]]:
    """Find every announcement that code is kept for an old caller.

    Takes the whole source rather than a line at a time so that a marker split
    across a line break is still found; see :data:`_GAP`.

    Args:
        source: The module's text.

    Returns:
        ``(line number, matched text)`` pairs in file order. The text has its
        whitespace collapsed, so a wrapped marker reads as one phrase in the
        refusal rather than carrying the indentation of its second line.
    """
    return [
        (source.count("\n", 0, match.start()) + 1, " ".join(match.group(0).split()))
        for match in _COMPATIBILITY.finditer(source)
    ]


class ShimRule:
    """Nothing in shipped code is kept for a caller that no longer exists."""

    name = "shim"

    def _in_scope(self, path: Path) -> bool:
        """Say whether a file is shipped code this rule checks.

        Args:
            path: The file.

        Returns:
            Whether it lives under a scanned root and outside ``tests``.
        """
        parts = path.parts
        return any(root in parts for root in SCANNED_ROOTS) and TESTS_ROOT not in parts

    def _violations(self, path: Path) -> list[Violation]:
        """Check one file.

        Args:
            path: The file to check.

        Returns:
            Its violations, empty when the file is out of scope or clean.
        """
        # The module that DEFINES the banned vocabulary necessarily contains
        # it. Compared by resolved path rather than by filename, so this is an
        # identity and not an allowlist that could gain a second entry.
        if not self._in_scope(path) or path.resolve() == Path(__file__).resolve():
            return []
        return [
            Violation(
                file=path,
                line_no=number,
                kind="shim-compatibility-marker",
                line=(
                    f"'{text}' announces code kept for an old caller. Delete the "
                    "old interface and update what calls it; there is no "
                    "released version to stay compatible with."
                ),
            )
            for number, text in compatibility_markers(read_source(path))
        ]

    def run(self, files: list[Path]) -> list[Violation]:
        """Check every file.

        Args:
            files: The files to check.

        Returns:
            Every violation found, in file order.
        """
        out: list[Violation] = []
        for path in files:
            out.extend(self._violations(path))
        return out


__all__ = [
    "COMPATIBILITY_PATTERNS",
    "SCANNED_ROOTS",
    "TESTS_ROOT",
    "ShimRule",
    "compatibility_markers",
]
