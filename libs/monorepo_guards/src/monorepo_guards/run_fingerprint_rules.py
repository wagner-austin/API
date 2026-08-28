"""Guard rule: a run fingerprint is never written out as a JSON literal.

WHAT MYPY ALREADY CATCHES, SO THIS DOES NOT. ``RunFingerprint`` is a
``TypedDict``, so a constructor call that omits an axis is a type error and
the type checker names every site. When the host and package axes were added
on 2026-08-27 that is exactly what happened: 79 errors across 10 files, every
one of them a real site that needed updating, and none of them able to reach
production.

WHAT IT DOES NOT CATCH, WHICH IS WHY THIS EXISTS. A fingerprint that has
already been ENCODED is a plain JSON object, and a test that writes one as a
dict literal is invisible to the type checker. Two such literals existed in
``Model-Trainer`` and both went stale in the same commit -- they listed four
axes, the type grew to six, and the failure surfaced as
``JSONTypeError: Missing required field 'host'`` at runtime in eleven tests
rather than as a type error at the site that was wrong.

The fix those two sites took is the rule: build the fingerprint with the
canonical builder and call ``encode_run_fingerprint`` on it. A literal cannot
fall behind the type when it is not a literal.

Violations:
- run-fingerprint-json-literal: a dict literal carries fingerprint axis keys
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

from monorepo_guards import Violation
from monorepo_guards.util import parse_source

#: Keys that together identify a dict literal as an encoded run fingerprint.
#:
#: All of them are required rather than any one, so an unrelated mapping that
#: happens to carry a ``"host"`` key -- a request header table, a cluster
#: config -- is not swept up. A literal carrying every one of these is not
#: plausibly anything else.
_FINGERPRINT_KEYS: Final[frozenset[str]] = frozenset(
    {"image_digest", "gpu_model", "driver_version", "determinism"}
)

#: The module that defines the type, where the encoder itself lives and where
#: a literal IS the subject rather than a copy of it.
#:
#: A path fragment rather than a basename: a basename is not a path, and a
#: file called ``comparability.py`` in any other package would inherit an
#: exemption it was never granted.
_DEFINING_MODULE: Final[str] = "platform_core/src/platform_core/comparability.py"


def _comparison_operands(tree: ast.Module) -> frozenset[int]:
    """Identify dict literals that are operands of a comparison.

    THE DISTINCTION THIS DRAWS IS THE WHOLE RULE. A literal a fixture is BUILT
    from rots silently when the type grows an axis -- it keeps producing a
    value, just an incomplete one. A literal a test COMPARES against does the
    opposite: add an axis, and the captured value has a key the literal lacks,
    so ``==`` fails and names the site. An exhaustive equality assertion is
    therefore the one place a full literal is not merely safe but load-bearing,
    because it is what proves capture fills every axis.

    Caught by this rule's first run against real code, on
    ``test_run_fingerprint``'s assertion that `capture_run_fingerprint` returns
    exactly six named axes. Exempting that file would have been the wrong fix:
    the predicate was wrong, not the test.

    Args:
        tree: The parsed module.

    Returns:
        The ``id()`` of every dict literal used as a comparison operand.
    """
    operands: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for side in (node.left, *node.comparators):
            if isinstance(side, ast.Dict):
                operands.add(id(side))
    return frozenset(operands)


class _FingerprintLiteralVisitor(ast.NodeVisitor):
    """Finds dict literals shaped like an encoded run fingerprint."""

    def __init__(self, path: Path, compared: frozenset[int]) -> None:
        """Start a scan of one file.

        Args:
            path: The file being scanned, for violation reporting.
            compared: ``id()`` of dict literals used as comparison operands,
                which are assertion targets rather than fixtures.
        """
        self.path = path
        self.compared = compared
        self.violations: list[Violation] = []

    def visit_Dict(self, node: ast.Dict) -> None:
        """Record a dict literal that spells out a fingerprint.

        Args:
            node: The dict literal to inspect.
        """
        literal_keys = {k.value for k in node.keys if isinstance(k, ast.Constant)}
        if literal_keys >= _FINGERPRINT_KEYS and id(node) not in self.compared:
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=node.lineno,
                    kind="run-fingerprint-json-literal",
                    line=(
                        "an encoded run fingerprint written as a dict literal goes stale "
                        "silently when the type grows an axis; build it with "
                        "platform_core.testing.sample_run_fingerprint and pass it to "
                        "encode_run_fingerprint instead"
                    ),
                )
            )
        self.generic_visit(node)


class RunFingerprintLiteralRule:
    """Guard rule keeping encoded fingerprints out of source as literals."""

    name = "run-fingerprint-literal"

    def run(self, files: list[Path]) -> list[Violation]:
        """Scan every file for hand-written encoded fingerprints.

        Args:
            files: Python source files to check.

        Returns:
            Every literal found, in file order.
        """
        found: list[Violation] = []
        for path in files:
            if path.as_posix().endswith(_DEFINING_MODULE):
                continue
            tree = parse_source(path)
            visitor = _FingerprintLiteralVisitor(path, _comparison_operands(tree))
            visitor.visit(tree)
            found.extend(visitor.violations)
        return found


__all__ = ["RunFingerprintLiteralRule"]
