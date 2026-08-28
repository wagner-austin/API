"""Guard rule: the diagnostic_kind producer-consumer wiring stays live.

The link between an ``emit_diagnostic(diagnostic_kind="x")`` producer
and the reporting consumer that counts ``"x"`` is a string literal --
invisible to mypy, ruff, and import analysis. Three scorecard counters
(``combat_ghost_detected``, ``combat_stale_position``,
``equipment_approach``) survived their emitters' deletion for months,
rendering zeros forever, because nothing cross-checked the two sides
(found by the 2026-08-28 dead-diagnostic audit). This rule makes that
class a build failure:

* every kind a consumer in ``src/tankpit_bot/diagnostics/`` compares
  against must be emitted somewhere in ``src/`` -- a consumer of a
  dead kind fails the gate the day the refactor lands;
* every kind a test fabricates (``diagnostic_kind="x"`` kwarg or a
  ``"diagnostic_kind": "x"`` payload, including raw JSONL strings)
  must be production-emitted, so fixtures cannot certify corpses. A
  test that deliberately exercises unknown-kind fall-through names its
  kind with the ``fake_`` prefix -- self-describing, no allowlist.

Enforced against the real tree by
``tests/scripts/test_diagnostic_kind_rules.py``.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

_KIND_FIELD = "diagnostic_kind"

#: Variable names that hold a diagnostic kind by package convention
#: inside ``src/tankpit_bot/diagnostics/`` (``action_kind`` is a
#: different name on purpose -- naming an action-kind variable ``kind``
#: there would trip this rule, which is the discipline working).
_KIND_VARIABLE_NAMES = frozenset({"kind", "kind_field"})

#: Prefix marking a test-fabricated kind as intentionally unknown
#: (unknown-kind fall-through tests). Anything else a test fabricates
#: must be a kind production actually emits.
FAKE_KIND_PREFIX = "fake_"

_TEST_KIND_PATTERNS = (
    re.compile(r'diagnostic_kind="([a-z_0-9]+)"'),
    re.compile(r'\\?"diagnostic_kind\\?":\s*\\?"([a-z_0-9]+)\\?"'),
)


def _emitted_kinds(src_root: Path) -> set[str]:
    """Collect every constant kind passed to ``emit_diagnostic`` in src.

    Args:
        src_root: The ``src`` directory to scan.

    Returns:
        Set of emitted kind strings. Non-constant pass-throughs (the
        emitter helper itself) are skipped.
    """
    kinds: set[str] = set()
    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else None
            if name is None and isinstance(func, ast.Attribute):
                name = func.attr
            if name != "emit_diagnostic":
                continue
            for keyword in node.keywords:
                if keyword.arg != _KIND_FIELD:
                    continue
                value = keyword.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    kinds.add(value.value)
    return kinds


def _reads_kind(node: ast.expr) -> bool:
    """Report whether an expression derives from the kind field.

    True when the expression is a kind-holding variable by convention,
    or when any call argument / subscript index in its subtree is the
    literal ``"diagnostic_kind"`` (``fields.get("diagnostic_kind")``,
    ``record["diagnostic_kind"]``). A bare ``"diagnostic_kind"``
    comparator constant does NOT count -- that is field-name filtering,
    not kind consumption.

    Args:
        node: Expression to inspect.

    Returns:
        True when the expression reads the diagnostic kind.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in _KIND_VARIABLE_NAMES:
            return True
        if isinstance(child, ast.Call):
            for arg in child.args:
                if isinstance(arg, ast.Constant) and arg.value == _KIND_FIELD:
                    return True
        if isinstance(child, ast.Subscript):
            index = child.slice
            if isinstance(index, ast.Constant) and index.value == _KIND_FIELD:
                return True
    return False


def _constant_strings(node: ast.expr) -> list[str]:
    """Collect the plain string constants a comparator side offers.

    Args:
        node: One side of a comparison -- a constant, or a tuple / set /
            list of constants for membership tests.

    Returns:
        The string constants found (excluding the field name itself).
    """
    values: list[str] = []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        values.append(node.value)
    elif isinstance(node, (ast.Tuple, ast.Set, ast.List)):
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                values.append(element.value)
    return [value for value in values if value != _KIND_FIELD]


def _node_consumptions(node: ast.AST, path: Path) -> list[tuple[str, str]]:
    """Collect the kinds one AST node compares the kind field against.

    Covers ``==`` / ``!=`` / ``in`` / ``not in`` comparisons and
    ``match`` statements whose subject reads the kind field.

    Args:
        node: The AST node to inspect.
        path: Source file, for violation locations.

    Returns:
        ``(kind, "file:line")`` pairs, possibly empty.
    """
    consumed: list[tuple[str, str]] = []
    if isinstance(node, ast.Compare):
        sides: list[ast.expr] = [node.left, *node.comparators]
        if any(_reads_kind(side) for side in sides):
            for side in sides:
                for kind in _constant_strings(side):
                    consumed.append((kind, f"{path}:{side.lineno}"))
    elif isinstance(node, ast.Match) and _reads_kind(node.subject):
        for case in node.cases:
            pattern = case.pattern
            if isinstance(pattern, ast.MatchValue):
                for kind in _constant_strings(pattern.value):
                    consumed.append((kind, f"{path}:{pattern.value.lineno}"))
    return consumed


def _consumed_kinds(diagnostics_root: Path) -> list[tuple[str, str]]:
    """Collect every kind the diagnostics consumers compare against.

    Args:
        diagnostics_root: The ``src/tankpit_bot/diagnostics`` package.

    Returns:
        ``(kind, "file:line")`` pairs in scan order.
    """
    consumed: list[tuple[str, str]] = []
    for path in sorted(diagnostics_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            consumed.extend(_node_consumptions(node, path))
    return consumed


def _test_kind_references(tests_root: Path) -> list[tuple[str, str]]:
    """Collect every kind the tests fabricate, with its location.

    Text-level scan on purpose: fabrications live in kwargs, dict
    literals, AND raw JSONL strings inside test fixtures -- the last is
    invisible to AST value analysis.

    Args:
        tests_root: The ``tests`` directory to scan.

    Returns:
        ``(kind, "file:line")`` pairs in scan order.
    """
    references: list[tuple[str, str]] = []
    for path in sorted(tests_root.rglob("*.py")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for pattern in _TEST_KIND_PATTERNS:
                for match in pattern.finditer(line):
                    references.append((match.group(1), f"{path}:{line_no}"))
    return references


def run_diagnostic_kind_rules(project_root: Path) -> int:
    """Run the diagnostic-kind wiring rule over a project tree.

    Args:
        project_root: Project root containing ``src`` and ``tests``.

    Returns:
        Number of violations found (0 means the rule passes).
    """
    src_root = project_root / "src"
    emitted = _emitted_kinds(src_root) if src_root.is_dir() else set()
    violations: list[str] = []

    diagnostics_root = src_root / "tankpit_bot" / "diagnostics"
    if diagnostics_root.is_dir():
        for kind, location in _consumed_kinds(diagnostics_root):
            if kind not in emitted:
                violations.append(
                    f"{location} consumer reads diagnostic_kind '{kind}' "
                    "but no emit_diagnostic in src emits it -- the counter "
                    "is dead wiring (delete it or restore the emitter)"
                )

    tests_root = project_root / "tests"
    if tests_root.is_dir():
        for kind, location in _test_kind_references(tests_root):
            if kind not in emitted and not kind.startswith(FAKE_KIND_PREFIX):
                violations.append(
                    f"{location} test fabricates diagnostic_kind '{kind}' "
                    "which production never emits -- fixtures must mirror "
                    "real emitters (use a production kind, or the "
                    f"'{FAKE_KIND_PREFIX}' prefix for a deliberate "
                    "unknown-kind fall-through test)"
                )

    for violation in violations:
        sys.stdout.write(f"diagnostic_kind_rule_violation {violation}\n")
    return len(violations)


__all__ = [
    "FAKE_KIND_PREFIX",
    "run_diagnostic_kind_rules",
]
