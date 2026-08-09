"""Guard rule: a test that swaps a ``_test_hooks`` attribute must restore it.

Tests inject fakes by save-and-restore on ``_test_hooks.<attr>``, so a
swap that is never put back leaks into every later test on the same
xdist worker. ``tests/conftest.py`` carries an autouse ``_restore_hooks``
fixture that resets fifteen attrs for exactly this reason -- its own
docstring records the 4-test replay flake of 2026-07-03, where a leaked
frozen clock made every replay enemy read as ``stale_map_data``.

The gap is the attrs outside that list. A sweep on 2026-08-08 found 391
assignment sites across 20 distinct attrs, five of them outside the
autouse list, and one genuinely unrestored: ``remove_file`` in
``tests/bot/test_tick_loop_lifecycle.py`` left every later test on the
worker with a ``remove_file`` that silently did not delete and appended
to a dead closure list. One defect in 391 sites is precisely the density
a hand sweep cannot be trusted to hold -- hence a rule.

Restoration is recognised at three scopes, because the first draft of
this sweep reported 24 violations of which 23 were false positives:

* function -- the attr is reassigned inside a ``finally`` body;
* function -- the attr is reassigned after a ``yield`` (a fixture that
  saves, yields, then puts back);
* class/module -- the attr is reassigned in a ``teardown_*`` method, or
  anywhere in an ancestor ``conftest.py`` under one of the above, which
  is how an autouse fixture protects every test in its directory.

A rule that flagged those would be deleted within a day, so all three
count as guarded.

Runs as part of ``make lint`` via ``scripts.guard``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_HOOKS_SUFFIX = "_test_hooks"
_RESET_FIXTURE = "_restore_hooks"


def _hook_attr(target: ast.expr) -> str | None:
    """Return the attr name when ``target`` is a ``*_test_hooks.<attr>``.

    Args:
        target: Assignment target expression.

    Returns:
        The attribute name, or None when the target is something else.
    """
    if not isinstance(target, ast.Attribute):
        return None
    value = target.value
    if isinstance(value, ast.Name) and value.id.endswith(_HOOKS_SUFFIX):
        return target.attr
    if isinstance(value, ast.Attribute) and value.attr.endswith(_HOOKS_SUFFIX):
        return target.attr
    return None


def _assigned_attrs(node: ast.AST) -> set[str]:
    """Collect every ``_test_hooks`` attr assigned anywhere under ``node``.

    Tuple and list targets are flattened: ``a.x, b.y = ...`` stores two
    attributes in one statement.

    Args:
        node: Any AST node to walk.

    Returns:
        Set of attribute names assigned.
    """
    found: set[str] = set()
    for child in ast.walk(node):
        targets: list[ast.expr] = []
        if isinstance(child, ast.Assign):
            targets = list(child.targets)
        elif isinstance(child, (ast.AugAssign, ast.AnnAssign)):
            targets = [child.target]
        flat: list[ast.expr] = []
        for target in targets:
            if isinstance(target, (ast.Tuple, ast.List)):
                flat.extend(target.elts)
            else:
                flat.append(target)
        for target in flat:
            attr = _hook_attr(target)
            if attr is not None:
                found.add(attr)
    return found


def _restored_attrs(tree: ast.Module) -> set[str]:
    """Collect attrs a module puts back under a recognised guard.

    Args:
        tree: Parsed module.

    Returns:
        Set of attribute names restored in a ``finally`` body, after a
        ``yield``, or inside a ``teardown_*`` function.
    """
    restored: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for stmt in node.finalbody:
                restored |= _assigned_attrs(stmt)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("teardown"):
                restored |= _assigned_attrs(node)
                continue
            yields = [n.lineno for n in ast.walk(node) if isinstance(n, ast.Yield)]
            if not yields:
                continue
            first_yield = min(yields)
            for stmt in node.body:
                if stmt.lineno > first_yield:
                    restored |= _assigned_attrs(stmt)
    return restored


def _reset_fixture_attrs(conftest: Path) -> set[str]:
    """Read the attrs the autouse reset fixture resets.

    Args:
        conftest: Path to the root ``tests/conftest.py``.

    Returns:
        Set of attribute names the fixture restores, empty when absent.
    """
    if not conftest.is_file():
        return set()
    tree = ast.parse(conftest.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name == _RESET_FIXTURE
        ):
            return _assigned_attrs(node)
    return set()


def _ancestor_conftest_attrs(path: Path, tests_root: Path) -> set[str]:
    """Collect restored attrs from every ``conftest.py`` above ``path``.

    A fixture in a directory's ``conftest.py`` applies to the tests
    beneath it, so its restorations protect them.

    Args:
        path: Test module being checked.
        tests_root: The ``tests`` directory.

    Returns:
        Set of attribute names restored by an ancestor conftest.
    """
    attrs: set[str] = set()
    relative = path.parent.relative_to(tests_root)
    for ancestor in (relative, *relative.parents):
        conftest = tests_root / ancestor / "conftest.py"
        if conftest.is_file():
            attrs |= _restored_attrs(ast.parse(conftest.read_text(encoding="utf-8")))
    return attrs


def _module_violations(path: Path, tests_root: Path, reset: set[str]) -> list[str]:
    """Report attrs a module swaps without any recognised restoration.

    Args:
        path: Test module to check.
        tests_root: The ``tests`` directory.
        reset: Attrs the autouse reset fixture already covers.

    Returns:
        Violation messages, one per unrestored attribute.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assigned = _assigned_attrs(tree)
    if not assigned:
        return []
    safe = reset | _restored_attrs(tree) | _ancestor_conftest_attrs(path, tests_root)
    return [f"{path}:{attr}" for attr in sorted(assigned - safe)]


def run_hook_restore_rules(project_root: Path) -> int:
    """Run the hook-restoration guard rule over a project's tests.

    Args:
        project_root: Project root containing ``tests``.

    Returns:
        Number of violations found (0 means the rule passes).
    """
    tests_root = project_root / "tests"
    if not tests_root.is_dir():
        return 0
    reset = _reset_fixture_attrs(tests_root / "conftest.py")
    violations: list[str] = []
    for module_path in sorted(tests_root.rglob("*.py")):
        violations.extend(_module_violations(module_path, tests_root, reset))
    for violation in violations:
        sys.stdout.write(f"hook_restore_violation {violation}\n")
    return len(violations)


__all__ = ["run_hook_restore_rules"]
