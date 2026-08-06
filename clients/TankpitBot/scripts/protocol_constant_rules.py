"""Guard rule: no shadow re-declarations of protocol wire constants.

``protocol/constants.py`` is the single home of the 0x52 supervisor
error vocabulary (``SUPERVISOR_ERROR_*`` codes and their canonical
names). Until 2026-08-03 three modules privately re-declared it
(``bot/tick_loop_actions.py`` + ``bot/tick_loop.py`` as
``_COMMAND_ERROR_*`` ints, ``sniffer/world_state_dispatch.py`` as a
name dict) — a correction to any code's meaning had to be remembered
in four places. The forks were consolidated; this rule keeps the
class dead: any assignment under ``src/tankpit_bot`` (outside the
canonical module) to a name matching the error-constant patterns
whose value embeds a bare integer literal is a violation. Tables of
NAMED constants (e.g. per-kind applicability whitelists) stay legal —
the rule bans re-encoding the numbers, not referencing them.

Runs as part of ``make lint`` via ``scripts.guard``.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

CANONICAL_MODULE = Path("protocol") / "constants.py"
SHADOW_NAME_PATTERN = re.compile(r"^_?(SUPERVISOR_ERROR|COMMAND_ERROR)_[A-Z0-9_]+$")


def _embeds_int_literal(node: ast.AST) -> bool:
    """Report whether an expression contains any bare integer literal.

    Args:
        node: Value expression of an assignment.

    Returns:
        True if any descendant is an int constant (bools excluded).
    """
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Constant)
            and not isinstance(child.value, bool)
            and isinstance(child.value, int)
        ):
            return True
    return False


def _assignment_targets(node: ast.stmt) -> tuple[list[ast.expr], ast.expr | None]:
    """Extract the targets and value of a plain or annotated assignment.

    Args:
        node: Statement to inspect.

    Returns:
        Pair of (target expressions, value expression or None).
    """
    if isinstance(node, ast.Assign):
        return node.targets, node.value
    if isinstance(node, ast.AnnAssign):
        return [node.target], node.value
    return [], None


def _module_violations(path: Path) -> list[str]:
    """Collect shadow-constant violations in one module.

    Args:
        path: Python file to scan.

    Returns:
        Violation messages, one per shadow declaration.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        targets, value = _assignment_targets(node)
        if value is None:
            continue
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if not SHADOW_NAME_PATTERN.match(target.id):
                continue
            if _embeds_int_literal(value):
                violations.append(
                    f"{path}:{node.lineno} '{target.id}' re-declares protocol "
                    "constants with integer literals - import from "
                    "protocol/constants.py instead"
                )
    return violations


def run_protocol_constant_rules(project_root: Path) -> int:
    """Run the shadow-constant guard rule over a project tree.

    Args:
        project_root: Project root containing ``src/tankpit_bot``.

    Returns:
        Number of violations found (0 means the rule passes).
    """
    package_root = project_root / "src" / "tankpit_bot"
    if not package_root.is_dir():
        return 0
    canonical = package_root / CANONICAL_MODULE
    violations: list[str] = []
    for module_path in sorted(package_root.rglob("*.py")):
        if module_path == canonical:
            continue
        violations.extend(_module_violations(module_path))
    for violation in violations:
        sys.stdout.write(f"protocol_constant_violation {violation}\n")
    return len(violations)


__all__ = [
    "CANONICAL_MODULE",
    "SHADOW_NAME_PATTERN",
    "run_protocol_constant_rules",
]
