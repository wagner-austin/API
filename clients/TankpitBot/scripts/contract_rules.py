"""Guard rule: public mutation functions must carry @enforce_contract.

Scans ``facts/``, ``ledger/``, and ``memory/`` under
``src/tankpit_bot/`` for module-level public functions whose names
mark them as state mutations (``apply_*``, ``record_*``, ``mutate_*``,
``set_*``, ``update_*``) and reports any that are not decorated with
``@enforce_contract(...)``. Runs as part of ``make lint`` via
``scripts.guard``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

GUARDED_PACKAGES: tuple[str, ...] = ("facts", "ledger", "memory")
MUTATION_PREFIXES: tuple[str, ...] = ("apply_", "record_", "mutate_", "set_", "update_")


def _has_enforce_contract(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Report whether a function carries the @enforce_contract decorator.

    Args:
        node: Function definition to inspect.

    Returns:
        True if any decorator is ``enforce_contract`` (bare, called,
        or attribute-qualified).
    """
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name) and target.id == "enforce_contract":
            return True
        if isinstance(target, ast.Attribute) and target.attr == "enforce_contract":
            return True
    return False


def _module_violations(path: Path) -> list[str]:
    """Collect contract-rule violations in one module.

    Args:
        path: Python file to scan.

    Returns:
        Violation messages, one per unenforced public mutation.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not node.name.startswith(MUTATION_PREFIXES):
            continue
        if not _has_enforce_contract(node):
            violations.append(
                f"{path}:{node.lineno} public mutation '{node.name}' lacks @enforce_contract"
            )
    return violations


def run_contract_rules(project_root: Path) -> int:
    """Run the contract-enforcement guard rule over a project tree.

    Args:
        project_root: Project root containing ``src/tankpit_bot``.

    Returns:
        Number of violations found (0 means the rule passes).
    """
    violations: list[str] = []
    for package in GUARDED_PACKAGES:
        package_root = project_root / "src" / "tankpit_bot" / package
        if not package_root.is_dir():
            continue
        for module_path in sorted(package_root.rglob("*.py")):
            violations.extend(_module_violations(module_path))
    for violation in violations:
        sys.stdout.write(f"contract_rule_violation {violation}\n")
    return len(violations)


__all__ = [
    "GUARDED_PACKAGES",
    "MUTATION_PREFIXES",
    "run_contract_rules",
]
