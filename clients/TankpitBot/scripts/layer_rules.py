"""Guard rule: the base layer stays the base layer.

At the 2026-08-06 audit every one of the 17 packages under
``src/tankpit_bot`` sat in a single strongly-connected component: any
package could reach any other, so there was no layering to violate.
Pulling packages out of that component is ongoing work; this rule
protects the part that is finished.

Each entry in :data:`BASE_LAYER` names a package and the complete set
of ``tankpit_bot`` packages it is allowed to import. Anything else is a
violation. The declarations are deliberately tight -- ``types``,
``wire`` and ``contracts`` allow NOTHING, which is what makes them
usable from anywhere without risking a cycle.

This is not an allowlist of known-bad files. It states an invariant
that currently holds, and fails the moment an edit breaks it. Adding a
package here is a commitment that it has been lifted clear of the
remaining component, not a way to record that it has not.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

#: Package -> the tankpit_bot packages it may import. A package listed
#: here with an empty set is a true leaf: it may import nothing from
#: this codebase at all.
BASE_LAYER: dict[str, frozenset[str]] = {
    "types": frozenset(),
    "wire": frozenset(),
    "contracts": frozenset(),
    "facts": frozenset({"contracts"}),
    "container": frozenset({"wire"}),
    "bus": frozenset({"types"}),
}

PACKAGE_ROOT = Path("src") / "tankpit_bot"


def _imported_packages(source: str, package_root: Path) -> set[str]:
    """Return the ``tankpit_bot`` packages one module imports.

    Both import spellings count. ``from tankpit_bot.state import X``
    names the package directly; ``from tankpit_bot import state`` names
    it as an alias, and a checker that misses the second form
    under-reports -- which is how three cycles stayed hidden through the
    first two passes of this work.

    Args:
        source: Module source text.
        package_root: Path to ``src/tankpit_bot``, used to tell a
            package name from a top-level module name.

    Returns:
        Package names, excluding top-level modules.
    """
    found: set[str] = set()
    modules: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            # ``module`` is None exactly when ``level`` is set (``from .
            # import x``), so the two skips are one condition -- split
            # apart, the second arm is unreachable.
            if node.level or node.module is None:
                continue
            if node.module == "tankpit_bot":
                modules.extend(f"tankpit_bot.{alias.name}" for alias in node.names)
            else:
                modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    for module in modules:
        if not module.startswith("tankpit_bot."):
            continue
        head = module.split(".")[1]
        if (package_root / head).is_dir():
            found.add(head)
    return found


def evaluate(package_root: Path) -> list[str]:
    """Check every declared base-layer package against its allowance.

    Args:
        package_root: Path to ``src/tankpit_bot``.

    Returns:
        Violation messages, one per offending import, in stable order.
    """
    violations: list[str] = []
    for package in sorted(BASE_LAYER):
        allowed = BASE_LAYER[package]
        directory = package_root / package
        if not directory.is_dir():
            violations.append(f"{package} is declared in BASE_LAYER but does not exist")
            continue
        for path in sorted(directory.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            source = path.read_text(encoding="utf-8")
            for imported in sorted(_imported_packages(source, package_root)):
                if imported == package or imported in allowed:
                    continue
                permitted = ", ".join(sorted(allowed)) or "nothing"
                violations.append(
                    f"{path.as_posix()} imports '{imported}': base-layer package "
                    f"'{package}' may import {permitted}"
                )
    return violations


def run_layer_rules(project_root: Path) -> int:
    """Run the base-layer guard rule over a project tree.

    Args:
        project_root: Project root containing ``src/tankpit_bot``.

    Returns:
        Number of violations found (0 means the rule passes).
    """
    package_root = project_root / PACKAGE_ROOT
    if not package_root.is_dir():
        return 0
    violations = evaluate(package_root)
    for violation in violations:
        sys.stdout.write(f"layer_violation {violation}\n")
    return len(violations)


__all__ = [
    "BASE_LAYER",
    "PACKAGE_ROOT",
    "evaluate",
    "run_layer_rules",
]
