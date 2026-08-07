"""Guard rule: no shims, no re-exports, no legacy markers.

The standing user rule is "no back-compat shims, no thin wrappers, no
fallbacks, no legacy code, no type alias, no re-exports". Every other
coding standard in [[coding-standards]] is machine-checked; this one
was not, and an unenforced rule is the one that rots — the 400-600
line ceiling went from a 40-file backlog to 77 in the six days it was
documented-but-unchecked.

Three things are checked, chosen because each is unambiguous from the
syntax alone. A rule that needs a human to adjudicate would need an
allowlist, and an allowlist is the thing this project refuses.

1. **Legacy vocabulary.** ``back-compat``, ``backward compatible``,
   ``deprecated``, ``legacy``, ``for compatibility`` anywhere in
   ``src/`` or ``scripts/``. Prose announcing a shim is a shim.
2. **Self-named aliases.** ``X = X`` at module scope, which exists only
   to re-export an imported name under the name it already had.
3. **Renamed re-exports.** ``NEW = OLD`` at module scope where ``OLD``
   is an imported name and ``NEW`` is listed in ``__all__`` — the exact
   shape of "export someone else's symbol under our own name".

``_test_hooks`` modules are exempt from rule 3 *structurally*, not by
name-listing: the DI pattern this codebase uses IS binding an imported
implementation to a patchable module attribute, so the alias is the
seam ([[testing-patterns]]). The exemption is the whole module kind,
not a list of blessed symbols, so it cannot silently grow.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

SCANNED_ROOTS: tuple[str, ...] = ("src", "scripts")

#: Words that announce a shim. Matched case-insensitively on whole words
#: so ``backward guard`` (a real algorithmic term in ``validate``) does
#: not trip ``backward compatible``.
LEGACY_PATTERNS: tuple[str, ...] = (
    r"back[- ]?compat\w*",
    r"backwards?[- ]compatib\w*",
    r"deprecat\w+",
    r"\blegacy\b",
    r"for compatibility",
    r"kept for (?:api|signature) compatibility",
)

_LEGACY_RE = re.compile("|".join(LEGACY_PATTERNS), re.IGNORECASE)


def _iter_python_files(root: Path) -> list[Path]:
    """Return every scanned Python file under a project root.

    Args:
        root: Project root containing the scanned directories.

    Returns:
        Sorted paths, excluding compiled-artifact directories.
    """
    found: list[Path] = []
    for name in SCANNED_ROOTS:
        directory = root / name
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            found.append(path)
    return found


def find_legacy_markers(source: str) -> list[tuple[int, str]]:
    """Return every line announcing legacy or back-compat intent.

    Args:
        source: Module source text.

    Returns:
        ``(line number, matched text)`` pairs in file order.
    """
    hits: list[tuple[int, str]] = []
    for number, line in enumerate(source.splitlines(), 1):
        match = _LEGACY_RE.search(line)
        if match is not None:
            hits.append((number, match.group(0)))
    return hits


def _imported_names(tree: ast.Module) -> set[str]:
    """Return every name a module binds through an import.

    Args:
        tree: Parsed module.

    Returns:
        Bound names, using the ``as`` name where one is given.
    """
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[-1])
    return names


def _exported_names(tree: ast.Module) -> set[str]:
    """Return the string literals a module lists in ``__all__``.

    Args:
        tree: Parsed module.

    Returns:
        Exported names, empty when the module declares no ``__all__``.
    """
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != "__all__":
            continue
        for element in ast.walk(node.value):
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                names.add(element.value)
    return names


def find_reexports(source: str) -> list[tuple[int, str]]:
    """Return module-level aliases that re-export an imported name.

    Args:
        source: Module source text.

    Returns:
        ``(line number, message)`` pairs in file order.
    """
    tree = ast.parse(source)
    imported = _imported_names(tree)
    exported = _exported_names(tree)

    hits: list[tuple[int, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target, value = node.targets[0], node.value
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Name):
            continue
        if target.id == value.id:
            hits.append((node.lineno, f"'{target.id} = {value.id}' re-exports its own name"))
        elif value.id in imported and target.id in exported:
            hits.append(
                (
                    node.lineno,
                    f"'{target.id} = {value.id}' re-exports an imported name; "
                    f"import '{value.id}' where it is used",
                )
            )
    return hits


def evaluate(root: Path) -> list[str]:
    """Check every scanned module for shim markers and re-exports.

    Args:
        root: Project root containing ``src`` and ``scripts``.

    Returns:
        Violation messages, one per offending line, in stable order.
    """
    violations: list[str] = []
    for path in _iter_python_files(root):
        source = path.read_text(encoding="utf-8")
        name = path.as_posix()
        # The module that DEFINES the banned vocabulary necessarily
        # contains it, exactly as `state_sentinel_rules` exempts the
        # module that owns the sentinel idiom. This is the pattern
        # owner, not an allowlist: it cannot grow to a second entry
        # without moving the patterns themselves.
        if path.name != Path(__file__).name:
            for number, text in find_legacy_markers(source):
                violations.append(f"{name}:{number} legacy marker '{text}'")
        if path.name == "_test_hooks.py" or "_test_hooks" in path.parts:
            continue
        for number, message in find_reexports(source):
            violations.append(f"{name}:{number} {message}")
    return violations


def run_shim_rules(project_root: Path) -> int:
    """Run the shim guard rule over a project tree.

    Args:
        project_root: Project root containing the scanned directories.

    Returns:
        Number of violations found (0 means the rule passes).
    """
    violations = evaluate(project_root)
    for violation in violations:
        sys.stdout.write(f"shim_violation {violation}\n")
    return len(violations)


__all__ = [
    "LEGACY_PATTERNS",
    "SCANNED_ROOTS",
    "evaluate",
    "find_legacy_markers",
    "find_reexports",
    "run_shim_rules",
]
