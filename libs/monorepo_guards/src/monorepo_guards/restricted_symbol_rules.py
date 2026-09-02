"""Guard rule restricting designated symbols to their owning module.

Some functions are raw mechanisms whose every legitimate use is meant
to flow through one sanctioned wrapper in one owning module. Referencing
the raw mechanism anywhere else silently bypasses the wrapper's
bookkeeping — the precedent is TankpitBot's resource lock
(2026-09-02): ``clear_resource_target`` wipes the lock with no
diagnostic, while the sanctioned ``release_collect_plan`` enumerates
and emits every drop. A recon module calling the raw clear on a lock it
never set amplified a nine-minute livelock; the fix removed the call,
and this rule keeps the class removed.

The restriction table maps a symbol name to the path suffixes allowed
to reference it. Adding a restricted symbol is adding a row. Matching
is AST-based (imports, names, attribute access), so prose mentions in
docstrings and comments never trip it.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

from monorepo_guards import Violation
from monorepo_guards.util import parse_source

RESTRICTED_SYMBOLS: dict[str, tuple[str, ...]] = {
    "clear_resource_target": ("bot/ai/intent.py",),
}
"""Symbol name to the path suffixes allowed to reference it.

Every other file — src, tests, and scripts alike — must go through the
symbol's sanctioned wrapper or not touch it at all. Zero exemptions:
no test references a restricted symbol either (verified for
``clear_resource_target`` at introduction; tests exercise the intent
module's public release paths instead).
"""


class RestrictedSymbolRule:
    """Flag references to owner-restricted symbols outside their owner."""

    name = "restricted-symbols"

    _restricted: ClassVar[dict[str, tuple[str, ...]]] = RESTRICTED_SYMBOLS

    def _allowed_here(self, posix_path: str, symbol: str) -> bool:
        """Return whether this file may reference the symbol.

        Args:
            posix_path: The file's path with forward slashes.
            symbol: Restricted symbol name.

        Returns:
            True when the path ends with one of the symbol's allowed
            suffixes.
        """
        return any(posix_path.endswith(suffix) for suffix in self._restricted[symbol])

    def _node_references(self, node: ast.AST) -> list[tuple[str, int]]:
        """Return (symbol, line) for restricted references on this node.

        Imports, bare names, and attribute access all count — each is
        a way to reach the raw mechanism.

        Args:
            node: AST node to inspect.

        Returns:
            Restricted symbol names the node references with their
            line numbers (possibly several for one import statement).
        """
        if isinstance(node, ast.ImportFrom):
            return [
                (alias.name, node.lineno) for alias in node.names if alias.name in self._restricted
            ]
        if isinstance(node, ast.Name) and node.id in self._restricted:
            return [(node.id, node.lineno)]
        if isinstance(node, ast.Attribute) and node.attr in self._restricted:
            return [(node.attr, node.lineno)]
        return []

    def run(self, files: list[Path]) -> list[Violation]:
        """Scan every file for out-of-owner restricted-symbol references.

        Args:
            files: Python files under guard.

        Returns:
            One violation per out-of-owner reference.
        """
        violations: list[Violation] = []
        for path in files:
            posix_path = path.as_posix()
            tree = parse_source(path)
            for node in ast.walk(tree):
                for symbol, line_no in self._node_references(node):
                    if self._allowed_here(posix_path, symbol):
                        continue
                    violations.append(
                        Violation(
                            file=path,
                            line_no=line_no,
                            kind=f"restricted-symbol-{symbol}",
                            line=symbol,
                        )
                    )
        return violations


__all__ = ["RESTRICTED_SYMBOLS", "RestrictedSymbolRule"]
