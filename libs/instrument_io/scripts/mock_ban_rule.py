"""Guard rule to ban unittest.mock imports in instrument_io.

This rule enforces that NO mocking is used in the test suite.
All tests must use real fixture files instead of mocks.
"""

from __future__ import annotations

import ast
from pathlib import Path


class Violation:
    """Represents a guard rule violation.

    This is a local copy of the Violation structure from monorepo_guards.
    We cannot import from monorepo_guards in this context since the
    guard.py dynamically loads it.
    """

    def __init__(
        self,
        *,
        file: Path,
        line_no: int,
        kind: str,
        line: str,
    ) -> None:
        self.file = file
        self.line_no = line_no
        self.kind = kind
        self.line = line


def _read_lines(path: Path) -> list[str] | None:
    """Read lines from a file.

    Returns:
        List of lines or None if file cannot be read.
    """
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8-sig", errors="strict")
    return text.splitlines()


class MockBanRule:
    """Rule to ban all unittest.mock imports."""

    name = "mock-ban"

    def _check_import_node(self, path: Path, node: ast.Import) -> list[Violation]:
        """Check regular import statements for mock imports."""
        violations: list[Violation] = []

        for alias in node.names:
            if alias.name == "mock" or alias.name.startswith("unittest.mock"):
                violations.append(
                    Violation(
                        file=path,
                        line_no=node.lineno,
                        kind="import-mock",
                        line="",
                    )
                )

        return violations

    def _check_import_from_node(
        self,
        path: Path,
        node: ast.ImportFrom,
    ) -> list[Violation]:
        """Check from ... import statements for mock imports."""
        violations: list[Violation] = []

        if node.module is None:
            return violations

        if node.module.startswith("unittest.mock") or node.module == "unittest.mock":
            violations.append(
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind="import-mock",
                    line="",
                )
            )
        elif node.module == "unittest":
            for alias in node.names:
                if alias.name == "mock":
                    violations.append(
                        Violation(
                            file=path,
                            line_no=node.lineno,
                            kind="import-mock",
                            line="",
                        )
                    )

        return violations

    def _check_ast(self, path: Path, tree: ast.AST) -> list[Violation]:
        """Check an AST for mock import violations."""
        violations: list[Violation] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                violations.extend(self._check_import_node(path, node))
            elif isinstance(node, ast.ImportFrom):
                violations.extend(self._check_import_from_node(path, node))

        return violations

    def run(self, files: list[Path]) -> list[Violation]:
        """Check all files for mock import violations."""
        out: list[Violation] = []
        for path in files:
            lines = _read_lines(path)
            if lines is None:
                continue
            source = "\n".join(lines)
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError as exc:
                raise RuntimeError(f"failed to parse {path}: {exc}") from exc
            out.extend(self._check_ast(path, tree))
        return out


__all__ = ["MockBanRule", "Violation"]
