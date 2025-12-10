"""Guard rule to ban unittest.mock imports and monkeypatching.

This rule enforces that NO mocking or monkeypatching is used in test suites.
Tests must use dependency injection via test hooks instead of runtime patching.
Production code sets hooks to real implementations at startup, tests set them to fakes.
"""

from __future__ import annotations

import ast
from pathlib import Path

from monorepo_guards import Violation
from monorepo_guards.util import read_lines


class MockBanRule:
    """Rule to ban all unittest.mock imports and monkeypatching."""

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

    def _check_monkeypatch_call(self, path: Path, node: ast.Call) -> list[Violation]:
        """Check for monkeypatch.setattr and similar calls."""
        violations: list[Violation] = []

        func = node.func
        if not isinstance(func, ast.Attribute):
            return violations

        value = func.value
        if not isinstance(value, ast.Name):
            return violations
        if value.id != "monkeypatch":
            return violations

        banned_methods = frozenset(
            {"setattr", "delattr", "setenv", "delenv", "setitem", "delitem", "syspath_prepend"}
        )
        if func.attr in banned_methods:
            violations.append(
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind="monkeypatch-banned",
                    line=f"monkeypatch.{func.attr}",
                )
            )

        return violations

    def _check_ast(self, path: Path, tree: ast.AST) -> list[Violation]:
        """Check an AST for mock import and monkeypatch violations."""
        violations: list[Violation] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                violations.extend(self._check_import_node(path, node))
            elif isinstance(node, ast.ImportFrom):
                violations.extend(self._check_import_from_node(path, node))
            elif isinstance(node, ast.Call):
                violations.extend(self._check_monkeypatch_call(path, node))

        return violations

    def _is_test_file(self, path: Path) -> bool:
        """Check if this is a test file."""
        posix = path.as_posix()
        return "/tests/" in posix or "\\tests\\" in str(path)

    def run(self, files: list[Path]) -> list[Violation]:
        """Check all test files for mock import and monkeypatch violations."""
        out: list[Violation] = []
        for path in files:
            if not self._is_test_file(path):
                continue
            lines = read_lines(path)
            source = "\n".join(lines)
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError as exc:
                raise RuntimeError(f"failed to parse {path}: {exc}") from exc
            out.extend(self._check_ast(path, tree))
        return out


__all__ = ["MockBanRule"]
