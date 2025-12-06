from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

from monorepo_guards import Violation


class JsonRule:
    """Ban direct json.loads outside the shared helper."""

    name = "json"

    _ALLOW_SUFFIXES: ClassVar[set[str]] = {
        "src/platform_core/json_utils.py",
        "tests/test_json_utils.py",
    }

    def _is_allowed(self, path: Path) -> bool:
        posix = path.as_posix()
        resolved = path.resolve().as_posix()
        for suffix in self._ALLOW_SUFFIXES:
            if posix.endswith(suffix) or resolved.endswith(suffix):
                return True
        return False

    def _call_is_json_loads(self, node: ast.Call) -> bool:
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            return func.value.id == "json" and func.attr == "loads"
        return isinstance(func, ast.Name) and func.id == "loads"

    def _import_is_stdlib_json(self, node: ast.Import | ast.ImportFrom) -> bool:
        if isinstance(node, ast.Import):
            return any(alias.name == "json" for alias in node.names)
        return node.module == "json"

    def _scan_node(self, path: Path, node: ast.AST) -> list[Violation]:
        if self._is_allowed(path):
            return []
        if isinstance(node, (ast.Import, ast.ImportFrom)) and self._import_is_stdlib_json(node):
            return [
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind="json-import-banned",
                    line="",
                )
            ]
        if isinstance(node, ast.Call) and self._call_is_json_loads(node):
            return [Violation(file=path, line_no=node.lineno, kind="json-loads-banned", line="")]
        return []

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="strict"), filename=str(path))
            for node in ast.walk(tree):
                out.extend(self._scan_node(path, node))
        return out


__all__ = ["JsonRule"]
