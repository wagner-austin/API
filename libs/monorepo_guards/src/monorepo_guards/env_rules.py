from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

from monorepo_guards import Violation


class EnvRule:
    """Ban direct env/TOML access everywhere except minimal core files."""

    name = "env"

    _ALLOW_SUFFIXES: ClassVar[set[str]] = {
        "src/platform_core/config/_utils.py",
        "src/platform_core/config/_test_hooks.py",
        "tests/test_config.py",
        "src/monorepo_guards/config_loader.py",
        "tests/test_config_loader.py",
        "tests/test_env_rules.py",
    }

    def _is_allowed(self, path: Path) -> bool:
        posix = path.as_posix()
        resolved = path.resolve().as_posix()
        for suffix in self._ALLOW_SUFFIXES:
            if posix.endswith(suffix) or resolved.endswith(suffix):
                return True
        return False

    def _attr_violations(self, path: Path, node: ast.Attribute) -> list[Violation]:
        target = node.value
        if not isinstance(target, ast.Name):
            return []
        if target.id != "os" or node.attr not in {"getenv", "environ"}:
            return []
        return [Violation(file=path, line_no=node.lineno, kind="env-access-banned", line="")]

    def _importfrom_violations(self, path: Path, node: ast.ImportFrom) -> list[Violation]:
        if node.module == "os":
            hits = [alias for alias in node.names if alias.name in {"getenv", "environ"}]
            if hits:
                return [
                    Violation(file=path, line_no=node.lineno, kind="env-access-banned", line="")
                ]
        if node.module == "tomllib":
            return [Violation(file=path, line_no=node.lineno, kind="tomllib-banned", line="")]
        return []

    def _import_violations(self, path: Path, node: ast.Import) -> list[Violation]:
        if any(alias.name == "tomllib" for alias in node.names):
            return [Violation(file=path, line_no=node.lineno, kind="tomllib-banned", line="")]
        return []

    def _call_violations(self, path: Path, node: ast.Call) -> list[Violation]:
        if not (isinstance(node.func, ast.Name) and node.func.id == "__import__" and node.args):
            return []
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and arg.value == "tomllib":
            return [Violation(file=path, line_no=node.lineno, kind="tomllib-banned", line="")]
        return []

    def _scan_node(self, path: Path, node: ast.AST) -> list[Violation]:
        if self._is_allowed(path):
            return []

        if isinstance(node, ast.Attribute):
            return self._attr_violations(path, node)
        if isinstance(node, ast.ImportFrom):
            return self._importfrom_violations(path, node)
        if isinstance(node, ast.Import):
            return self._import_violations(path, node)
        if isinstance(node, ast.Call):
            return self._call_violations(path, node)
        return []

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []
        for path in files:
            lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
            tree = ast.parse("\n".join(lines), filename=str(path))
            for node in ast.walk(tree):
                out.extend(self._scan_node(path, node))
        return out


__all__ = ["EnvRule"]
