from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

from monorepo_guards import Violation


class ConfigHelpersRule:
    """Ban local reimplementation of centralized config parsing helpers.

    All services must import _parse_int, _parse_float, _parse_bool, _parse_str
    from platform_core.config instead of implementing their own versions.
    """

    name = "config-helpers"

    _BANNED_FUNCTION_NAMES: ClassVar[set[str]] = {
        "_parse_int",
        "_parse_float",
        "_parse_bool",
        "_parse_str",
        "_clean_env_value",  # transcript-api pattern that should be removed
    }

    _ALLOW_SUFFIXES: ClassVar[set[str]] = {
        "src/platform_core/config/_utils.py",  # Where the centralized versions live
        "tests/test_config.py",  # Tests for platform_core
        "tests/test_config_helpers_rules.py",  # Tests for this rule
    }

    def _is_allowed(self, path: Path) -> bool:
        posix = path.as_posix()
        resolved = path.resolve().as_posix()
        for suffix in self._ALLOW_SUFFIXES:
            if posix.endswith(suffix) or resolved.endswith(suffix):
                return True
        return False

    def _check_function_def(self, path: Path, node: ast.FunctionDef) -> list[Violation]:
        if node.name in self._BANNED_FUNCTION_NAMES:
            return [
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind="config-helper-duplicate",
                    line=f"Use platform_core.config.{node.name} instead of local implementation",
                )
            ]
        return []

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []
        for path in files:
            if self._is_allowed(path):
                continue

            try:
                text = path.read_text(encoding="utf-8", errors="strict")
            except OSError as exc:
                raise RuntimeError(f"Failed to read {path}: {exc}") from exc

            try:
                tree = ast.parse(text, filename=str(path))
            except SyntaxError as exc:
                raise RuntimeError(f"Failed to parse {path}: {exc}") from exc

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    out.extend(self._check_function_def(path, node))

        return out


__all__ = ["ConfigHelpersRule"]
