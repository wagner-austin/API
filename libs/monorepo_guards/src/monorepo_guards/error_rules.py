from __future__ import annotations

import ast
from pathlib import Path

from monorepo_guards import Violation
from monorepo_guards.util import read_lines


class ErrorsRule:
    """Enforce centralized error handling by forbidding local error modules and types."""

    name = "errors"

    def _is_platform_core_errors_module(self, path: Path) -> bool:
        return path.name == "errors.py" and "platform_core" in path.parts

    def _is_local_error_module(self, path: Path) -> bool:
        if self._is_platform_core_errors_module(path):
            return False
        if path.name == "errors.py":
            return True
        return any(part == "errors" for part in path.parts)

    def _path_violations(self, path: Path) -> list[Violation]:
        if not self._is_local_error_module(path):
            return []
        return [
            Violation(
                file=path,
                line_no=1,
                kind="local-errors-module",
                line="Local error modules are forbidden; use platform_core.errors",
            )
        ]

    def _class_violations(
        self,
        path: Path,
        lines: list[str],
        tree: ast.AST,
    ) -> list[Violation]:
        violations: list[Violation] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name == "AppError" and not self._is_platform_core_errors_module(path):
                    line = lines[node.lineno - 1] if 0 <= node.lineno - 1 < len(lines) else ""
                    violations.append(
                        Violation(
                            file=path,
                            line_no=node.lineno,
                            kind="local-app-error",
                            line=line.rstrip("\n"),
                        )
                    )
                if node.name == "ErrorCode" and not self._is_platform_core_errors_module(path):
                    line = lines[node.lineno - 1] if 0 <= node.lineno - 1 < len(lines) else ""
                    violations.append(
                        Violation(
                            file=path,
                            line_no=node.lineno,
                            kind="local-error-code",
                            line=line.rstrip("\n"),
                        )
                    )
        return violations

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []
        for path in files:
            out.extend(self._path_violations(path))
            if self._is_platform_core_errors_module(path):
                continue

            lines = read_lines(path)
            source = "\n".join(lines)
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError as exc:
                raise RuntimeError(f"failed to parse {path}: {exc}") from exc

            out.extend(self._class_violations(path, lines, tree))
        return out


__all__ = ["ErrorsRule"]
