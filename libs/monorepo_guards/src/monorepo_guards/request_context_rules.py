from __future__ import annotations

import ast
from pathlib import Path

from monorepo_guards import Violation
from monorepo_guards.util import read_lines


class RequestContextRule:
    """Ensure services use platform_core.request_context.RequestIdMiddleware."""

    name = "request-context"

    def _is_request_id_middleware(self, node: ast.ClassDef) -> bool:
        return node.name == "RequestIdMiddleware"

    def run(self, files: list[Path]) -> list[Violation]:
        violations: list[Violation] = []
        for path in files:
            if "platform_core" in path.parts:
                continue
            if path.name != "middleware.py":
                continue
            lines = read_lines(path)
            source = "\n".join(lines)
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError as exc:
                raise RuntimeError(f"failed to parse {path}: {exc}") from exc
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and self._is_request_id_middleware(node):
                    idx = node.lineno - 1
                    line = lines[idx].rstrip("\n") if 0 <= idx < len(lines) else ""
                    violations.append(
                        Violation(
                            file=path,
                            line_no=node.lineno,
                            kind="local-request-id-middleware",
                            line=line,
                        )
                    )
        return violations


__all__ = ["RequestContextRule"]
