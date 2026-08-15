from __future__ import annotations

import ast
from pathlib import Path

from monorepo_guards import Violation


class WorkerImportsRule:
    """Prevent direct redis/rq imports outside platform_workers.

    Services should use platform_workers.redis and platform_workers.rq_harness
    instead of importing redis or rq directly. This ensures consistent connection
    handling, protocol-based typing, and testability.
    """

    name = "worker-imports"

    def _should_check(self, path: Path) -> bool:
        posix = path.as_posix()
        return not ("/tests/" in posix or "/scripts/" in posix)

    def _check_import_node(self, path: Path, node: ast.Import) -> list[Violation]:
        violations: list[Violation] = []
        for alias in node.names:
            if alias.name == "redis" or alias.name.startswith("redis."):
                violations.append(
                    Violation(
                        file=path,
                        line_no=node.lineno,
                        kind="direct-redis-import",
                        line=f"import {alias.name}: use platform_workers.redis instead",
                    )
                )
            elif alias.name == "rq" or alias.name.startswith("rq."):
                violations.append(
                    Violation(
                        file=path,
                        line_no=node.lineno,
                        kind="direct-rq-import",
                        line=f"import {alias.name}: use platform_workers.rq_harness instead",
                    )
                )
        return violations

    def _check_import_from_node(self, path: Path, node: ast.ImportFrom) -> list[Violation]:
        if node.module is None:
            return []
        # A relative import names a sibling module, not the third-party package:
        # "from .redis import X" inside platform_workers reaches its own redis.py
        # and carries module == "redis" with level == 1. Without this check the
        # rule reports the canonical wrapper itself as a direct redis import.
        if node.level > 0:
            return []
        if node.module == "redis" or node.module.startswith("redis."):
            return [
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind="direct-redis-import",
                    line=f"from {node.module}: use platform_workers.redis instead",
                )
            ]
        if node.module == "rq" or node.module.startswith("rq."):
            return [
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind="direct-rq-import",
                    line=f"from {node.module}: use platform_workers.rq_harness instead",
                )
            ]
        return []

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []
        for path in files:
            if not self._should_check(path):
                continue
            try:
                tree = ast.parse(
                    path.read_text(encoding="utf-8", errors="strict"), filename=str(path)
                )
            except SyntaxError as exc:
                raise RuntimeError(f"failed to parse {path}: {exc}") from exc
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    out.extend(self._check_import_node(path, node))
                elif isinstance(node, ast.ImportFrom):
                    out.extend(self._check_import_from_node(path, node))
        return out


__all__ = ["WorkerImportsRule"]
