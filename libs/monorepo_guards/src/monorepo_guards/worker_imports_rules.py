from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

from monorepo_guards import Violation


class WorkerImportsRule:
    """Prevent direct redis/rq imports outside platform_workers.

    Services should use platform_workers.redis and platform_workers.rq_harness
    instead of importing redis or rq directly. This ensures consistent connection
    handling, protocol-based typing, and testability.
    """

    name = "worker-imports"

    _CANONICAL_PATHS: ClassVar[set[str]] = {
        "libs/platform_workers/src/platform_workers/redis.py",
        "libs/platform_workers/src/platform_workers/rq_harness.py",
        "libs/platform_workers/src/platform_workers/testing.py",
        "libs/platform_workers/src/platform_workers/_fakes.py",
    }

    def _is_canonical(self, path: Path) -> bool:
        posix = path.as_posix()
        resolved = path.resolve().as_posix()
        for canonical in self._CANONICAL_PATHS:
            if posix.endswith(canonical) or resolved.endswith(canonical):
                return True
        return False

    def _should_check(self, path: Path) -> bool:
        posix = path.as_posix()
        if "/tests/" in posix or "/scripts/" in posix:
            return False
        return not self._is_canonical(path)

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
