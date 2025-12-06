from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

from monorepo_guards import Violation


class RedisRule:
    """Prevent duplicated Redis Protocol definitions outside platform_workers.redis."""

    name = "redis"

    _CANONICAL_PATH: ClassVar[str] = "libs/platform_workers/src/platform_workers/redis.py"

    def _is_canonical(self, path: Path) -> bool:
        posix = path.as_posix()
        resolved = path.resolve().as_posix()
        return posix.endswith(self._CANONICAL_PATH) or resolved.endswith(self._CANONICAL_PATH)

    def _should_check(self, path: Path) -> bool:
        posix = path.as_posix()
        if "/tests/" in posix or "/scripts/" in posix:
            return False
        return not self._is_canonical(path)

    def _class_is_protocol(self, node: ast.ClassDef) -> bool:
        return any(
            (isinstance(base, ast.Name) and base.id == "Protocol")
            or (isinstance(base, ast.Attribute) and base.attr == "Protocol")
            for base in node.bases
        )

    def _class_name_mentions_redis(self, node: ast.ClassDef) -> bool:
        return "redis" in node.name.lower()

    def _scan_node(self, path: Path, node: ast.AST) -> list[Violation]:
        if (
            self._should_check(path)
            and isinstance(node, ast.ClassDef)
            and self._class_name_mentions_redis(node)
            and self._class_is_protocol(node)
        ):
            return [
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind="redis-protocol-duplicate",
                    line=node.name,
                )
            ]
        return []

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="strict"), filename=str(path))
            for node in ast.walk(tree):
                out.extend(self._scan_node(path, node))
        return out


__all__ = ["RedisRule"]
