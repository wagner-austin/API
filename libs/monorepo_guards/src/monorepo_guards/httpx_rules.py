from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

from monorepo_guards import Violation


class HttpxRule:
    """Prevent direct httpx imports outside the canonical data_bank_client module.

    All HTTP requests should use the centralized DataBankClient from platform_core
    to ensure consistent correlation headers, error handling, and retry logic.

    Exceptions:
    - Services that need direct streaming access to external APIs may be allowlisted
      in _ALLOWED_PATHS until a streaming API is added to DataBankClient.
    """

    name = "httpx"

    _CANONICAL_PATH: ClassVar[str] = "libs/platform_core/src/platform_core/data_bank_client.py"

    # No service code should import httpx directly; use DataBankClient instead.
    _ALLOWED_PATHS: ClassVar[frozenset[str]] = frozenset([])

    def _is_canonical(self, path: Path) -> bool:
        posix = path.as_posix()
        resolved = path.resolve().as_posix()
        return posix.endswith(self._CANONICAL_PATH) or resolved.endswith(self._CANONICAL_PATH)

    def _should_check(self, path: Path) -> bool:
        posix = path.as_posix()
        if "/tests/" in posix or "/scripts/" in posix:
            return False
        return not self._is_canonical(path)

    def _check_import_node(self, path: Path, node: ast.Import) -> list[Violation]:
        violations: list[Violation] = []
        for alias in node.names:
            if alias.name == "httpx" or alias.name.startswith("httpx."):
                violations.append(
                    Violation(
                        file=path,
                        line_no=node.lineno,
                        kind="httpx-direct-import",
                        line=f"import {alias.name}",
                    )
                )
        return violations

    def _check_import_from_node(self, path: Path, node: ast.ImportFrom) -> list[Violation]:
        if node.module is None:
            return []
        if node.module == "httpx" or node.module.startswith("httpx."):
            return [
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind="httpx-direct-import",
                    line=f"from {node.module} import ...",
                )
            ]
        return []

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []
        for path in files:
            if not self._should_check(path):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8", errors="strict"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    out.extend(self._check_import_node(path, node))
                elif isinstance(node, ast.ImportFrom):
                    out.extend(self._check_import_from_node(path, node))
        return out


__all__ = ["HttpxRule"]
