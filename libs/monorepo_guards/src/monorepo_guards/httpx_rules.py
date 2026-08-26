from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

from monorepo_guards import Violation
from monorepo_guards.util import parse_source


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
    #
    # Exception: a dependency-injection hook module has to name the transport
    # types it injects, or the Protocol it declares cannot match the real
    # signature. That applies to the whole hook module regardless of how it is
    # divided — a registry large enough to split into its contracts and its
    # production implementations does not stop being a hook module.
    _ALLOWED_PATHS: ClassVar[frozenset[str]] = frozenset(
        ["_test_hooks.py", "_hook_protocols.py", "_hook_defaults.py"]
    )

    def _is_canonical(self, path: Path) -> bool:
        posix = path.as_posix()
        resolved = path.resolve().as_posix()
        return posix.endswith(self._CANONICAL_PATH) or resolved.endswith(self._CANONICAL_PATH)

    def _is_allowed(self, path: Path) -> bool:
        name = path.name
        return name in self._ALLOWED_PATHS

    def _should_check(self, path: Path) -> bool:
        posix = path.as_posix()
        if "/tests/" in posix or "/scripts/" in posix:
            return False
        if self._is_allowed(path):
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
            tree = parse_source(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    out.extend(self._check_import_node(path, node))
                elif isinstance(node, ast.ImportFrom):
                    out.extend(self._check_import_from_node(path, node))
        return out


__all__ = ["HttpxRule"]
