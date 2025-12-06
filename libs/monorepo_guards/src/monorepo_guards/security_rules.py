from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

from monorepo_guards import Violation


class SecurityRule:
    """Enforce centralized security patterns across the monorepo.

    Ensures:
    1. API key authentication uses create_api_key_dependency from platform_core.security
    2. No custom APIKeyMiddleware implementations (use the centralized dependency)
    3. No custom BaseHTTPMiddleware for auth (use FastAPI dependencies)
    """

    name = "security"

    _PLATFORM_CORE_SECURITY: ClassVar[str] = "libs/platform_core/src/platform_core/security.py"

    def _is_canonical_security(self, path: Path) -> bool:
        posix = path.as_posix()
        resolved = path.resolve().as_posix()
        return posix.endswith(self._PLATFORM_CORE_SECURITY) or resolved.endswith(
            self._PLATFORM_CORE_SECURITY
        )

    def _should_check(self, path: Path) -> bool:
        posix = path.as_posix()
        if "/tests/" in posix or "/scripts/" in posix:
            return False
        return not self._is_canonical_security(path)

    def _check_class_def(self, path: Path, node: ast.ClassDef) -> list[Violation]:
        violations: list[Violation] = []

        # Check for custom APIKeyMiddleware class
        if "APIKey" in node.name and "Middleware" in node.name:
            violations.append(
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind="custom-api-key-middleware",
                    line=f"class {node.name}: use create_api_key_dependency from platform_core",
                )
            )

        # Check for BaseHTTPMiddleware subclass with auth-related name
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "BaseHTTPMiddleware":
                name_lower = node.name.lower()
                if "auth" in name_lower or "apikey" in name_lower or "api_key" in name_lower:
                    violations.append(
                        Violation(
                            file=path,
                            line_no=node.lineno,
                            kind="auth-middleware-not-dependency",
                            line=f"class {node.name}: use create_api_key_dependency instead",
                        )
                    )

        return violations

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
                if isinstance(node, ast.ClassDef):
                    out.extend(self._check_class_def(path, node))
        return out


__all__ = ["SecurityRule"]
