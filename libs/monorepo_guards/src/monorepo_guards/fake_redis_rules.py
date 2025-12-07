"""Guard rules for enforcing proper FakeRedis usage in tests.

This module enforces three key patterns:

1. **FakeRedis Extension**: Custom Redis stub classes in tests must extend
   FakeRedis from platform_workers.testing, not implement Redis methods
   from scratch. This ensures call tracking is available.

2. **Call Verification**: Test files that use FakeRedis must call
   assert_only_called() to verify which Redis methods were invoked.
   This prevents tests from silently allowing unexpected Redis calls.

3. **Use Provided Doubles**: Tests should use the provided test doubles from
   platform_workers.testing instead of defining custom subclasses:
   - FakeRedisNoPong: ping() returns False
   - FakeRedisError: ping() raises RedisError
   - FakeRedisNonRedisError: ping() raises non-Redis error (ValueError, etc.)
   - FakeRedisPublishError: publish() raises error

Violations:
- fake-redis-not-extended: Custom Redis stub doesn't extend FakeRedis
- fake-redis-no-assert: File uses FakeRedis but never calls assert_only_called()
- fake-redis-use-provided: Custom FakeRedis subclass duplicates a provided double
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

from monorepo_guards import Violation
from monorepo_guards.util import read_lines


class _FakeRedisVisitor(ast.NodeVisitor):
    """AST visitor to analyze FakeRedis usage patterns.

    Tracks imports, class definitions, and method calls to detect
    improper FakeRedis usage patterns.
    """

    # Redis protocol method names that indicate a Redis stub class
    _REDIS_METHODS: ClassVar[frozenset[str]] = frozenset(
        {
            "ping",
            "get",
            "set",
            "delete",
            "expire",
            "hset",
            "hget",
            "hgetall",
            "publish",
            "scard",
            "sadd",
            "sismember",
            "close",
        }
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        self.imports_fake_redis: bool = False
        self.imports_fake_redis_nopong: bool = False
        self.imports_fake_redis_error: bool = False
        self.imports_fake_redis_non_redis_error: bool = False
        self.imports_fake_redis_publish_error: bool = False
        self.has_assert_only_called: bool = False
        self.custom_stub_violations: list[Violation] = []
        self.duplicate_stub_violations: list[Violation] = []

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track imports of FakeRedis variants from platform_workers.testing."""
        if node.module == "platform_workers.testing":
            for alias in node.names:
                if alias.name == "FakeRedis":
                    self.imports_fake_redis = True
                elif alias.name == "FakeRedisNoPong":
                    self.imports_fake_redis_nopong = True
                elif alias.name == "FakeRedisError":
                    self.imports_fake_redis_error = True
                elif alias.name == "FakeRedisNonRedisError":
                    self.imports_fake_redis_non_redis_error = True
                elif alias.name == "FakeRedisPublishError":
                    self.imports_fake_redis_publish_error = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Track calls to assert_only_called()."""
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "assert_only_called":
            self.has_assert_only_called = True
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Check class definitions for improper Redis stub patterns."""
        if self._class_extends_fake_redis(node):
            self._check_duplicate_stubs(node)
            self.generic_visit(node)
            return

        if self._is_protocol_class(node):
            # Protocol classes are type hints, not implementations
            self.generic_visit(node)
            return

        redis_methods = self._get_redis_method_names(node)
        # If class defines 3+ Redis methods, it's likely a custom stub
        # (ping+close alone is too minimal to be a real Redis stub)
        if len(redis_methods) >= 3:
            methods_str = ", ".join(sorted(redis_methods))
            self.custom_stub_violations.append(
                Violation(
                    file=self.path,
                    line_no=node.lineno,
                    kind="fake-redis-not-extended",
                    line=f"class {node.name} defines [{methods_str}] but does not extend FakeRedis",
                )
            )
        self.generic_visit(node)

    def _class_extends_fake_redis(self, node: ast.ClassDef) -> bool:
        """Check if a class extends FakeRedis."""
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "FakeRedis":
                return True
            if isinstance(base, ast.Attribute) and base.attr == "FakeRedis":
                return True
        return False

    def _check_duplicate_stubs(self, node: ast.ClassDef) -> None:
        """Check if FakeRedis subclass duplicates a provided test double."""
        # Map of behavior -> suggested replacement
        ping_replacements = {
            "returns_false": ("ping() to return False", "FakeRedisNoPong"),
            "raises_redis_error": ("ping() to raise RedisError", "FakeRedisError"),
            "raises_non_redis_error": ("ping() to raise non-Redis error", "FakeRedisNonRedisError"),
        }
        scard_replacements = {
            "raises_redis_error": ("scard() to raise RedisError", "FakeRedisScardError"),
            "raises_non_redis_error": (
                "scard() to raise non-Redis error",
                "FakeRedisNonRedisScardError",
            ),
        }

        # Check ping
        ping_behavior = self._detect_ping_override_behavior(node)
        if ping_behavior in ping_replacements:
            desc, replacement = ping_replacements[ping_behavior]
            self._add_duplicate_violation(node, desc, replacement)

        # Check publish
        publish_behavior = self._detect_publish_override_behavior(node)
        if publish_behavior == "raises":
            self._add_duplicate_violation(node, "publish() to raise", "FakeRedisPublishError")

        # Check scard
        scard_behavior = self._detect_method_override_behavior(node, "scard")
        if scard_behavior in scard_replacements:
            desc, replacement = scard_replacements[scard_behavior]
            self._add_duplicate_violation(node, desc, replacement)

        # Check hset
        hset_behavior = self._detect_method_override_behavior(node, "hset")
        if hset_behavior and hset_behavior.startswith("raises"):
            self._add_duplicate_violation(node, "hset() to raise", "FakeRedisHsetError")

    def _add_duplicate_violation(self, node: ast.ClassDef, desc: str, replacement: str) -> None:
        """Add a duplicate stub violation."""
        self.duplicate_stub_violations.append(
            Violation(
                file=self.path,
                line_no=node.lineno,
                kind="fake-redis-use-provided",
                line=f"class {node.name} overrides {desc}; use {replacement} instead",
            )
        )

    def _is_protocol_class(self, node: ast.ClassDef) -> bool:
        """Check if a class extends Protocol (typing.Protocol)."""
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "Protocol":
                return True
            if isinstance(base, ast.Attribute) and base.attr == "Protocol":
                return True
        return False

    def _get_redis_method_names(self, node: ast.ClassDef) -> set[str]:
        """Get the set of Redis-like method names defined in a class."""
        methods: set[str] = set()
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name in self._REDIS_METHODS:
                methods.add(item.name)
            if isinstance(item, ast.AsyncFunctionDef) and item.name in self._REDIS_METHODS:
                methods.add(item.name)
        return methods

    def _detect_ping_override_behavior(self, node: ast.ClassDef) -> str | None:
        """Detect if a FakeRedis subclass overrides ping() with a common pattern.

        Returns:
            "returns_false" if ping() returns False
            "raises_redis_error" if ping() raises a RedisError
            "raises_non_redis_error" if ping() raises a non-Redis error
            None if ping() is not overridden or has a different behavior
        """
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "ping":
                return self._analyze_method_body(item)
        return None

    def _detect_publish_override_behavior(self, node: ast.ClassDef) -> str | None:
        """Detect if a FakeRedis subclass overrides publish() to raise."""
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "publish":
                result = self._analyze_method_body(item)
                # For publish, we just care if it raises (any error)
                if result and result.startswith("raises"):
                    return "raises"
        return None

    def _detect_method_override_behavior(self, node: ast.ClassDef, method_name: str) -> str | None:
        """Detect if a FakeRedis subclass overrides a method with a common pattern."""
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == method_name:
                return self._analyze_method_body(item)
        return None

    def _analyze_method_body(self, node: ast.FunctionDef) -> str | None:
        """Analyze the body of a method to detect common patterns."""
        # Look for return False or raise statements
        for stmt in ast.walk(node):
            if (
                isinstance(stmt, ast.Return)
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is False
            ):
                return "returns_false"
            if isinstance(stmt, ast.Raise):
                # Try to determine the exception type
                if self._is_redis_error_raise(stmt):
                    return "raises_redis_error"
                return "raises_non_redis_error"
        return None

    def _is_redis_error_raise(self, node: ast.Raise) -> bool:
        """Check if a raise statement raises a Redis error.

        Heuristic: If the exception is loaded via _load_redis_error_class() or
        has "Redis" or "redis" in the name, it's a Redis error.
        """
        if node.exc is None:
            return False

        # Check for pattern: error_cls(...) where error_cls = _load_redis_error_class()
        if isinstance(node.exc, ast.Call):
            func = node.exc.func
            if isinstance(func, ast.Name):
                # Heuristic: variable name contains "redis" or "error_cls" = Redis error
                name_lower = func.id.lower()
                if "redis" in name_lower or func.id in ("error_cls", "_ActualRedisError"):
                    return True
            if isinstance(func, ast.Attribute) and "redis" in func.attr.lower():
                return True
        return False


class FakeRedisRule:
    """Guard rule for enforcing proper FakeRedis usage in tests.

    This rule ensures that:
    1. Custom Redis stub classes extend FakeRedis to get call tracking
    2. Tests using FakeRedis call assert_only_called() to verify interactions

    Only test files (in tests/ directory with test_ prefix) are checked.
    """

    name = "fake-redis"

    def _is_test_file(self, path: Path) -> bool:
        """Check if path is a test file."""
        posix = path.as_posix()
        return ("/tests/" in posix or "\\tests\\" in str(path)) and path.name.startswith("test_")

    def run(self, files: list[Path]) -> list[Violation]:
        """Run the guard rule on the given files.

        Args:
            files: List of Python source file paths to check.

        Returns:
            List of violations found.

        Raises:
            RuntimeError: If a file cannot be parsed due to syntax errors.
        """
        violations: list[Violation] = []

        for path in files:
            if not self._is_test_file(path):
                continue

            lines = read_lines(path)
            source = "\n".join(lines)

            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError as exc:
                raise RuntimeError(f"failed to parse {path}: {exc}") from exc

            visitor = _FakeRedisVisitor(path)
            visitor.visit(tree)

            # Collect custom stub violations
            violations.extend(visitor.custom_stub_violations)

            # Collect duplicate stub violations
            violations.extend(visitor.duplicate_stub_violations)

            # Check for missing assert_only_called
            imports_any_fake = (
                visitor.imports_fake_redis
                or visitor.imports_fake_redis_nopong
                or visitor.imports_fake_redis_error
                or visitor.imports_fake_redis_non_redis_error
                or visitor.imports_fake_redis_publish_error
            )
            if imports_any_fake and not visitor.has_assert_only_called:
                violations.append(
                    Violation(
                        file=path,
                        line_no=1,
                        kind="fake-redis-no-assert",
                        line="File uses FakeRedis but does not call assert_only_called()",
                    )
                )

        return violations


__all__ = ["FakeRedisRule"]
