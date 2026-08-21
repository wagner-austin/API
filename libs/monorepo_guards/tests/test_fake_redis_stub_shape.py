"""FakeRedisRule stub-shape detection: not-extended, filtering, metadata.

The assert-discipline tests live in ``test_fake_redis_assert_discipline.py``;
the use-provided duplicate-stub tests in ``test_fake_redis_use_provided.py``.
"""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.fake_redis_rules import FakeRedisRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class TestFakeRedisNotExtended:
    """Tests for fake-redis-not-extended violation detection."""

    def test_flags_custom_stub_without_fake_redis_base(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
class CustomRedis:
    def ping(self) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return None

    def set(self, key: str, value: str) -> bool:
        return True
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "fake-redis-not-extended"
        assert "CustomRedis" in violations[0].line
        assert "get, ping, set" in violations[0].line  # Sorted methods

    def test_flags_stub_with_three_redis_methods(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
class BadStub:
    def ping(self) -> bool:
        return True

    def set(self, key: str, value: str) -> bool:
        return True

    def delete(self, key: str) -> int:
        return 1
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "fake-redis-not-extended"

    def test_allows_class_extending_fake_redis(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedis

class CustomRedis(FakeRedis):
    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        self._record("ping")
        return False

    def get(self, key: str) -> str | None:
        self._record("get", key)
        return None

def test_example() -> None:
    redis = CustomRedis()
    redis.ping()
    redis.assert_only_called({"ping"})
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        not_extended = [v for v in violations if v.kind == "fake-redis-not-extended"]
        assert len(not_extended) == 0

    def test_allows_class_with_only_one_redis_method(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
class SingleMethodStub:
    def ping(self) -> bool:
        return True
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        not_extended = [v for v in violations if v.kind == "fake-redis-not-extended"]
        assert len(not_extended) == 0

    def test_allows_class_with_two_redis_methods(self, tmp_path: Path) -> None:
        """Two methods (like ping+close) is too minimal to be a full Redis stub."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
class MinimalStub:
    def ping(self) -> bool:
        return True

    def close(self) -> None:
        pass
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        not_extended = [v for v in violations if v.kind == "fake-redis-not-extended"]
        assert len(not_extended) == 0

    def test_allows_protocol_class_with_redis_methods(self, tmp_path: Path) -> None:
        """Protocol classes are type hints, not implementations."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from typing import Protocol

class RedisProto(Protocol):
    def ping(self) -> bool: ...
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str) -> bool: ...
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        not_extended = [v for v in violations if v.kind == "fake-redis-not-extended"]
        assert len(not_extended) == 0

    def test_allows_protocol_class_with_attribute_style_base(self, tmp_path: Path) -> None:
        """Protocol classes with typing.Protocol base are type hints."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
import typing

class RedisProto(typing.Protocol):
    def ping(self) -> bool: ...
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str) -> bool: ...
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        not_extended = [v for v in violations if v.kind == "fake-redis-not-extended"]
        assert len(not_extended) == 0

    def test_flags_stub_with_set_operations(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
class SetStub:
    def scard(self, key: str) -> int:
        return 0

    def sadd(self, key: str, member: str) -> int:
        return 1

    def sismember(self, key: str, member: str) -> bool:
        return False
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "fake-redis-not-extended"

    def test_flags_stub_with_hash_methods(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
class HashStub:
    def hset(self, key: str, mapping: dict[str, str]) -> int:
        return 1

    def hget(self, key: str, field: str) -> str | None:
        return None

    def hgetall(self, key: str) -> dict[str, str]:
        return {}
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "fake-redis-not-extended"

    def test_allows_extending_via_attribute_import(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
import platform_workers.testing

class CustomRedis(platform_workers.testing.FakeRedis):
    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        self._record("ping")
        return False

    def get(self, key: str) -> str | None:
        self._record("get", key)
        return None
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        not_extended = [v for v in violations if v.kind == "fake-redis-not-extended"]
        assert len(not_extended) == 0

    def test_flags_stub_with_async_redis_methods(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
class AsyncRedisStub:
    async def ping(self) -> bool:
        return True

    async def get(self, key: str) -> str | None:
        return None

    async def set(self, key: str, value: str) -> bool:
        return True
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "fake-redis-not-extended"


class TestFileFiltering:
    """Tests for file filtering behavior."""

    def test_ignores_non_test_directory(self, tmp_path: Path) -> None:
        src_file = tmp_path / "src" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedis

def test_example() -> None:
    redis = FakeRedis()
    redis.ping()
"""
        _write(src_file, code)

        rule = FakeRedisRule()
        violations = rule.run([src_file])

        assert len(violations) == 0

    def test_ignores_non_test_prefix_file(self, tmp_path: Path) -> None:
        conftest = tmp_path / "tests" / "conftest.py"
        code = """
from platform_workers.testing import FakeRedis

def make_redis() -> FakeRedis:
    return FakeRedis()
"""
        _write(conftest, code)

        rule = FakeRedisRule()
        violations = rule.run([conftest])

        assert len(violations) == 0

    def test_ignores_helper_module_in_tests(self, tmp_path: Path) -> None:
        helper = tmp_path / "tests" / "helpers.py"
        code = """
from platform_workers.testing import FakeRedis

class TestHelper:
    def ping(self) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return None
"""
        _write(helper, code)

        rule = FakeRedisRule()
        violations = rule.run([helper])

        assert len(violations) == 0

    def test_checks_test_file_with_windows_path(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedis

def test_example() -> None:
    redis = FakeRedis()
    redis.ping()
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        no_assert = [v for v in violations if v.kind == "fake-redis-no-assert"]
        assert len(no_assert) == 1


class TestRuleName:
    """Tests for rule metadata."""

    def test_rule_name(self) -> None:
        rule = FakeRedisRule()
        assert rule.name == "fake-redis"


class TestBranchCoverage:
    """Additional tests for complete branch coverage."""

    def test_import_other_from_platform_workers_testing(self, tmp_path: Path) -> None:
        """Import non-FakeRedis item from platform_workers.testing."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeQueue, FakeJob

def test_example() -> None:
    queue = FakeQueue()
    assert queue is not None
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        # No FakeRedis import, so no violations
        assert len(violations) == 0

    def test_class_with_non_fake_redis_attribute_base(self, tmp_path: Path) -> None:
        """Class extends attribute-style base that is not FakeRedis."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
import some_module

class CustomStub(some_module.OtherBase):
    def ping(self) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return None

    def set(self, key: str, value: str) -> bool:
        return True
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        # Should flag as not extending FakeRedis
        assert len(violations) == 1
        assert violations[0].kind == "fake-redis-not-extended"

    def test_import_fake_redis_with_other_imports(self, tmp_path: Path) -> None:
        """Import FakeRedis along with other items from platform_workers.testing."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeQueue, FakeRedis, FakeJob

def test_example() -> None:
    redis = FakeRedis()
    redis.ping()
    redis.assert_only_called({"ping"})
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        # Has assert_only_called, no violations
        assert len(violations) == 0
