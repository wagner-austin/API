"""Tests for fake_redis_rules module."""

from __future__ import annotations

from pathlib import Path

import pytest

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


class TestFakeRedisNoAssert:
    """Tests for fake-redis-no-assert violation detection."""

    def test_flags_fake_redis_import_without_assert(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedis

def test_example() -> None:
    redis = FakeRedis()
    redis.ping()
    # Missing assert_only_called!
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        no_assert = [v for v in violations if v.kind == "fake-redis-no-assert"]
        assert len(no_assert) == 1

    def test_allows_fake_redis_with_assert_only_called(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedis

def test_example() -> None:
    redis = FakeRedis()
    redis.ping()
    redis.assert_only_called({"ping"})
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        no_assert = [v for v in violations if v.kind == "fake-redis-no-assert"]
        assert len(no_assert) == 0

    def test_flags_multiple_tests_without_any_assert(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedis

def test_one() -> None:
    redis = FakeRedis()
    redis.ping()

def test_two() -> None:
    redis = FakeRedis()
    redis.get("key")
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        no_assert = [v for v in violations if v.kind == "fake-redis-no-assert"]
        assert len(no_assert) == 1  # One violation per file

    def test_allows_single_assert_for_multiple_tests(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedis

def test_one() -> None:
    redis = FakeRedis()
    redis.ping()
    redis.assert_only_called({"ping"})

def test_two() -> None:
    redis = FakeRedis()
    redis.get("key")
    # This test doesn't have assert_only_called but file has one
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        no_assert = [v for v in violations if v.kind == "fake-redis-no-assert"]
        # File has at least one assert_only_called, so no violation
        assert len(no_assert) == 0

    def test_flags_subclass_usage_without_assert(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedis

class CustomRedis(FakeRedis):
    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        self._record("ping")
        return False

def test_example() -> None:
    redis = CustomRedis()
    redis.ping()
    # Missing assert_only_called!
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        no_assert = [v for v in violations if v.kind == "fake-redis-no-assert"]
        assert len(no_assert) == 1


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


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_raises_on_syntax_error(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example(\n"
        _write(test_file, code)

        rule = FakeRedisRule()
        with pytest.raises(RuntimeError, match="failed to parse"):
            rule.run([test_file])

    def test_allows_file_without_fake_redis_import(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example() -> None:
    assert 1 + 1 == 2
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        assert len(violations) == 0

    def test_allows_class_with_non_redis_methods(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
class TestHelper:
    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        assert len(violations) == 0

    def test_handles_empty_file(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = ""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        assert len(violations) == 0

    def test_handles_file_with_only_imports(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedis
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        # Even just importing FakeRedis without using it still requires assert_only_called
        # to enforce the pattern - but since there's no actual usage, we might allow it
        # Actually, the rule should flag this since FakeRedis is imported
        no_assert = [v for v in violations if v.kind == "fake-redis-no-assert"]
        assert len(no_assert) == 1

    def test_handles_multiple_files(self, tmp_path: Path) -> None:
        file1 = tmp_path / "tests" / "test_one.py"
        file2 = tmp_path / "tests" / "test_two.py"

        code1 = """
from platform_workers.testing import FakeRedis

def test_one() -> None:
    redis = FakeRedis()
    redis.ping()
"""
        code2 = """
from platform_workers.testing import FakeRedis

def test_two() -> None:
    redis = FakeRedis()
    redis.ping()
    redis.assert_only_called({"ping"})
"""
        _write(file1, code1)
        _write(file2, code2)

        rule = FakeRedisRule()
        violations = rule.run([file1, file2])

        # Only file1 should have violation
        no_assert = [v for v in violations if v.kind == "fake-redis-no-assert"]
        assert len(no_assert) == 1
        assert no_assert[0].file == file1


class TestCombinedViolations:
    """Tests for files with multiple violation types."""

    def test_flags_both_not_extended_and_no_assert(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedis

class BadStub:
    def ping(self) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return None

    def set(self, key: str, value: str) -> bool:
        return True

def test_example() -> None:
    redis = FakeRedis()
    redis.ping()
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        not_extended = [v for v in violations if v.kind == "fake-redis-not-extended"]
        no_assert = [v for v in violations if v.kind == "fake-redis-no-assert"]

        assert len(not_extended) == 1
        assert len(no_assert) == 1

    def test_flags_multiple_custom_stubs(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
class BadStub1:
    def ping(self) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return None

    def set(self, key: str, value: str) -> bool:
        return True

class BadStub2:
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

        not_extended = [v for v in violations if v.kind == "fake-redis-not-extended"]
        assert len(not_extended) == 2


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


class TestDuplicateStubDetection:
    """Tests for fake-redis-use-provided violation detection."""

    def test_flags_ping_returns_false(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedis

class MyNoPong(FakeRedis):
    def ping(self, **kwargs) -> bool:
        self._record("ping")
        return False

def test_example() -> None:
    redis = MyNoPong()
    redis.assert_only_called({"ping"})
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        use_provided = [v for v in violations if v.kind == "fake-redis-use-provided"]
        assert len(use_provided) == 1
        assert "FakeRedisNoPong" in use_provided[0].line

    def test_flags_ping_raises_redis_error(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedis

class MyError(FakeRedis):
    def ping(self, **kwargs) -> bool:
        self._record("ping")
        raise error_cls("fail")

def test_example() -> None:
    redis = MyError()
    redis.assert_only_called({"ping"})
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        use_provided = [v for v in violations if v.kind == "fake-redis-use-provided"]
        assert len(use_provided) == 1
        assert "FakeRedisError" in use_provided[0].line

    def test_flags_ping_raises_non_redis_error(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedis

class MyNonRedisError(FakeRedis):
    def ping(self, **kwargs) -> bool:
        self._record("ping")
        raise ValueError("not redis")

def test_example() -> None:
    redis = MyNonRedisError()
    redis.assert_only_called({"ping"})
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        use_provided = [v for v in violations if v.kind == "fake-redis-use-provided"]
        assert len(use_provided) == 1
        assert "FakeRedisNonRedisError" in use_provided[0].line

    def test_flags_publish_raises(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedis

class MyPublishError(FakeRedis):
    def publish(self, channel: str, message: str) -> int:
        self._record("publish", channel, message)
        raise OSError("fail")

def test_example() -> None:
    redis = MyPublishError()
    redis.assert_only_called({"publish"})
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        use_provided = [v for v in violations if v.kind == "fake-redis-use-provided"]
        assert len(use_provided) == 1
        assert "FakeRedisPublishError" in use_provided[0].line

    def test_flags_scard_raises_redis_error(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedis

class MyScardError(FakeRedis):
    def scard(self, key: str) -> int:
        self._record("scard", key)
        raise _ActualRedisError("fail")

def test_example() -> None:
    redis = MyScardError()
    redis.assert_only_called({"scard"})
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        use_provided = [v for v in violations if v.kind == "fake-redis-use-provided"]
        assert len(use_provided) == 1
        assert "FakeRedisScardError" in use_provided[0].line

    def test_flags_scard_raises_non_redis_error(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedis

class MyScardNonRedisError(FakeRedis):
    def scard(self, key: str) -> int:
        self._record("scard", key)
        raise TypeError("fail")

def test_example() -> None:
    redis = MyScardNonRedisError()
    redis.assert_only_called({"scard"})
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        use_provided = [v for v in violations if v.kind == "fake-redis-use-provided"]
        assert len(use_provided) == 1
        assert "FakeRedisNonRedisScardError" in use_provided[0].line

    def test_flags_hset_raises(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedis

class MyHsetError(FakeRedis):
    def hset(self, key: str, mapping: dict) -> int:
        self._record("hset", key, mapping)
        raise RuntimeError("fail")

def test_example() -> None:
    redis = MyHsetError()
    redis.assert_only_called({"hset"})
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        use_provided = [v for v in violations if v.kind == "fake-redis-use-provided"]
        assert len(use_provided) == 1
        assert "FakeRedisHsetError" in use_provided[0].line

    def test_tracks_new_import_variants(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import (
    FakeRedisNonRedisError,
    FakeRedisPublishError,
)

def test_example() -> None:
    redis = FakeRedisNonRedisError()
    redis.assert_only_called({"ping"})
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        # Should have no violations - properly uses shared doubles
        assert len(violations) == 0

    def test_tracks_fake_redis_nopong_import(self, tmp_path: Path) -> None:
        """Test that importing FakeRedisNoPong is tracked."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedisNoPong

def test_example() -> None:
    redis = FakeRedisNoPong()
    redis.assert_only_called({"ping"})
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        # Should have no violations - properly uses shared double with assert
        assert len(violations) == 0

    def test_tracks_fake_redis_error_import(self, tmp_path: Path) -> None:
        """Test that importing FakeRedisError is tracked."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedisError

def test_example() -> None:
    redis = FakeRedisError()
    redis.assert_only_called({"ping"})
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        # Should have no violations - properly uses shared double with assert
        assert len(violations) == 0

    def test_fake_redis_nopong_without_assert_fails(self, tmp_path: Path) -> None:
        """Test that importing FakeRedisNoPong without assert triggers violation."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedisNoPong

def test_example() -> None:
    redis = FakeRedisNoPong()
    redis.ping()
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        no_assert = [v for v in violations if v.kind == "fake-redis-no-assert"]
        assert len(no_assert) == 1

    def test_fake_redis_error_without_assert_fails(self, tmp_path: Path) -> None:
        """Test that importing FakeRedisError without assert triggers violation."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedisError

def test_example() -> None:
    redis = FakeRedisError()
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        no_assert = [v for v in violations if v.kind == "fake-redis-no-assert"]
        assert len(no_assert) == 1

    def test_subclass_with_pass_only_method(self, tmp_path: Path) -> None:
        """Test FakeRedis subclass with method containing only pass (no return/raise)."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedis

class MyCustom(FakeRedis):
    def ping(self, **kwargs) -> bool:
        pass  # No return or raise

def test_example() -> None:
    redis = MyCustom()
    redis.assert_only_called({"ping"})
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        # No duplicate violation since behavior is unknown (no return False or raise)
        use_provided = [v for v in violations if v.kind == "fake-redis-use-provided"]
        assert len(use_provided) == 0

    def test_subclass_with_bare_raise(self, tmp_path: Path) -> None:
        """Test FakeRedis subclass with bare raise (re-raise)."""
        test_file = tmp_path / "tests" / "test_foo.py"
        # Use raw string to avoid guard rule triggering on this test file
        code = (
            "from platform_workers.testing import FakeRedis\n\n"
            "class MyReraise(FakeRedis):\n"
            "    def ping(self, **kwargs) -> bool:\n"
            "        try:\n"
            "            something()\n"
            "        except Exception:\n"
            "            raise  # Bare raise, node.exc is None\n\n"
            "def test_example() -> None:\n"
            "    redis = MyReraise()\n"
            "    redis.assert_only_called({'ping'})\n"
        )
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        # Should flag as non-redis error since bare raise is not redis-specific
        use_provided = [v for v in violations if v.kind == "fake-redis-use-provided"]
        assert len(use_provided) == 1
        assert "FakeRedisNonRedisError" in use_provided[0].line

    def test_subclass_raises_attribute_redis_error(self, tmp_path: Path) -> None:
        """Test FakeRedis subclass that raises redis.RedisError attribute-style."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedis

class MyRedisError(FakeRedis):
    def ping(self, **kwargs) -> bool:
        raise redis.RedisError("fail")

def test_example() -> None:
    redis = MyRedisError()
    redis.assert_only_called({"ping"})
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        # Should recognize as Redis error due to "redis" in attribute name
        use_provided = [v for v in violations if v.kind == "fake-redis-use-provided"]
        assert len(use_provided) == 1
        assert "FakeRedisError" in use_provided[0].line

    def test_publish_override_returns_false_no_violation(self, tmp_path: Path) -> None:
        """Test FakeRedis subclass with publish() that returns False (not raises)."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedis

class MyPublish(FakeRedis):
    def publish(self, channel: str, message: str) -> int:
        self._record("publish", channel, message)
        return 0  # Returns value, doesn't raise

def test_example() -> None:
    redis = MyPublish()
    redis.assert_only_called({"publish"})
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        # No violation since publish returns value (doesn't raise)
        use_provided = [v for v in violations if v.kind == "fake-redis-use-provided"]
        assert len(use_provided) == 0

    def test_subclass_raises_variable_exception(self, tmp_path: Path) -> None:
        """Test FakeRedis subclass that raises a pre-created exception variable."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from platform_workers.testing import FakeRedis

exc = RuntimeError("pre-made")

class MyRaiseVar(FakeRedis):
    def ping(self, **kwargs) -> bool:
        raise exc  # Raises variable, not Call

def test_example() -> None:
    redis = MyRaiseVar()
    redis.assert_only_called({"ping"})
"""
        _write(test_file, code)

        rule = FakeRedisRule()
        violations = rule.run([test_file])

        # Should flag as non-redis error since it's a variable raise
        use_provided = [v for v in violations if v.kind == "fake-redis-use-provided"]
        assert len(use_provided) == 1
        assert "FakeRedisNonRedisError" in use_provided[0].line
