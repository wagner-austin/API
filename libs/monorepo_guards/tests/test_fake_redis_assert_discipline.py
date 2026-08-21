"""FakeRedisRule assert discipline: no-assert, edge cases, combined files.

The stub-shape tests live in ``test_fake_redis_stub_shape.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from monorepo_guards.fake_redis_rules import FakeRedisRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
