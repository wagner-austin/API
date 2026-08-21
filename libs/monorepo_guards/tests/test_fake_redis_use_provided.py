"""FakeRedisRule use-provided detection: duplicate stubs of shared doubles.

The stub-shape tests live in ``test_fake_redis_stub_shape.py``.
"""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.fake_redis_rules import FakeRedisRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
