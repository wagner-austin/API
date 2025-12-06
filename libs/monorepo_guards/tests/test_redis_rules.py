from __future__ import annotations

from pathlib import Path

from monorepo_guards.redis_rules import RedisRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_redis_rule_flags_protocol_duplicates(tmp_path: Path) -> None:
    bad = tmp_path / "services" / "demo" / "redis_proto.py"
    good = tmp_path / "libs" / "platform_workers" / "src" / "platform_workers" / "redis.py"
    _write(
        bad,
        "from typing import Protocol\n"
        "class RedisThing(Protocol):\n"
        "    def ping(self) -> bool: ...\n",
    )
    _write(
        good,
        "from typing import Protocol\n"
        "class RedisProto(Protocol):\n"
        "    def ping(self) -> bool: ...\n",
    )

    rule = RedisRule()
    violations = rule.run([bad, good])
    assert len(violations) == 1
    v = violations[0]
    assert v.kind == "redis-protocol-duplicate"
    assert v.file == bad


def test_redis_rule_allows_non_protocols(tmp_path: Path) -> None:
    ok = tmp_path / "src" / "other.py"
    _write(ok, "class RedisHelper:\n    pass\n")
    rule = RedisRule()
    assert rule.run([ok]) == []
