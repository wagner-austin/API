from __future__ import annotations

from pathlib import Path

from monorepo_guards.worker_imports_rules import WorkerImportsRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_worker_imports_rule_flags_direct_redis_import(tmp_path: Path) -> None:
    bad = tmp_path / "services" / "demo" / "src" / "worker.py"
    _write(bad, "import redis\n\ndef connect(): pass\n")

    rule = WorkerImportsRule()
    violations = rule.run([bad])
    assert len(violations) == 1
    assert violations[0].kind == "direct-redis-import"


def test_worker_imports_rule_flags_from_redis_import(tmp_path: Path) -> None:
    bad = tmp_path / "services" / "demo" / "src" / "deps.py"
    _write(bad, "from redis import Redis\n\ndef connect(): pass\n")

    rule = WorkerImportsRule()
    violations = rule.run([bad])
    assert len(violations) == 1
    assert violations[0].kind == "direct-redis-import"


def test_worker_imports_rule_flags_redis_exceptions_import(tmp_path: Path) -> None:
    bad = tmp_path / "services" / "demo" / "src" / "health.py"
    _write(bad, "from redis import exceptions\n\ndef check(): pass\n")

    rule = WorkerImportsRule()
    violations = rule.run([bad])
    assert len(violations) == 1
    assert violations[0].kind == "direct-redis-import"


def test_worker_imports_rule_flags_direct_rq_import(tmp_path: Path) -> None:
    bad = tmp_path / "services" / "demo" / "src" / "jobs.py"
    _write(bad, "import rq\n\ndef enqueue(): pass\n")

    rule = WorkerImportsRule()
    violations = rule.run([bad])
    assert len(violations) == 1
    assert violations[0].kind == "direct-rq-import"


def test_worker_imports_rule_flags_from_rq_import(tmp_path: Path) -> None:
    bad = tmp_path / "services" / "demo" / "src" / "tasks.py"
    _write(bad, "from rq import get_current_job\n\ndef run(): pass\n")

    rule = WorkerImportsRule()
    violations = rule.run([bad])
    assert len(violations) == 1
    assert violations[0].kind == "direct-rq-import"


def test_worker_imports_rule_flags_canonical_redis_path(tmp_path: Path) -> None:
    """No path is exempt: platform_workers' own redis.py is checked like any other.

    The wrapper reaches the third-party package through __import__, which is a
    call rather than an import statement, so it never needed an exemption.
    """
    canonical = tmp_path / "libs" / "platform_workers" / "src" / "platform_workers" / "redis.py"
    _write(canonical, "import redis\n\nclass Client: pass\n")

    kinds = {violation.kind for violation in WorkerImportsRule().run([canonical])}

    assert "direct-redis-import" in kinds


def test_worker_imports_rule_flags_canonical_rq_path(tmp_path: Path) -> None:
    """The same holds for rq_harness.py."""
    canonical = (
        tmp_path / "libs" / "platform_workers" / "src" / "platform_workers" / "rq_harness.py"
    )
    _write(canonical, "import rq\n\nclass Queue: pass\n")

    kinds = {violation.kind for violation in WorkerImportsRule().run([canonical])}

    assert "direct-rq-import" in kinds


def test_worker_imports_rule_allows_test_files(tmp_path: Path) -> None:
    test_file = tmp_path / "services" / "demo" / "tests" / "test_worker.py"
    _write(test_file, "import redis\nimport rq\n\ndef test_thing(): pass\n")

    rule = WorkerImportsRule()
    violations = rule.run([test_file])
    assert len(violations) == 0


def test_worker_imports_rule_allows_platform_workers_import(tmp_path: Path) -> None:
    ok = tmp_path / "services" / "demo" / "src" / "worker.py"
    _write(
        ok,
        """\
from platform_workers.redis import redis_for_kv
from platform_workers.rq_harness import rq_queue

def connect(): pass
""",
    )

    rule = WorkerImportsRule()
    violations = rule.run([ok])
    assert len(violations) == 0


def test_worker_imports_rule_ignores_relative_imports(tmp_path: Path) -> None:
    ok = tmp_path / "services" / "demo" / "src" / "utils.py"
    _write(ok, "from . import helpers\nfrom .. import config\n")

    rule = WorkerImportsRule()
    violations = rule.run([ok])
    assert len(violations) == 0


def test_worker_imports_rule_raises_on_syntax_error(tmp_path: Path) -> None:
    bad = tmp_path / "services" / "demo" / "src" / "broken.py"
    _write(bad, "import redis\nimport (\n")  # Invalid syntax

    rule = WorkerImportsRule()
    import pytest

    with pytest.raises(RuntimeError, match="failed to parse"):
        rule.run([bad])


def test_relative_redis_import_is_not_a_direct_import(tmp_path: Path) -> None:
    """A relative "from .redis import" names a sibling module, not the package.

    ast records module == "redis" with level == 1 for that form, so a rule that
    ignores level reports platform_workers' own wrapper as a direct import.
    """
    path = tmp_path / "src" / "pkg" / "fakes.py"
    path.parent.mkdir(parents=True)
    path.write_text("from .redis import RedisStrProto\n", encoding="utf-8")

    assert WorkerImportsRule().run([path]) == []


def test_relative_rq_import_is_not_a_direct_import(tmp_path: Path) -> None:
    """The same holds for a relative import of a sibling rq module."""
    path = tmp_path / "src" / "pkg" / "harness.py"
    path.parent.mkdir(parents=True)
    path.write_text("from .rq_harness import RQJobLike\n", encoding="utf-8")

    assert WorkerImportsRule().run([path]) == []


def test_absolute_redis_import_is_still_flagged_in_platform_workers(tmp_path: Path) -> None:
    """Removing the canonical allowlist means no file is exempt from the rule."""
    path = tmp_path / "libs" / "platform_workers" / "src" / "platform_workers" / "redis.py"
    path.parent.mkdir(parents=True)
    path.write_text("from redis import Redis\n", encoding="utf-8")

    kinds = {violation.kind for violation in WorkerImportsRule().run([path])}

    assert "direct-redis-import" in kinds
