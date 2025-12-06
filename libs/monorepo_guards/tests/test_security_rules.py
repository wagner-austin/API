from __future__ import annotations

from pathlib import Path

from monorepo_guards.security_rules import SecurityRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_security_rule_flags_custom_api_key_middleware(tmp_path: Path) -> None:
    bad = tmp_path / "services" / "demo" / "src" / "middleware.py"
    _write(
        bad,
        """\
from starlette.middleware.base import BaseHTTPMiddleware

class APIKeyMiddleware(BaseHTTPMiddleware):
    pass
""",
    )

    rule = SecurityRule()
    violations = rule.run([bad])
    assert len(violations) == 2  # Both custom-api-key-middleware and auth-middleware-not-dependency
    kinds = {v.kind for v in violations}
    assert "custom-api-key-middleware" in kinds


def test_security_rule_flags_auth_middleware_subclass(tmp_path: Path) -> None:
    bad = tmp_path / "services" / "demo" / "src" / "auth.py"
    _write(
        bad,
        """\
from starlette.middleware.base import BaseHTTPMiddleware

class AuthMiddleware(BaseHTTPMiddleware):
    pass
""",
    )

    rule = SecurityRule()
    violations = rule.run([bad])
    assert len(violations) == 1
    assert violations[0].kind == "auth-middleware-not-dependency"


def test_security_rule_allows_test_files(tmp_path: Path) -> None:
    test_file = tmp_path / "services" / "demo" / "tests" / "test_auth.py"
    _write(
        test_file,
        """\
class APIKeyMiddleware:
    pass
""",
    )

    rule = SecurityRule()
    violations = rule.run([test_file])
    assert len(violations) == 0


def test_security_rule_allows_platform_core_security(tmp_path: Path) -> None:
    canonical = tmp_path / "libs" / "platform_core" / "src" / "platform_core" / "security.py"
    _write(
        canonical,
        """\
class APIKeyMiddleware:
    pass
""",
    )

    rule = SecurityRule()
    violations = rule.run([canonical])
    assert len(violations) == 0


def test_security_rule_allows_unrelated_middleware(tmp_path: Path) -> None:
    ok = tmp_path / "services" / "demo" / "src" / "middleware.py"
    _write(
        ok,
        """\
from starlette.middleware.base import BaseHTTPMiddleware

class LoggingMiddleware(BaseHTTPMiddleware):
    pass
""",
    )

    rule = SecurityRule()
    violations = rule.run([ok])
    assert len(violations) == 0


def test_security_rule_allows_regular_classes(tmp_path: Path) -> None:
    ok = tmp_path / "services" / "demo" / "src" / "models.py"
    _write(
        ok,
        """\
class UserModel:
    pass

class APIClient:
    pass
""",
    )

    rule = SecurityRule()
    violations = rule.run([ok])
    assert len(violations) == 0


def test_security_rule_raises_on_syntax_error(tmp_path: Path) -> None:
    bad = tmp_path / "services" / "demo" / "src" / "broken.py"
    _write(bad, "class Foo(\n")  # Invalid syntax

    rule = SecurityRule()
    import pytest

    with pytest.raises(RuntimeError, match="failed to parse"):
        rule.run([bad])
