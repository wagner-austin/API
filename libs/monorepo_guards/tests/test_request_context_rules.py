from __future__ import annotations

from pathlib import Path

import pytest

from monorepo_guards.request_context_rules import RequestContextRule


def test_request_context_rule_flags_local_request_id(tmp_path: Path) -> None:
    middleware = tmp_path / "middleware.py"
    middleware.write_text(
        "class RequestIdMiddleware:\n    pass\n",
        encoding="utf-8",
    )
    rule = RequestContextRule()
    violations = rule.run([middleware])
    assert violations and violations[0].kind == "local-request-id-middleware"


def test_request_context_rule_ignores_other_classes(tmp_path: Path) -> None:
    middleware = tmp_path / "middleware.py"
    middleware.write_text(
        "class Other:\n    pass\n",
        encoding="utf-8",
    )
    rule = RequestContextRule()
    violations = rule.run([middleware])
    assert violations == []


def test_request_context_rule_skips_platform_core_path(tmp_path: Path) -> None:
    platform_path = tmp_path / "platform_core" / "request_context" / "middleware.py"
    platform_path.parent.mkdir(parents=True, exist_ok=True)
    platform_path.write_text(
        "class RequestIdMiddleware:\n    pass\n",
        encoding="utf-8",
    )
    rule = RequestContextRule()
    violations = rule.run([platform_path])
    assert violations == []


def test_request_context_rule_reports_line_content(tmp_path: Path) -> None:
    middleware = tmp_path / "middleware.py"
    middleware.write_text(
        "class RequestIdMiddleware:\n    sentinel = True\n",
        encoding="utf-8",
    )
    rule = RequestContextRule()
    violations = rule.run([middleware])
    assert violations and "RequestIdMiddleware" in violations[0].line


def test_request_context_rule_raises_on_invalid_syntax(tmp_path: Path) -> None:
    bad = tmp_path / "middleware.py"
    bad.write_text("class RequestIdMiddleware(:\n    pass\n", encoding="utf-8")
    rule = RequestContextRule()
    with pytest.raises(RuntimeError):
        rule.run([bad])
