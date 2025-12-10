from __future__ import annotations

from pathlib import Path

from monorepo_guards.config import GuardConfig
from monorepo_guards.pattern_rules import PatternRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_pattern_rule_flags_drifts_and_noqa(tmp_path: Path) -> None:
    root = tmp_path
    src = root / "src"
    path = src / "mod.py"
    # Obfuscate tokens to avoid triggering repo-wide PatternRule when scanning tests themselves
    token_todo = "TO" + "DO"
    token_fixme = "FIX" + "ME"
    token_hack = "HA" + "CK"
    token_xxx = "X" + "XX"
    token_wip = "WI" + "P"
    token_noqa = "no" + "qa"
    code = (
        "def f() -> None:\n"
        f"    # {token_todo}: refactor\n"
        f"    # {token_fixme} something\n"
        f"    # {token_hack}\n"
        f"    # {token_xxx}\n"
        f"    # {token_wip}\n"
        f"    x = 1  # {token_noqa}\n"
    )
    _write(path, code)

    cfg = GuardConfig(
        root=root,
        directories=("src",),
        exclude_parts=(".venv", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"),
        forbid_pyi=True,
        allow_print_in_tests=False,
        dataclass_ban_segments=(),
    )
    rule = PatternRule(cfg)
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert f"pattern-{token_todo}" in kinds
    assert f"pattern-{token_fixme}" in kinds
    assert f"pattern-{token_hack}" in kinds
    assert f"pattern-{token_xxx}" in kinds
    assert f"pattern-{token_wip}" in kinds
    assert f"pattern-{token_noqa}" in kinds


def test_pattern_rule_flags_pyi_when_forbidden(tmp_path: Path) -> None:
    root = tmp_path
    src = root / "src"
    path = src / "mod.pyi"
    _write(path, "from __future__ import annotations\n")

    cfg = GuardConfig(
        root=root,
        directories=("src",),
        exclude_parts=(".venv", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"),
        forbid_pyi=True,
        allow_print_in_tests=False,
        dataclass_ban_segments=(),
    )
    rule = PatternRule(cfg)
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "pyi-disallowed" in kinds
